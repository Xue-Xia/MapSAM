import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms

def dice_loss(logits, target, smooth=1e-6):
    # Convert logits to probabilities using sigmoid
    probs = torch.sigmoid(logits)

    # Flatten logits and target tensors
    flat_logits = probs.view(logits.size(0), -1)
    flat_target = target.view(target.size(0), -1)

    # Compute intersection and sums
    intersection = torch.sum(flat_logits * flat_target, dim=1)
    sum_logits = torch.sum(flat_logits, dim=1)
    sum_target = torch.sum(flat_target, dim=1)

    # Compute Dice coefficient
    dice_coeff = (2. * intersection + smooth) / (sum_logits + sum_target + smooth)

    # Compute Dice loss
    dice_loss = 1. - dice_coeff

    # Average Dice loss over the batch
    return torch.mean(dice_loss)


def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
        Returns:
            Loss tensor
        """
    inputs = inputs.flatten(1)
    prob = inputs.sigmoid()
    targets = targets.float().flatten(1)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).mean()


def trainer_hm(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_hm import hm_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = hm_dataset(base_dir=args.root_path, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            outputs, coarse_mask = model(image_batch, multimask_output, args.img_size)

            dice_weight = args.dice_param

            input = outputs['low_res_logits']
            target = low_res_label_batch.unsqueeze(1)

            loss_focal = focal_loss(input, target)
            loss_dice = dice_loss(input, target)
            loss_focal_hint = focal_loss(coarse_mask, target)
            loss_dice_hint = dice_loss(coarse_mask, target)

            loss = (1 - dice_weight) * loss_focal + dice_weight * loss_dice + (1 - dice_weight) * loss_focal_hint + dice_weight * loss_dice_hint

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_focal', loss_focal, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_focal_hint', loss_focal_hint, iter_num)
            writer.add_scalar('info/loss_dice_hint', loss_dice_hint, iter_num)

            logging.info('iteration %d : loss : %f, loss_focal: %f, loss_dice: %f, loss_focal_hint: %f, loss_dice_hint: %f' % (iter_num, loss.item(), loss_focal.item(), loss_dice.item(), loss_focal_hint.item(), loss_dice_hint.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 80 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_dora_parameters(save_mode_path)
            except:
                model.module.save_dora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_dora_parameters(save_mode_path)
            except:
                model.module.save_dora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
