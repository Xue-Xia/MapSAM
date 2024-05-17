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
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic

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


def focal_loss(prediction, target, alpha=0.25, gamma=2, smooth=1e-6):
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(prediction)

    # Compute the binary cross-entropy loss
    bce_loss = F.binary_cross_entropy_with_logits(prediction, target.float(), reduction='none')

    # Compute the modulating factor (1 - p_t)^gamma
    modulating_factor = (1 - probs).pow(gamma)

    # Compute the focal loss
    focal_loss = alpha * modulating_factor * bce_loss

    # Smooth the loss to avoid numerical instability
    smooth_loss = -torch.log(1.0 - bce_loss + smooth)

    # Weighted combination of focal loss and smooth loss
    loss = torch.mean(focal_loss + smooth_loss)

    return loss

def calc_loss(outputs, coarse_mask, low_res_label_batch, ce_loss, dice_loss, bce_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)

    loss_hint = bce_loss(torch.sigmoid(coarse_mask), low_res_label_batch.unsqueeze(1).float())

    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice + loss_hint
    return loss, loss_ce, loss_dice, loss_hint


def trainer_synapse(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_hm import hm_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = hm_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
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
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes+1)
    bce_loss = nn.BCELoss()

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
            loss, loss_ce, loss_dice,loss_hint = calc_loss(outputs, coarse_mask, low_res_label_batch, ce_loss, dice_loss, bce_loss, args.dice_param)

            # dice_weight = args.dice_param
            #
            # input = outputs['low_res_logits']
            # target = low_res_label_batch.unsqueeze(1)
            #
            # loss_focal = focal_loss(input, target)
            # loss_dice = dice_loss(input, target)
            # loss = (1 - dice_weight) * loss_focal + dice_weight * loss_dice

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
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_hint', loss_hint, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_hint: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_hint.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 20 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
