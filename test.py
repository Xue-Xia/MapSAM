import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils_hm import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_hm import hm_dataset


def inference(args, multimask_output, db_config, model, test_save_path=None):
    db_test = db_config['Dataset'](base_dir=args.root_path, split='test')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch['image'].shape[2:]
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case=case_name)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s p %f r %f f1 %f iou %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes + 1):
        try:
            logging.info('Mean class %d name %s p %f r %f f1 %f iou %f' % (i, class_to_name[i], metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2], metric_list[i - 1][3]))
        except:
            logging.info('Mean class %d p %f r %f f1 %f iou %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2], metric_list[i - 1][3]))
    p = np.mean(metric_list, axis=0)[0]
    r = np.mean(metric_list, axis=0)[1]
    f1 = np.mean(metric_list, axis=0)[2]
    iou = np.mean(metric_list, axis=0)[3]
    logging.info('Testing performance in best val model: p : %f r : %f f1 : %f iou : %f' % (p, r, f1, iou))
    logging.info("Testing Finished!")
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--root_path', type=str, default='data/dataset_rail/')
    parser.add_argument('--dataset', type=str, default='Siegfried', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='/output')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--dora_ckpt', type=str, default='checkpoints/epoch_159.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_dora_image_encoder')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Siegfried': {
            'Dataset': hm_dataset,
            'root_path': args.root_path,
            'num_classes': args.num_classes
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    net = pkg.DoRA_Sam(sam, args.rank).cuda()

    assert args.dora_ckpt is not None
    net.load_dora_parameters(args.dora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    test_save_path = os.path.join(args.output_dir, 'predictions')
    os.makedirs(test_save_path, exist_ok=True)

    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path)
