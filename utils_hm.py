import os
import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn
import torch.nn.functional as F
import imageio
from einops import repeat
import cv2


def calculate_metric(pred_array, gt_array):
    # Flatten arrays
    gt_array = gt_array.flatten()
    pred_array = pred_array.flatten()

    # Calculate True Positive (TP), False Positive (FP), False Negative (FN)
    tp = np.sum((gt_array == 1) & (pred_array == 1))
    fp = np.sum((gt_array == 0) & (pred_array == 1))
    fn = np.sum((gt_array == 1) & (pred_array == 0))
    tn = np.sum((gt_array == 0) & (pred_array == 0))  # True Negative (TN), for completeness

    # Calculate precision, recall, F1 score, IoU
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return p, r, f1, iou

def test_single_volume(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        x, y = image.shape[1], image.shape[2]
        if x != input_size[0] or y != input_size[1]:
            image = zoom(image, (1, input_size[0] / x, input_size[1] / y), order=3)  # previous using 0
        new_x, new_y = image.shape[1], image.shape[2]  # [input_size[0], input_size[1]]
        if new_x != patch_size[0] or new_y != patch_size[1]:
            image = zoom(image, (1, patch_size[0] / new_x, patch_size[1] / new_y), order=3)  # previous using 0, patch_size[0], patch_size[1]
        inputs = torch.from_numpy(image).unsqueeze(0).float().cuda()

        net.eval()
        with torch.no_grad():
            outputs, coarse_mask = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            # out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            out = (torch.sigmoid(output_masks).squeeze() > 0.5).int()
            out = out.cpu().detach().numpy()
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction = pred

    else:
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric(prediction, label))

    if test_save_path is not None:
        prediction_uint8 = (prediction * 255).astype(np.uint8)
        label_uint8 = (label * 255).astype(np.uint8)
        cv2.imwrite(test_save_path + '/' + case + "_pred.png", prediction_uint8)
        cv2.imwrite(test_save_path + '/' + case + "_gt.png", label_uint8)

    return metric_list
