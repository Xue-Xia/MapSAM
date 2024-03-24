import os
import numpy as np
import SimpleITK as sitk

def read_nifti(file_path):
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)
    return array

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_iou_per_class(pred_mask, gt_mask, num_classes):
    iou_per_class = []
    for class_id in range(num_classes):
        pred_class = pred_mask == class_id + 1
        gt_class = gt_mask == class_id + 1
        intersection = np.logical_and(pred_class, gt_class)
        union = np.logical_or(pred_class, gt_class)
        union_sum = np.sum(union)
        if union_sum == 0:
            iou = 0  # If the union is zero, set IoU to 0
        else:
            iou = np.sum(intersection) / union_sum
        iou_per_class.append(iou)
    return iou_per_class


folder_path = "result/predictions_partial10"
files = os.listdir(folder_path)
num_classes = 3
total_iou_per_class = [0] * num_classes
count_per_class = [0] * num_classes


iou_values = []
for file_name in files:
    if file_name.endswith("_gt.nii.gz"):
        gt_file_path = os.path.join(folder_path, file_name)
        pred_file_path = os.path.join(folder_path, file_name.replace("_gt.nii.gz", "_pred.nii.gz"))

        # Read the NIfTI files
        gt_mask = read_nifti(gt_file_path)
        pred_mask = read_nifti(pred_file_path)

        # Compute IoU per class
        iou_per_class = calculate_iou_per_class(pred_mask, gt_mask, num_classes)

        # Update total IoU and count for each class where it appears
        for class_id in range(num_classes):
            if iou_per_class[class_id] > 0:
                total_iou_per_class[class_id] += iou_per_class[class_id]
                count_per_class[class_id] += 1

# Calculate mean IoU (mIoU) for each class
miou_per_class = [total_iou_per_class[i] / count_per_class[i] if count_per_class[i] > 0 else 0 for i in
                  range(num_classes)]

print(total_iou_per_class, count_per_class)
print("mIoU per class:", miou_per_class)
