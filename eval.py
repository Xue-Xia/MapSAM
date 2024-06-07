import os
import numpy as np
import SimpleITK as sitk
import argparse

def read_nifti(file_path):
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)
    return array

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    # Handle the case where the union is empty
    if np.sum(union) == 0:
        # If both masks are empty, return IoU of 1
        if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
            return 1.0
        # If only one is empty, return IoU of 0
        else:
            return 0.0
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

def main(folder_path):
    files = os.listdir(folder_path)
    iou_values = []
    for file_name in files:
        if file_name.endswith("_gt.nii.gz"):
            gt_file_path = os.path.join(folder_path, file_name)
            pred_file_path = os.path.join(folder_path, file_name.replace("_gt.nii.gz", "_pred.nii.gz"))

            # Read the NIfTI files
            gt_mask = read_nifti(gt_file_path)
            pred_mask = read_nifti(pred_file_path)

            # Calculate IoU for this pair of masks
            iou = calculate_iou(gt_mask, pred_mask)
            iou_values.append(iou)

    # Calculate average IoU
    avg_iou = np.mean(iou_values)
    print("mIoU:", avg_iou)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the mean IoU for predictions.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing prediction and ground truth NIfTI files.')
    args = parser.parse_args()
    main(args.folder_path)