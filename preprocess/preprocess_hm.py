import os
import cv2
import numpy as np

import os
import cv2
import numpy as np

# Function to convert multi-channel binary label image to single-channel multi-class label image
def convert_to_multiclass_label(label_image):
    # Initialize an empty single-channel multi-class label image
    single_channel_label = np.zeros(label_image.shape[:2], dtype=np.uint8)
    num_classes = label_image.shape[2]

    # Assign unique class labels
    for i in range(num_classes):
        single_channel_label[label_image[:, :, i] == 255] = i + 1  # Assign class labels (1-indexed)

    return single_channel_label

# Input and output directories
input_dir = '../data/dataset_hm/partial_10/annotation'
output_dir = '../data/dataset_hm/partial_10/anno_multiclass'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each subset (train, val, test)
subsets = ['train', 'val']
for subset in subsets:
    subset_input_dir = os.path.join(input_dir, subset)
    subset_output_dir = os.path.join(output_dir, subset)
    os.makedirs(subset_output_dir, exist_ok=True)

    # Process each image in the subset
    image_files = os.listdir(subset_input_dir)
    for image_file in image_files:
        # Read the input image
        image_path = os.path.join(subset_input_dir, image_file)
        label_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # Convert multi-channel binary label image to single-channel multi-class label image
        single_channel_label = convert_to_multiclass_label(label_image)

        # Save the converted image to output directory
        output_path = os.path.join(subset_output_dir, image_file)
        cv2.imwrite(output_path, single_channel_label)

print("Conversion complete.")

