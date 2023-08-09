import os
import numpy as np
from PIL import Image


'''
def check_convergence(iou_valuesA, iou_valuesB, iou_valuesC, threshold=0.9, convergence_threshold=0.025):

    iou_values = [iou_valuesA, iou_valuesB, iou_valuesC]

    if len(iou_values) < 3:
        return False
    else:
        last_iou_values = iou_values[-3:]
        # return all(abs(iou - threshold) <= convergence_threshold for iou in last_iou_values)
    
        return all(iou > threshold for iou in last_iou_values) and all(abs(iou - last_iou_values[0]) <= convergence_threshold for iou in last_iou_values)
'''


def calculate_iou(mask_pred, mask_true):
    intersection = np.logical_and(mask_pred, mask_true)
    union = np.logical_or(mask_pred, mask_true)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def save_reliable_dataset(filenames, output_path):
    with open(output_path, 'w') as f:
        f.write('\n'.join(filenames))

# Paths to folders A, B, C, and D
folder_a_path = 'C:/Users/11473/OneDrive/桌面/semiURNET/result/LEVIR_paper_test/checkpoint1'
folder_b_path = 'C:/Users/11473/OneDrive/桌面/semiURNET/result/LEVIR_paper_test/checkpoint2'
folder_c_path = 'C:/Users/11473/OneDrive/桌面/semiURNET/result/LEVIR_paper_test/checkpoint3'
folder_d_path = 'C:/Users/11473/OneDrive/桌面/semi_Grad_CAM/res/LEVIR_paper_test/10/mask'
output_file_path = './reliable_dataset.txt'

# Get the list of mask files in folder D
folder_d_files = os.listdir(folder_d_path)

# Initialize IoU values and converged filenames list
iou_values = []
converged_filenames = []

# Iterate through each mask in folders A, B, and C
for filename in os.listdir(folder_d_path):
    # Load the mask from folder D with the same filename
    mask_d_path = os.path.join(folder_d_path, filename)
    if not os.path.isfile(mask_d_path):
        continue
    
    # Load the masks from folders A, B, and C
    mask_a_path = os.path.join(folder_a_path, filename)
    mask_b_path = os.path.join(folder_b_path, filename)
    mask_c_path = os.path.join(folder_c_path, filename)
    
    mask_a = np.array(Image.open(mask_a_path).convert('L'))
    mask_b = np.array(Image.open(mask_b_path).convert('L'))
    mask_c = np.array(Image.open(mask_c_path).convert('L'))
    mask_d = np.array(Image.open(mask_d_path).convert('L'))

    
    # Calculate IoU for each mask
    iou_a1 = calculate_iou(mask_a, mask_b)
    iou_a2 = calculate_iou(mask_a, mask_c)
    iou_b = calculate_iou(mask_b, mask_c)
    # iou_c = calculate_iou(mask_c, mask_c)
    iou_d = calculate_iou(mask_d, mask_c)
    
    # # Append the average IoU to the list
    # iou_values.append(iou_a)
    # iou_values.append(iou_b)
    # iou_values.append(iou_c)
    
    # Check convergence and IoU threshold
    if iou_a1 > 0.90 and iou_a2 > 0.90 and iou_b > 0.90 and iou_d > 0.85:
    # if iou_d > 0.9:
        converged_filenames.append(filename)

# Save the converged filenames to a text file
save_reliable_dataset(converged_filenames, output_file_path)
