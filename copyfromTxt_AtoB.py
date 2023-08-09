import os
import shutil

def copy_files_from_list(source_folderA, source_folderB, source_folderLabel, target_folder, file_list_path):
    with open(file_list_path, 'r') as file:
        file_names = file.read().splitlines()

    for file_name in file_names:
        source_pathA = os.path.join(source_folderA, file_name)
        target_pathA = os.path.join(target_folder, 'A', file_name)
        shutil.copyfile(source_pathA, target_pathA)

        source_pathB = os.path.join(source_folderB, file_name)
        target_pathB = os.path.join(target_folder, 'B', file_name)
        shutil.copyfile(source_pathB, target_pathB)

        source_pathLabel = os.path.join(source_folderLabel, file_name)
        target_pathLabel = os.path.join(target_folder, 'label', file_name)
        shutil.copyfile(source_pathLabel, target_pathLabel)

def main():
    A_folder = "C:/Users/11473/OneDrive/桌面/semiURNET/dataset/LEVIR/LEVIR/10_paper_test/train_rest/A/"
    B_folder = "C:/Users/11473/OneDrive/桌面/semiURNET/dataset/LEVIR/LEVIR/10_paper_test/train_rest/B/"
    pesudolabel_folder = "C:/Users/11473/OneDrive/桌面/semiURNET/result/LEVIR_paper_test/checkpoint3/"
    des_folder = "C:/Users/11473/OneDrive/桌面/semiURNET/dataset/LEVIR/LEVIR/10_paper_test/train_selftraining1/"
    file_list_path = "./converged_masks.txt"
    os.makedirs(os.path.join(des_folder, 'A'), exist_ok=True)
    os.makedirs(os.path.join(des_folder, 'B'), exist_ok=True)
    os.makedirs(os.path.join(des_folder, 'label'), exist_ok=True)

    copy_files_from_list(A_folder, B_folder, pesudolabel_folder, des_folder, file_list_path)

if __name__ == "__main__":
    main()
