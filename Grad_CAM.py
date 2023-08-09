import os
from matplotlib.style import use
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import ToTensor
from data_loader import Rescale
import glob
from model import UResNet_VGG
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torch.functional as F
import torch
from torch.autograd import Variable
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, LayerCAM
import numpy as np
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class CDTarget:
    def __init__(self, predict) -> None:
        self.mask = torch.from_numpy(predict)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
    
    def __call__(self, model_output):
        return (model_output[0,:,:]*self.mask).sum()
    
def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

image_paths_A = glob.glob('C:/Users/11473/OneDrive/桌面/semiURNET/dataset/LEVIR/LEVIR/10/train1/A/*.png')
image_paths_B = glob.glob('C:/Users/11473/OneDrive/桌面/semiURNET/dataset/LEVIR/LEVIR/10/train1/B/*.png')
image_paths_label = glob.glob('C:/Users/11473/OneDrive/桌面/semiURNET/dataset/LEVIR/LEVIR/10/train1/label/*.png')


save_path = './res/LEVIR_paper_test/10/'                
if not os.path.exists(save_path):
        os.makedirs(save_path)

for i in range(len(image_paths_A)):
    image_path_A = image_paths_A[i]
    image_path_B = image_paths_B[i]
    image_path_label = image_paths_label[i]

    imageA = np.array(Image.open(image_path_A))
    imageB = np.array(Image.open(image_path_B))
    imageLabel = np.array(Image.open(image_path_label))


    if np.sum(imageLabel) == 0:
        pass
    else:
        imageLabel = Image.fromarray(imageLabel*255).convert('RGB')
    

        RGB_imageA = np.float32(imageA) / 255
        RGB_imageB = np.float32(imageB) / 255

        inputs_testA = preprocess_image(RGB_imageA,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        inputs_testB = preprocess_image(RGB_imageB,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # loader model and weights

        model_dir = "C:/Users/11473/OneDrive/桌面/semiURNET/epochs/LEVIR/10/netCD_epoch_10_best_val_iou.pth"
        

        model = UResNet_VGG(3, 1)
        # print(model)
        # exit(-1)
        model.load_state_dict(torch.load(model_dir))
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
            inputs_testA = inputs_testA.cuda()
            inputs_testB = inputs_testB.cuda()


        target_layers = [model.decoder0]
        # print(target_layers4)
        # exit(-1)
        pred_cdmap_ref_float = np.ones((1, 256, 256), dtype=np.float32)
        targets = [CDTarget(pred_cdmap_ref_float)]

        with GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam_0, grayscale_cam_1 = cam(input_tensor=[inputs_testA, inputs_testB], targets=targets)
            grayscale_cam_0 = grayscale_cam_0[0,:]
            grayscale_cam_1 = grayscale_cam_1[0,:]


            file_name = os.path.splitext(os.path.basename(image_path_A))[0]
            save_file_path = os.path.join(save_path, 'CAM1',file_name + '.png')
            os.makedirs(os.path.join(save_path, 'CAM1'), exist_ok=True)
            
        
            # Save the GradCAM image to disk
            
            if np.max(grayscale_cam_0) > 1.0 or np.min(grayscale_cam_0) < 0 or np.max(imageLabel) > 1.0 or np.min(imageLabel) < 0:
                pass
            else:
                cam_image0 = show_cam_on_image(imageLabel, grayscale_cam_0, use_rgb=True)
                cam1 = Image.fromarray(cam_image0)
                cam1.save(save_file_path)


            
            
            
            # Convert grayscale_cam_0 to binary mask using a threshold
            # threshold = 0.1
            # binary_mask = (grayscale_cam_0 > threshold).astype(np.uint8) * 255
            grayscale_cam_0 = (grayscale_cam_0 * 255).astype(np.uint8)
            threshold, binary_mask = cv2.threshold(grayscale_cam_0, int(0.7 * cv2.threshold(grayscale_cam_0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]), 255, cv2.THRESH_BINARY)

            # Save the binary mask to disk
            file_name = os.path.splitext(os.path.basename(image_path_A))[0]
            binary_mask_file_path = os.path.join(save_path, 'mask',file_name + '.png')
            os.makedirs(os.path.join(save_path, 'mask'), exist_ok=True)
            binary_mask_image = Image.fromarray(binary_mask)
            binary_mask_image.save(binary_mask_file_path)

        