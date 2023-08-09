import os
from skimage import io
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import Rescale
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset, SalCDDataset


from model import UResNet_VGG

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	# img_name = image_name.split("/")[-1]
	_, img_name = os.path.split(image_name)
	# print(image_name)
	# print(img_name)
	# print(img_nameC)
	# exit(-1)
	image = io.imread(image_name)
	# print(image.shape)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]
	# print(d_dir+imidx+'.png')
	# exit(-1)
	imo.save(d_dir+imidx+'.png')
	# imo.save(d_dir+imidx+'.jpg')


if __name__ == '__main__':
	# --------- 1. get image path and name ---------

	image_dirA = 'C:/Users/11473/OneDrive/桌面/semiURNET/dataset/LEVIR/test/A/'
	image_dirB = 'C:/Users/11473/OneDrive/桌面/semiURNET/dataset/LEVIR/remove_blank/test/B/'	
	prediction_dir_cdmap_ref = 'C:/Users/11473/OneDrive/桌面/semiURNET/result/LEVIR_paper_test/final_results/'
	



	if not os.path.exists(prediction_dir_cdmap_ref):
		os.makedirs(prediction_dir_cdmap_ref, exist_ok=True)
	

	# # BCDDDDD
	# model_dir = "C:/Users/11473/OneDrive/桌面/semiURNET/epochs/BCD_noBF/self_training/self_training_consistency/20AUG_selftraining/netCD_epoch_10_val_iou_0.8682.pth"
	

	#LEVIR
	model_dir = "C:/Users/11473/OneDrive/桌面/semiURNET/epochs/LEVIR_paper_test/10_train_STCR/netCD_epoch_49_val_iou_0.8240.pth" 
	
	
	img_name_listA= glob.glob(image_dirA + '*.png')
	img_name_listB= glob.glob(image_dirB + '*.png')


	
	# --------- 2. dataloader ---------
	#1. dataload
	test_salobj_dataset = SalCDDataset(
		img_name_listA = img_name_listA, 
		img_name_listB = img_name_listB, 
		lbl_name_list = [],
		transform=transforms.Compose([Rescale(256),ToTensor()]))
	test_salobj_dataloader = DataLoader(
		test_salobj_dataset, 
		batch_size=1,shuffle=False,num_workers=1)
	
	# --------- 3. model define ---------
	print("...load STCRNet...")
	net = UResNet_VGG(3, 1)
	net.load_state_dict(torch.load(model_dir))
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	
	
	# --------- 4. inference for each image ---------
	for i_test, data_test in enumerate(test_salobj_dataloader):
		# print("inferencing:",img_name_listA[i_test].split("/")[-1])
	
		inputs_testA, inputs_testB = data_test['imageA'], data_test['imageB']
		inputs_testA = inputs_testA.type(torch.FloatTensor)
		inputs_testB = inputs_testB.type(torch.FloatTensor)
	
		if torch.cuda.is_available():
			inputs_testA = Variable(inputs_testA.cuda())
			inputs_testB = Variable(inputs_testB.cuda())
		else:
			inputs_testA = Variable(inputs_testA)
			inputs_testB = Variable(inputs_testB)
	
		refcd,_ = net(inputs_testA, inputs_testB)
	
		# normalization
		pred_cdmap_ref = refcd[:,0,:,:]
		pred_cdmap_ref = normPRED(pred_cdmap_ref)
		pred_cdmap_ref = torch.ge(pred_cdmap_ref, 0.5).float()

	
		save_output(img_name_listA[i_test],pred_cdmap_ref,prediction_dir_cdmap_ref)
	
		del refcd