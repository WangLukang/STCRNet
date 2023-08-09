import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import pandas as pd

import numpy as np
import glob
import os

from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalCDDataset

# from model import BASNet
from model import UResNet, UResNet_VGG, UResNet_34


def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn



def calMetric_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict==1)
    fn = np.sum(label == 1)
    return tp,fp+fn-tp

def calMetric_acc(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    tn = np.sum(np.logical_and(predict == 0, label == 0))
    fp = np.sum(predict==1) - tp
    fn = np.sum(label == 1) - tp
    acc = (tp + tn) / (tp + tn + fp + fn)
    return tp+tn,tp + tn + fp + fn

def main():
    # ------- 1. define loss function --------

    bce_loss = nn.BCELoss(size_average=True)
    def muti_bce_loss_fusion(d0, labels_v):

        loss0 = bce_loss(d0,labels_v)
        # loss0 = bce_ssim_loss(d0,labels_v)
        loss = loss0 
        return loss, loss

    # ------- 2. set the directory of training dataset --------
  
    data_dir = 'C:/Users/11473/OneDrive/桌面/semiURNET/dataset/LEVIR/LEVIR/10_paper_test/'

    tra_image_dirA = 'pre_train/A/'
    tra_image_dirB = 'pre_train/B/'
    tra_label_dir = 'pre_train/label/'

 
    val_image_dirA = 'val/A/'
    val_image_dirB = 'val/B/'
    val_label_dir = 'val/label/'
    # image_ext = '.tif'
    image_ext = '.png'
    label_ext = '.png'
    
    model_dir = "epochs/LEVIR_paper_test/10_pre_train/"
    sta_dir = "statistics/LEVIR_paper_test_10_pre_train.csv"
    # model_dir_con = ""
    os.makedirs(model_dir, exist_ok=True)
  

    
    epoch_num = 100
    batch_size_train = 8
    batch_size_val = 1


    tra_img_name_listA = glob.glob(data_dir + tra_image_dirA + '*' + image_ext)
    tra_img_name_listB = glob.glob(data_dir + tra_image_dirB + '*' + image_ext)
    tra_lbl_name_list = glob.glob(data_dir + tra_label_dir + '*' + image_ext)

    print("---")
    print("train imagesA: ", len(tra_img_name_listA))
    print("train imagesB: ", len(tra_img_name_listB))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    val_img_name_listA = glob.glob(data_dir + val_image_dirA + '*' + label_ext)
    val_img_name_listB = glob.glob(data_dir + val_image_dirB + '*' + label_ext)
    val_lbl_name_list = glob.glob(data_dir + val_label_dir + '*' + label_ext)

    salobj_dataset = SalCDDataset(
        img_name_listA=tra_img_name_listA,
        img_name_listB=tra_img_name_listB,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(256),
            RandomCrop(224),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

    salobj_dataset_val = SalCDDataset(
        img_name_listA=val_img_name_listA,
        img_name_listB=val_img_name_listB,
        lbl_name_list=val_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(256),
            # RandomCrop(224),
            ToTensorLab(flag=0)]))
    salobj_dataloader_val = DataLoader(salobj_dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=0)
    # ------- 3. define model --------
    # define the net
    net = UResNet_VGG(3, 1)
    # net.load_state_dict(torch.load(model_dir_con))
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(3407)
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80 ,90], gamma=0.9)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    best_model = 0
    results = {'train_loss': [], 'train_IoU':[],'val_IoU': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, epoch_num+1):
        running_results = {'batch_sizes': 0, 'IoU':0, 'CD_loss':0, 'acc':0}
        net.train()
        train_bar = tqdm(salobj_dataloader)
        iou1, iou2, acc_1, acc_2 = 0,0,0,0
        for data in train_bar:
            # ite_num = ite_num + 1
            # ite_num4val = ite_num4val + 1
            # exit(-1)
            running_results['batch_sizes']+=batch_size_train

            inputsA, inputsB, labels = data['imageA'],data['imageB'], data['label']

            inputsA = inputsA.type(torch.FloatTensor)
            inputsB = inputsB.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_vA,inputs_vB, labels_v = Variable(inputsA.cuda(), requires_grad=False), Variable(inputsB.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
            else:
                inputs_vA,inputs_vB, labels_v = Variable(inputsA, requires_grad=False), Variable(inputsB, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, _ = net(inputs_vA, inputs_vB)
            loss2, loss = muti_bce_loss_fusion(d0, labels_v)
            

            
            for cc in range(data['imageA'].shape[0]):
                pred_cdmap_ref = d0[cc, :, :, :]
                pred_cdmap_ref = normPRED(pred_cdmap_ref)
                pred_cdmap_ref = torch.ge(pred_cdmap_ref, 0.5).float()
                pred_cdmap_ref = pred_cdmap_ref.squeeze()
                pred_cdmap_ref = pred_cdmap_ref.cpu().data.numpy()
                gt_value = labels_v[cc, :, :, :].squeeze().cpu().detach().numpy()
                acc_11,acc_22 = calMetric_acc(gt_value, pred_cdmap_ref)
                iou11,iou22 = calMetric_iou(gt_value, pred_cdmap_ref)
                iou1 += iou11
                iou2 += iou22
                acc_1 += acc_11
                acc_2 += acc_22
                
            loss.backward()
            optimizer.step()

            # running_results['Tar_loss'] += loss2.item() * batch_size_train
            running_results['CD_loss'] += loss.item() * batch_size_train

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del temporary outputs and loss
            del d0, loss2, loss

            train_bar.set_description(
                desc='[%d/%d] CD: %.4f  ' % (
                    epoch, epoch_num, running_results['CD_loss'] / running_results['batch_sizes'],
                    ))
        
       
        running_results['acc'] = acc_1/acc_2
        running_results['IoU'] = iou1/iou2

        # scheduler.step()
        net.eval()
        with torch.no_grad():
            val_bar = tqdm(salobj_dataloader_val)
            inter, unin, acc1v, acc2v = 0, 0, 0, 0
            valing_results = {'CD_loss': 0, 'batch_sizes': 0, 'Tar_loss':0, 'IoU': 0, 'acc':0}

            for data in val_bar:
                valing_results['batch_sizes'] += batch_size_val

                inputsA, inputsB, labels = data['imageA'],data['imageB'], data['label']

                inputsA = inputsA.type(torch.FloatTensor)
                inputsB = inputsB.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs_vA,inputs_vB, labels_v = Variable(inputsA.cuda(), requires_grad=False), Variable(inputsB.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
                else:
                    inputs_vA,inputs_vB, labels_v = Variable(inputsA, requires_grad=False), Variable(inputsB, requires_grad=False), Variable(labels, requires_grad=False)

                CD_final, _ = net(inputs_vA, inputs_vB)
                pred_cdmap_ref = CD_final[:, 0, :, :]
                pred_cdmap_ref = normPRED(pred_cdmap_ref)
                pred_cdmap_ref = torch.ge(pred_cdmap_ref, 0.5).float()
                pred_cdmap_ref = pred_cdmap_ref.squeeze()
                pred_cdmap_ref = pred_cdmap_ref.cpu().data.numpy()
                gt_value = labels_v.squeeze().cpu().detach().numpy()
                intr, unn = calMetric_iou(gt_value, pred_cdmap_ref)
                acc_111,acc_222 = calMetric_acc(gt_value, pred_cdmap_ref)
                acc1v += acc_111
                acc2v += acc_222
                inter = inter + intr
                unin = unin + unn

                val_bar.set_description(desc='IoU: %.4f' % (inter * 1.0 / unin))
            valing_results['IoU'] = inter * 1.0 / unin
            valing_results['acc'] = acc1v/acc2v
            val_loss = valing_results['IoU']
            if val_loss > best_model or epoch%5==0:
                best_model = val_loss
                torch.save(net.state_dict(),  model_dir+'netCD_epoch_%d_val_iou_%.4f.pth' % (epoch, val_loss))
            results['train_loss'].append(running_results['CD_loss'] / running_results['batch_sizes'])
            results['val_IoU'].append(valing_results['IoU'])
            results['train_IoU'].append(running_results['IoU'])
            results['train_acc'].append(running_results['acc'])
            results['val_acc'].append(valing_results['acc'])

            if epoch % 1 == 0 :
                data_frame = pd.DataFrame(
                    data={'train_loss': results['train_loss'],
                        'train_IoU': results['train_IoU'],
                        'val_IoU': results['val_IoU'],
                        'train_acc': results['train_acc'],
                        'val_acc': results['val_acc'],
                        },
                    index=range(1, epoch + 1))
                data_frame.to_csv(sta_dir, index_label='Epoch')

    print('-------------Congratulations! Training Done!!!-------------')

if __name__ == '__main__':
    main()