import numpy as np
from skimage import io
import os
import glob

HIST = np.zeros((2,2))

def computeCD_pre_rec_iou(gt, predict, thresholds):
    '''
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    IOU       = TP / (TP + FP + FN)
    F1-score  = 2*pre*rec / (pre + rec)
    Accuracy  = TP + TN / (TP + FP + FN + TN)
    '''
    if(len(gt.shape) < 2 or len(predict.shape) < 2):
        print("ERROR: gt or mask is not matrix!")
        exit()
    if(len(gt.shape)>2): # convert to one channel
        gt = gt[:,:,0]
    if(len(predict.shape)>2): # convert to one channel
        predict = predict[:,:,0]
    if(gt.shape!=predict.shape):
        print("ERROR: The shapes of gt and mask are different!")
        exit()
    predict[predict >= thresholds] = 255
    predict[predict < thresholds] = 0
    TP = np.count_nonzero((gt == predict) & (gt > 0)) # TP
    TN = np.count_nonzero((gt == predict) & (gt == 0)) # TN
    FP = np.count_nonzero(gt < predict) # FP
    FN = np.count_nonzero(gt > predict) # FN

    HIST[0,0] = HIST[0,0] + TP
    HIST[1,1] = HIST[1,1] + TN
    HIST[0,1] = HIST[0,1] + FP + 1
    HIST[1,0] = HIST[1,0] + FN - 1

    return HIST

def compute_PRE_REC_IOU_FM_of_methods(gt_name_list, predict_dir):
    num_gt = len(gt_name_list)
    if(num_gt == 0):
        #print("ERROR: The ground truth directory is empty!")
        exit()

    for i in range(0, num_gt):
        print('>>Processed %d/%d'%(i+1,num_gt),end='\r')
        gt = io.imread(gt_name_list[i])
        gt= gt.astype('int')
        gt[gt==1]=255
        # #   统计影像像素值
        # pixel_counts = np.zeros(256, dtype=int)
        # for row in range(gt.shape[0]):
        #     for col in range(gt.shape[1]):
        #         pixel_value = gt[row, col]
        #         pixel_counts[pixel_value] += 1

        #     # 输出统计结果
        # for i in range(256):
        #     print(f'Pixel value {i}: {pixel_counts[i]} pixels')
        # exit(-1)
        gt_name = os.path.split(gt_name_list[i])[-1] # get the file name of the ground truth "xxx.png"
        # print(gt_name)
        # print("aaaa")
        # exit(-1)
        # pre, rec, iou, f1, acc = 0, 0, 0, 0, 0
        try:
            predict = io.imread(predict_dir+gt_name) # read the corresponding mask from each method
            #   统计影像像素值
            # pixel_counts = np.zeros(256, dtype=int)
            # for row in range(predict.shape[0]):
            #     for col in range(predict.shape[1]):
            #         pixel_value = predict[row, col]
            #         pixel_counts[pixel_value] += 1

            #     # 输出统计结果
            # for i in range(256):
            #     print(f'Pixel value {i}: {pixel_counts[i]} pixels')
            # exit(-1)
        except IOError:
            continue
        try:
            # _ = computeCD_pre_rec_iou(gt, predict, 1)
            _ = computeCD_pre_rec_iou(gt, predict, 127)
        except IOError:
            continue

    print('/n')
    # mPRE = np.sum(PRE) / (num_gt + 1e-8)
    # mREC = np.sum(REC) / (num_gt + 1e-8)
    # mIOU = np.sum(IOU) / (num_gt + 1e-8)
    # mF1M = np.sum(F1M) / (num_gt + 1e-8)
    # mACC = np.sum(ACC) / (num_gt + 1e-8)

    mPRE = HIST[0,0] / (HIST[0,0] + HIST[0,1])
    mREC = HIST[0,0] / (HIST[0,0] + HIST[1,0])
    mIOU = HIST[0,0] / (HIST[0,0] + HIST[0,1] + HIST[1,0])
    # mIOU = HIST[0,0] / (HIST[0,1] + HIST[1,0] - HIST[0,0])
    mF1M = 2*mPRE*mREC / (mPRE + mREC)
    mACC = (HIST[0,0] + HIST[1,1]) / (HIST[0,0] + HIST[0,1] + HIST[1,0] + HIST[1,1])
    # print(HIST[0,0] + HIST[0,1] + HIST[1,0] + HIST[1,1])
    # exit(-1)
    print('-'*20)
    print('Precision:{0:.4f}'.format(mPRE))
    print('   Recall:{0:.4f}'.format(mREC))
    print(' mean IOU:{0:.4f}'.format(mIOU))
    print(' F1-score:{0:.4f}'.format(mF1M))
    print(' Accuracy:{0:.4f}'.format(mACC))

    return mPRE, mREC, mIOU, mF1M, mACC



def main():


    # Comparative Experiments BCD_noBF
    # gt_name_list = glob.glob('C:/Users/11473/OneDrive/桌面/Building change detection dataset/left_middle_right_split/split/100/label'+'/'+'*.png')
    # predict_dir = 'C:/Users/11473/OneDrive/桌面/semiURNET/result/self_training/self_training_consistency/test_20AUG_selftraining/' 

   
    # # # # Comparative Experiments LEVIR
    gt_name_list = glob.glob('C:/Users/11473/OneDrive/桌面/semiURNET/dataset/LEVIR/test/label/'+'/'+'*.png')
    predict_dir = 'C:/Users/11473/OneDrive/桌面/semiURNET/result/LEVIR/self_training_consistency/test_5_ALL(aug+self+const)/' 
  
    


    compute_PRE_REC_IOU_FM_of_methods(
        gt_name_list,
        predict_dir
    )

if __name__ == '__main__':
    main()
