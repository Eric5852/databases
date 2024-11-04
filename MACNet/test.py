import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse

from utils.dataloader import get_loader,test_dataset
from utils.eval_functions import *
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./weights/MACNet/MACNet.pt')

import imageio
def draw_contours_with_map(bground_path, mask_path, gtruth_path,name,save_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    mask2 = cv2.imread(gtruth_path, 0)
    mask = cv2.imread(mask_path, 0)
    img = cv2.imread(bground_path, 1)
    img = cv2.resize(img, (mask2.shape[1],mask2.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (mask2.shape[1],mask2.shape[0]), interpolation=cv2.INTER_LINEAR)
    #part1  将mask合成到image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # findContours函数用于找出边界点
    cv2.drawContours(img, contours, -1, (0, 0, 255), 4)  ##drawContours函数用于根据边界点画出图形
    img = img[:, :, ::-1]
    img[..., 2] = np.where(mask == 1, 255, img[..., 2])
    imageio.imwrite(save_path+"\\"+name, img)
    # part2 将segementation合成到part1的合成图里面
    imgfile =save_path+"\\"+name # 原图路径
    img1 = cv2.imread(imgfile, 1)
    # mask2 = cv2.imread(gtruth_path, 0)
    # mask2 = cv2.resize(mask2, (224, 224), interpolation=cv2.INTER_LINEAR)
    contours1, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # findContours函数用于找出边界点
    cv2.drawContours(img1, contours1, -1, (255,0,0),4)  ##drawContours函数用于根据边界点画出图形
    img1 = img1[:, :, ::-1]
    img1[..., 2] = np.where(mask == 1, 255, img1[..., 2])
    imageio.imwrite(save_path+"\\"+name, img1)

def draw_contours_with_gt(bground_path,mask_path, gtruth_path,name,save_path):


    if not os.path.exists(save_path): os.makedirs(save_path)

    gtruth = cv2.imread(gtruth_path, 0)
    gtruth = cv2.cvtColor(gtruth, cv2.COLOR_GRAY2RGB)

    mask = cv2.imread(mask_path, 0)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask=cv2.resize(mask, (gtruth.shape[1],gtruth.shape[0]), interpolation=cv2.INTER_LINEAR)

    img = cv2.imread(bground_path, 1)
    img = cv2.resize(img, (gtruth.shape[1], gtruth.shape[0]), interpolation=cv2.INTER_LINEAR)


    idx = np.where(mask[..., 0] > 0)
    mask[idx[0], idx[1], :] = [255,0,0]
    # part2 将segementation合成到part1的合成图里面

    idx = np.where(gtruth[..., 0] > 0)
    gtruth[idx[0], idx[1], :] = [0, 255,0]
    # 保存
    # print(gtruth.shape,mask.shape)
    vv= cv2.addWeighted(mask, 1, gtruth, 1, gamma=0)

    # img= cv2.addWeighted(img, 1, mask, 1, gamma=0)
    # img= cv2.addWeighted(img, 1, gtruth, 1,gamma=0)
    # vv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    imageio.imwrite(save_path + "//" + name , vv)

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    
    # print('-----------strating -------------')
    
    data_path = 'data/TestDataset/{}'.format(_data_name)
    save_path = 'Snapshot/seg_maps/{}/'.format(_data_name)
    composite_path = 'Snapshot/composite_maps/{}/'.format(_data_name)
    composite_path1 = 'Snapshot/composite_maps1/{}/'.format(_data_name)
    savegt_path="data/tesesize/{}/".format(_data_name)

    os.makedirs(savegt_path, exist_ok=True)
    os.makedirs(composite_path, exist_ok=True)
    os.makedirs(composite_path1, exist_ok=True)

    opt   = parser.parse_args()
    model=torch.load(opt.pth_path)
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    
    image_root = '{}/images/'.format(data_path)
    gt_root    = '{}/masks/'.format(data_path)

    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    DSC=[]
    IOU=[]
    SENS=[]
    PPV=[]

    from tqdm import tqdm
    for i in tqdm(range(test_loader.size)):

        
        # print(['--------------processing-------------', i])

        
        image, gt, name = test_loader.load_data()
        mask=gt
        
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res3,res2,res1,res = model(image)
        res = F.interpolate(res1+res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC.append(dice)
        iou=(intersection.sum()+smooth ) / (input.sum() + target.sum()-intersection.sum() + smooth)
        IOU.append(iou)
        sensitivity=(intersection.sum()+smooth ) / (target.sum() + smooth)
        SENS.append(sensitivity)
        ppv = (intersection.sum() + smooth) / (input.sum() + smooth)
        PPV.append(ppv)
        cv2.imwrite(savegt_path+ name, target * 255)

        cv2.imwrite(save_path+name, res*255)

    print(f"\033[1;36mDatesName:{_data_name}\033[0m","||\033[1;31mdice_coef:{:.4f}\033[0m  \033[1;32mjaccard_index:{:.4f}\033[0m  \033[1;31msensitivity:{:.4f}\033[0m  \033[1;32mppv:{:.4f}\033[0m".format(sum(DSC)/len(DSC),sum(IOU)/len(IOU),sum(SENS)/len(SENS),sum(PPV)/len(PPV)))
        # print(res.shape,gt.shape)

    # 遍历图片文件列表
    for image_file in sorted([f for f in os.listdir(image_root) if f.endswith((".jpg", ".jpeg", ".png"))]):
        # 构建完整的图片文件路径
        # print(image_root+image_file,save_path+ image_file,gt_root+image_file)

        bground_path =image_root+image_file# 原图路径
        mask_path = save_path+ image_file# mask路径
        gtruth_path = gt_root+image_file
        # name = '{}'.format(i)
        # print("开始合成图：{}".format(image_file))
        draw_contours_with_map(bground_path, mask_path, gtruth_path, name=image_file, save_path=composite_path)
        draw_contours_with_gt(bground_path,mask_path, gtruth_path,name=image_file, save_path=composite_path1)
print(f"\033[1;31m|***|>{ composite_path.split('/')[0] } successful test<|***|\033[0m")


