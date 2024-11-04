import os
import torch
import argparse

from utils.dataloader import get_loader, test_dataset
from utils.trainer import adjust_lr
from datetime import datetime

import torch.nn.functional as F
import numpy as np
import logging

best_epoch = 0
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def test(model,path,dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        _,_,res2,res1 = model(image)
        # eval Dice
        res = F.interpolate(res2+res1, size=gt.shape, mode='bilinear', align_corners=True)
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
        DSC = DSC + dice

    return DSC / num1

def train(train_loader, model, optimizer, epoch, opt, loss_func, total_step):

    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    for step, data_pack in enumerate(train_loader):

        images, gts, egs = data_pack

        for rate in size_rates:

            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            egs = egs.cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                egs = F.interpolate(egs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            cam_edge, sal_out1, sal_out2, sal_out3 = model(images)
            loss_edge = loss_func(cam_edge,egs)
            loss_sal1 = structure_loss(sal_out1, gts)
            loss_sal2 = structure_loss(sal_out2, gts)
            loss_sal3 = structure_loss(sal_out3, gts)

            loss_total = loss_edge + loss_sal1 + loss_sal2 + loss_sal3

            loss_total.backward()
            optimizer.step()

        if step % 20 == 0 or step == total_step:
            print(
                '\033[1;34m[Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] =>  [Loss_edge: {:.3f} Loss_sal1: {:0.3f} Loss_sal2: {:0.3f} Loss_sal3: {:0.3f} Loss_total: {:0.3f}，lr:{}]\033[0m'.
                format(epoch, opt.epoch, step, total_step, loss_edge.data, loss_sal1.data, loss_sal2.data,
                       loss_sal3.data, loss_total.data,optimizer.state_dict()['param_groups'][0]['lr']))

            logging.info(
                '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_edge: {:.4f} Loss_sal1: {:0.4f} Loss_sal2: {:0.4f} Loss_sal3: {:0.4f} Loss_total: {:0.4f}'.
                format(epoch, opt.epoch, step, total_step, loss_edge.data, loss_sal1.data, loss_sal2.data,
                       loss_sal3.data, loss_total.data))
    test1path = './data/TestDataset/'
    test0=[]
    if (epoch + 1) >10:
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            dataset_dice = test(model, path=test1path, dataset=dataset)
            test0.append(dataset_dice)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(f"\033[1;31m{dataset}:\033[0m\033[1;32m{dataset_dice:.4f}\033[0m")

        meandice = np.mean(test0)
        if meandice > best:
            best = meandice
            torch.save(model, save_path + str(epoch) + "_DICE_{:.4f}".format(best) +save_path.split("/")[2]+ '.pt')
            torch.save(model.state_dict(), save_path + str(epoch) + "_DICE_{:.4f}".format(best) +save_path.split("/")[2]+ '.pht')
            print('\033[1;35m############################################################################>>\033[0m\033[1;31mbest:{:.4f}\033[0m'.format(best))
            logging.info(
                '\033[1;37m##############################################################################>>best:{}\033[0m'.format(
                    best))
    return loss_total









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=60, help='epoch number, default=30')
    parser.add_argument('--lr', type=float, default=1e-4, help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every N epochs decay lr')
    parser.add_argument('--gpu', type=int, default=0, help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=5, help='every N epochs save your trained snapshot')
    # parser.add_argument('--save_model', type=str, default='./weights/MACNet/')

    parser.add_argument('--train_img_dir', type=str, default='./data/TrainDataset/images/')
    parser.add_argument('--train_gt_dir', type=str, default='./data/TrainDataset/masks/')
    parser.add_argument('--train_eg_dir', type=str, default='./data/TrainDataset/edges/')

    parser.add_argument('--test_img_dir', type=str, default='./data/TestDataset/CVC-300/images/')
    parser.add_argument('--test_gt_dir', type=str, default='./data/TestDataset/CVC-300/masks/')
    parser.add_argument('--test_eg_dir', type=str, default='./data/TestDataset/CVC-300/edges/')
    parser.add_argument('--save_model', type=str, default='./weights/MACNet/')
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(filename=opt.save_model + '/log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                        datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("COD-Train")
    logging.info("Config")
    logging.info(
        'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(opt.epoch,
                                                                                                            opt.lr,
                                                                                                            opt.batchsize,
                                                                                                            opt.trainsize,
                                                                                                            opt.clip,
                                                                                                            opt.decay_rate,
                                                                                                            opt.save_model,
                                                                                                            opt.decay_epoch))
    from lib.MACNet import MACNet

    model = MACNet(n_classes=1, in_channels=256).cuda()
    # print('-' * 30, model, '-' * 30)

    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter:%.2fM' % (total / 1e6))

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    LogitsBCE = torch.nn.BCEWithLogitsLoss()

    train_loader = get_loader(opt.train_img_dir, opt.train_gt_dir, opt.train_eg_dir, batchsize=opt.batchsize,
                              trainsize=opt.trainsize, num_workers=8)
    test_loader = test_dataset(opt.test_img_dir, opt.test_gt_dir, testsize=opt.trainsize)

    total_step = len(train_loader)

    print('-' * 30, "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\n".format(opt.train_img_dir, opt.train_gt_dir, opt.lr,
                                                                opt.batchsize, opt.save_model, total_step), '-' * 30)
    best = 0
    from torch import optim
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=7)
    for epoch_iter in range(1, opt.epoch):
        # adjust_lr(optimizer, epoch_iter, opt.decay_rate)
        loss=train(train_loader, model, optimizer, epoch_iter, opt, LogitsBCE, total_step)
        scheduler.step(loss)

