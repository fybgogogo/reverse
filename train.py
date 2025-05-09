#coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import reverse
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser, criterions
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
from predict import AverageMeter, test_softmax

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--dataname', default='BRATS2018', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1024, type=int)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
         [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True],
         [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair',
             't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
             'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
             'flairt1cet1t2']
print(masks_torch.int())


def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False  #cuDNN的benchmark模式会尝试为每个卷积层选择最优的算法，但可能会导致结果的不可重复性
    cudnn.deterministic = True  #保证每次实验的结果相同

    ##########setting models
    if args.dataname in ['BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print('dataset is error')
        exit(0)
    model = reverse.Model(num_cls=num_cls)
    print(model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]

    total_params1 = sum(p.numel() for p in model.parameters())
    trainable_params1 = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型总参数量：{total_params1}")
    print(f"可训练参数量：{trainable_params1}")

    optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
    if args.dataname in ['BRATS2020', 'BRATS2015', 'BRATS2021']:
        train_file = 'train.txt'
        test_file = 'test_files.txt'
    elif args.dataname == 'BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = 'train3.txt'
        test_file = 'test6.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls,
                                  train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Evaluate
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        # test_score = AverageMeter()
        # with torch.no_grad():
        #     logging.info('###########test set wi post process###########')
        #     for i, mask in enumerate(masks[::-1]):
        #         logging.info('{}'.format(mask_name[::-1][i]))
        #         dice_score = test_softmax(
        #                         test_loader,
        #                         model,
        #                         dataname = args.dataname,
        #                         feature_mask = mask,
        #                         mask_name = mask_name[::-1][i])
        #         test_score.update(dice_score)
        #     logging.info('Avg scores: {}'.format(test_score.avg))
        #     exit(0)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    # iter_per_epoch = args.iter_per_epoch
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch + 1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except Exception as e:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)  #设置为非阻塞操作，这样可以在数据传输和计算之间并行执行
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            #print(mask[0][0])
            model.module.is_training = True
            fuse_pred, sep_preds, prm_preds, global_modality, reverse_forward_out, reverse_out, intra_token_x5 = model(x, mask)
            #, flair_preds, t1ce_preds, t1_preds, t2_preds
            ###Loss compute
            #计算模态融合后的损失
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss
            #计算未经掩码处理的，每一单个模态的损失
            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss

            #计算reverse损失
            mse_loss = nn.MSELoss()
            forward_mse_loss = 0
            reverse_mse_loss = 0
            for j in range(4):
                reverse_mse_loss = reverse_mse_loss + mse_loss(global_modality, reverse_out[j])
            for j in range(4):
                forward_mse_loss = forward_mse_loss + mse_loss(intra_token_x5[j], reverse_forward_out[j])

            #计算辅助正则项损失，用来提高模型的精度
            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss

            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss + forward_mse_loss + reverse_mse_loss
            else:
                loss = fuse_loss + sep_loss + prm_loss + forward_mse_loss + reversereverse_mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch + 1), args.num_epochs, (i + 1), iter_per_epoch,
                                                                  loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            file_name)

        if (epoch + 1) % 50 == 0 or (epoch >= (args.num_epochs - 10)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch + 1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

    msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            logging.info('{}'.format(mask_name[i]))
            dice_score, pred = test_softmax(
                i,
                test_loader,
                model,
                dataname=args.dataname,
                feature_mask=mask,
                save_dir='/home/lx/project/fyb/reverse-main/reverse/output/BraTS2018_pre_mmformer_reverese/')
            test_score.update(dice_score)
        logging.info('Avg scores: {}'.format(test_score.avg))


if __name__ == '__main__':
    main()

