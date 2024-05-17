import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import pandas as pd
import numpy as np
from prettytable import PrettyTable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps, ImageDraw
import timm

num_gpus = torch.cuda.device_count()
print("available GPUs: ", num_gpus)
device_index = 0
device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
print("current device: ", device)

input_size = 448
backbone_name = "dla60_res2net"
assert backbone_name in ["dla60_res2net",]


def load_config():
    parser = argparse.ArgumentParser(description='PWS OCTA')
    parser.add_argument('--src_path', type=str, default="/data/LateOrchestration/Event_Domain_Adaptation/")

    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 32)",
    )
    parser.add_argument(
        "--epochs", default=130, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=4,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
             "batch size of all GPUs on the current node when "
             "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument('--cos', type=bool, default=False, help="use cosine lr schedule")
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=1,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('-e', '--evaluation', action='store_true',
                        help='evaluate model on validation set')

    return parser.parse_args()


def collate_fn(batch):
    imgs, events, batch_hms, batch_whs, batch_regs, batch_reg_masks = [], [], [], [], [], []

    for img, event, batch_hm, batch_wh, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        events.append(event)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)

    imgs = torch.stack(imgs)
    events = torch.stack(events)
    batch_hms = torch.stack(batch_hms)
    batch_whs = torch.stack(batch_whs)
    batch_regs = torch.stack(batch_regs)
    batch_reg_masks = torch.stack(batch_reg_masks)
    return imgs, events, batch_hms, batch_whs, batch_regs, batch_reg_masks


def train(train_loader, model, criterion, optimizer_D, optimizer_G, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses_task1 = AverageMeter("task1", ":.4f")
    losses_task2 = AverageMeter("task2", ":.4f")
    losses_gen = AverageMeter("gen", ":.4f")
    losses_disc = AverageMeter("disc", ":.4f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_task1, losses_task2, losses_gen, losses_disc],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, events, hm, wh, offset, mask) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            images = images.to(device)
            events = events.to(device)
            hm = hm.to(device)
            wh = wh.to(device)
            offset = offset.to(device)
            mask = mask.to(device)

        adjust_learning_rate(optimizer_D, epoch, i + 1, len(train_loader)) 
        # 更新判别器
        optimizer_D.zero_grad()
        
        loss_disc = model.discrimination_step(images, events)
        
        losses_disc.update(loss_disc.item(), images.size(0))
        loss_disc.backward()
        optimizer_D.step()
        
        if i % 2 == 0:
            if i > 0:
                adjust_learning_rate(optimizer_G, epoch, i + 1, len(train_loader)) 
                # 更新生成器
                optimizer_G.zero_grad()
                
                outputs1, outputs2, loss_gen = model.generation_step(images, events)
                
                loss_task_1 = criterion(outputs1[0], hm) + 0.1*reg_l1_loss(outputs1[1], wh, mask) + reg_l1_loss(outputs1[2], offset, mask)
                loss_task_2 = criterion(outputs2[0], hm) + 0.1*reg_l1_loss(outputs2[1], wh, mask) + reg_l1_loss(outputs2[2], offset, mask)
                
                losses_task1.update(loss_task_1.item(), images.size(0))
                losses_task2.update(loss_task_2.item(), images.size(0))
                losses_gen.update(loss_gen.item(), images.size(0))
                
                loss = loss_task_1 + loss_task_2 + loss_gen
                
                loss.backward()
                optimizer_G.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, epoch, args,):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses_task1 = AverageMeter("task1", ":.4f")
    losses_task2 = AverageMeter("task2", ":.4f")
    losses_gen = AverageMeter("gen", ":.4f")
    losses_disc = AverageMeter("disc", ":.4f")
    losses = AverageMeter("total", ":.4f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses_task1, losses_task2, losses_gen, losses_disc, losses],
        prefix="Epoch: [{}]".format(epoch),
    )
    
    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, events, hm, wh, offset, mask) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
        
            if args.cuda:
                images = images.to(device)
                events = events.to(device)
                hm = hm.to(device)
                wh = wh.to(device)
                offset = offset.to(device)
                mask = mask.to(device)

            # 判别损失
            loss_disc = model.discrimination_step(images, events)
            losses_disc.update(loss_disc.item(), images.size(0))
            
            # 生成损失
            outputs1, outputs2, loss_gen = model.generation_step(images, events)
            
            loss_task_1 = criterion(outputs1[0], hm) + 0.1*reg_l1_loss(outputs1[1], wh, mask) + reg_l1_loss(outputs1[2], offset, mask)
            loss_task_2 = criterion(outputs2[0], hm) + 0.1*reg_l1_loss(outputs2[1], wh, mask) + reg_l1_loss(outputs2[2], offset, mask)
            
            losses_task1.update(loss_task_1.item(), images.size(0))
            losses_task2.update(loss_task_2.item(), images.size(0))
            losses_gen.update(loss_gen.item(), images.size(0))
            
            total_loss = loss_disc + loss_gen + loss_task_1 + loss_task_2
            losses.update(total_loss.item(), images.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                progress.display(i)
    
    return losses.avg


def inference(test_loader, model, epoch, args, save_path):
    # switch to evaluate mode
    model.eval()
    
    save_dir = "{}/epoch_{}".format(save_path, epoch)
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (images, events, hm, wh, offset, mask) in enumerate(test_loader):
            if args.cuda:
                images = images.to(device)
                events = events.to(device)
                hm = hm.to(device)
                wh = wh.to(device)
                offset = offset.to(device)
                mask = mask.to(device)
            
            outputs_images, outputs_fake_events, fake_events = model.detection_step_fake_event(images, events)
            outputs_events = model.detection_step_event(events)
            
            for detect_inputs, detect_outputs, suffix_ in zip([images, events, fake_events], [outputs_images, outputs_events, outputs_fake_events], ["image", "event", "fake"]):
                decode_bboxes(detect_inputs, detect_outputs, "{}/{}_{}.png".format(save_dir, i, suffix_))


def _nms(heat, kernel=3):
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep


def get_peak_points(heatmaps):
    """
    :param heatmaps: numpy array (N,15,96,96)
    :return:numpy array (N,15,2)
    """
    N, C, H, W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy, xx = np.where(heatmaps[i, j] == heatmaps[i, j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x, y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points


def coordinate_transform(x, y, src_x, src_y, target_x, target_y, mode='bi-linear'):
    assert mode in ['bi-linear', 'scale']
    if mode == 'scale':
        y_ = (target_y / src_y) * y
        x_ = (target_x / src_x) * x
    else:
        y_ = (target_y / src_y) * (y + 0.5) - 0.5
        x_ = (target_x / src_x) * (x + 0.5) - 0.5
    
    return int(x_), int(y_)


def decode_bboxes(inputs, outputs, save_path):
    batch_size = inputs.shape[0]
    assert batch_size == 1
    
    # input: tensor --> image
    if inputs.shape[1] == 3:
        input_array = inputs.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        image = (input_array * 255.).astype(np.uint8)
        rgb_image = Image.fromarray(image).convert("RGB")
    else:
        input_array = inputs.detach().cpu().squeeze(0).squeeze(0).numpy()
        image = (input_array * 255.).astype(np.uint8)
        gray_image = Image.fromarray(image).convert("L")
        rgb_image = ImageOps.colorize(gray_image, (0, 0, 0), (255, 255, 255))
        
    rgb_image = rgb_image.resize((input_size, input_size))
    
    # outputs: drawing bbox on image
    hm, wh, _ = outputs
    hm, wh = hm.detach().cpu(), wh.detach().cpu().squeeze(0)
    
    # 非极大值抑制
    hm = _nms(hm)
    # 提取中心点
    peak_points = get_peak_points(hm).squeeze(0)  # [K, 2]
    
    # 提取wh
    # wh.shape: [2, 112, 112]
    
    # 目前，hm提取的中心点和宽高信息都是基于特征图尺寸的
    
    # 创建一个可绘制对象
    draw = ImageDraw.Draw(rgb_image)
        
    for k, bbox_color in zip([0, 1], ["blue", "red"]):
        x_ct, y_ct = peak_points[k]
        w, h = wh[0, y_ct, x_ct], wh[1, y_ct, x_ct]
        
        # 变换到输入尺寸
        half_w = int(w * 4 / 2)
        half_h = int(h * 4 / 2)
        
        x_ct, y_ct = coordinate_transform(x_ct, y_ct,
                                          int(input_size / 4), int(input_size / 4),
                                          input_size, input_size)
        
        

        # 定义矩形框的坐标
        x1, y1 = max(x_ct - half_w, 0), max(y_ct - half_h, 0)
        x2, y2 = min(x_ct + half_w, input_size), min(y_ct + half_h, input_size)

        # 绘制矩形框
        draw.rectangle([(x1, y1), (x2, y2)], outline=bbox_color, width=2)

    # 保存绘制后的图像
    rgb_image.save(save_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def save_checkpoint(best_prec1, model, epoch, check_path, is_best):
    print('Best Model Saving...')
    model_state_dict = model.state_dict()

    # Save checkpoint
    checkpoint = {
        'state_dict': model_state_dict,
        'best_prec1': best_prec1,
    }
    torch.save(checkpoint, os.path.join(check_path, '{}_{:.4f}.pth.tar'.format(epoch, best_prec1)))
    # torch.save(checkpoint, os.path.join(check_path, "best_model.pth.tar"))

    if is_best:
        shutil.copyfile(os.path.join(check_path, '{}_{:.4f}.pth.tar'.format(epoch, best_prec1)),
                        os.path.join(check_path, "best_model.pth.tar"))


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    if args.cos:  # cosine lr schedule
        lr = args.lr
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # milestones
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
        lr = args.lr * (0.1 ** factor)
        """Warmup"""
        if epoch < 5:
            lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)
            # if args.local_rank == 0:
            # print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Detection Loss: wh, offsets
def reg_l1_loss(pred, target, mask):
    loss = F.l1_loss(pred * mask, target, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def main(args):
    global best_score
    best_score = 100.

    suffix_ = "dla60_res2net"
    # save PATH
    checkpoint_path = 'checkpoints/{}'.format(suffix_)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    save_path = 'inference/{}'.format(suffix_)
    os.makedirs(save_path, exist_ok=True)

    # Data loading code
    train_transfrom = transforms.Compose([
        # optical_augmentation
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 5.0))], p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.RandomAutocontrast(),
        transforms.RandomEqualize(),
        # geometric_augmentation
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        ])
    
    val_transfrom = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        ])

    from datas.data_loader import FrameEventData
    train_dataset = FrameEventData(
        src_path=args.src_path, 
        ann_file="frame2event_train_dataset.csv", 
        img_transform=train_transfrom, 
        gt_size=int(input_size/4), 
        num_classes=2)

    val_dataset = FrameEventData(
        src_path=args.src_path, 
        ann_file="frame2event_val_dataset.csv", 
        img_transform=val_transfrom, 
        gt_size=int(input_size/4), 
        num_classes=2)

    test_dataset = FrameEventData(
        src_path=args.src_path, 
        ann_file="frame2event_test_dataset.csv", 
        img_transform=val_transfrom, 
        gt_size=int(input_size/4), 
        num_classes=2)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    from models.event_model import Frame2Event
    # create model
    model = Frame2Event(backbone_name, 2)

    from models.loss_func import FocalLoss
    # define loss function (criterion) and optimizer
    criterion = FocalLoss(2., 4.)
    
    optimizer_D = optim.SGD([{'params': model.content_disc.parameters()}, {'params': model.refine_disc.parameters()}],
                          lr=args.lr,
                          weight_decay=args.weight_decay,
                          momentum=args.momentum)
    
    # other_parameters = [param for name, param in model.named_parameters() if 'content_disc' not in name and 'refine_disc' not in name]
    
    other_parameters = []
    for name, param in model.named_parameters():
        if 'content_disc' not in name and 'refine_disc' not in name:
            print(name)
            other_parameters.append(param)    
    
    optimizer_G = optim.SGD([{'params': other_parameters},],
                          lr=args.lr,
                          weight_decay=args.weight_decay,
                          momentum=args.momentum)

    if args.evaluation:
        checkpoints = torch.load("{}/best_model.pth.tar".format(checkpoint_path), 'cpu')
        # checkpoints = torch.load("{}/amp_checkpoint.pt".format(checkpoint_path), 'cpu')
        model.load_state_dict(checkpoints['state_dict'])
        print("# loaded checkpoint from: ", "{}/best_model.pth.tar".format(checkpoint_path))
        # test_score_1, test_score_2 = checkpoints['best_prec1']
        # print("# dice score = {:.4f}, iou score = {:.4f} ".format(test_score_1, test_score_2))

    start_epoch = 1

    if args.cuda:
        model = model.to(device)
        criterion = criterion.to(device)

    if not args.evaluation:
        for epoch in range(start_epoch, args.epochs + 1):
            print("Training Phase:")
            train(train_loader, model, criterion, optimizer_D, optimizer_G, epoch, args)
            
            if epoch > 4:
                print("Testing Phase:")
                test_score = validate(val_loader, model, criterion, epoch, args)
                print("# test score = {:.4f}".format(test_score))

                is_best = test_score < best_score
                best_score = min(test_score, best_score)

                if is_best:
                    save_checkpoint(best_score, model, epoch, checkpoint_path, is_best=True,)

                for param_group in optimizer_D.param_groups:
                    print("| Current learning rate is: {}".format(param_group['lr']))
                
                # inference(test_loader, model, epoch, args, save_path)

    else:
        # test_score = validate(val_loader, model, criterion, 1, args)
        # print("# test score = {:.4f}".format(test_score))
        inference(test_loader, model, "best", args, save_path)


if __name__ == '__main__':
    args = load_config()
    main(args)
