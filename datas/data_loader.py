from torchvision import transforms
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import random
import cv2
import json
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True


def CoordinateTransform(x, y, origin_x, origin_y, target_x, target_y, mode='bi-linear'):
    assert mode in ['bi-linear', 'scale']
    if mode == 'scale':
        y_ = (target_y / origin_y) * y
        x_ = (target_x / origin_x) * x
    else:
        y_ = (target_y / origin_y) * (y + 0.5) - 0.5
        x_ = (target_x / origin_x) * (x + 0.5) - 0.5
    return x_, y_


# ----------Gaussian Heatmap-----------------
def gaussian(array_like_hm, mean, sigma):
    """modifyed version normal distribution pdf, vector version"""
    array_like_hm -= mean
    x_term = array_like_hm[:, 0] ** 2
    y_term = array_like_hm[:, 1] ** 2
    exp_value = - (x_term + y_term) / 2 / pow(sigma, 2)
    return np.exp(exp_value)


def draw_heatmap(width, height, x, y, sigma, array_like_hm):
    m1 = (x, y)
    zz = gaussian(array_like_hm, m1, sigma)
    img = zz.reshape((height, width))
    return img


def gaussian_heatmap(X_t, Y_t, target_size, sigma=0.4):
    g_heatmap = []
    for i in range(len(X_t)):
        xres = target_size
        yres = target_size

        x = np.arange(xres, dtype=np.float_)
        y = np.arange(yres, dtype=np.float_)
        xx, yy = np.meshgrid(x, y)

        # evaluate kernels at grid points
        xxyy = np.c_[xx.ravel(), yy.ravel()]

        # heatmap = np.zeros((img_size, img_size))
        # heatmap = test(xres, yres, X_t[i], Y_t[i], xxyy.copy())
        heatmap = draw_heatmap(xres, yres, X_t[i], Y_t[i], sigma, xxyy.copy())
        g_heatmap.append(torch.from_numpy(heatmap).unsqueeze(0))

    return torch.cat(g_heatmap, 0)


class FrameEventData(Dataset):
    def __init__(self, src_path, ann_file, img_transform, gt_size, num_classes=2):
        self.src_path = src_path
        # self.ann = pd.read_csv(self.src_path + ann_file)
        with open(self.src_path + ann_file, 'r') as json_file:
            ann = json.load(json_file)
        
        self.ann = ann
        self.img_transform = img_transform
        self.gt_size = gt_size
        self.num_classes = num_classes

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        info = self.ann[idx]

        image_file = str(info["img file"])        
        img = Image.open(self.src_path + image_file).convert('RGB')
        img_tensor = self.img_transform(img)
        
        event_file = str(info["event path"])        
        event = Image.open(self.src_path + event_file).convert('L')
        event_tensor = self.img_transform(event)
            
        image_width = float(img.size[0])
        image_height = float(img.size[1])
        
        hm = torch.zeros((self.num_classes, self.gt_size, self.gt_size))
        wh = torch.zeros((2, self.gt_size, self.gt_size))
        offset = torch.zeros((2, self.gt_size, self.gt_size))
        mask = torch.zeros((1, self.gt_size, self.gt_size))
        
        for class_id in info["annotations"].keys():
            
            ann_file_list = info["annotations"][class_id]
            class_id = int(class_id)
            
            class_hm = torch.zeros((1, self.gt_size, self.gt_size))
            for ann_file in ann_file_list:
                
                ann_path = self.src_path + ann_file
                with open(ann_path, "r") as ann_json:
                    ann_data = json.load(ann_json)
                
                center_x = ann_data["center_x"]
                center_y = ann_data["center_y"]
                w = ann_data["w"]
                h = ann_data["h"]
                
                # scaling
                scale_x, scale_y = CoordinateTransform(center_x, center_y,
                                                       image_width, image_height, 
                                                       self.gt_size, self.gt_size)
                
                w_scale = math.ceil((self.gt_size / image_width) * w)
                h_scale = math.ceil((self.gt_size / image_height) * h)
                
                ct_x = math.floor(scale_x)
                ct_y = math.floor(scale_y)
                
                offset_x = float(scale_x - ct_x)
                offset_y = float(scale_y - ct_y)
                
                soft_gt = gaussian_heatmap([ct_x], [ct_y], self.gt_size, sigma=1.0)  # 1, 128, 128
                
                class_hm += soft_gt
                
                wh[0, ct_y, ct_x] = w_scale
                wh[1, ct_y, ct_x] = h_scale
                
                offset[0, ct_y, ct_x] = offset_x
                offset[1, ct_y, ct_x] = offset_y
                
                mask[0, ct_y, ct_x] = 1.0
                
            hm[class_id:class_id+1, :, :] = class_hm
                
        return img_tensor, event_tensor, hm, wh, offset, mask

