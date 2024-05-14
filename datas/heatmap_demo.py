import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch


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


def gaussian_heatmap(X_t, Y_t, target_size, sigma=2.0):
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
        # print(torch.from_numpy(heatmap).unsqueeze(0).shape)
        g_heatmap.append(torch.from_numpy(heatmap).unsqueeze(0))

    heatmap_gt = torch.cat(g_heatmap, dim=0)
    return heatmap_gt


x = [24, 90,]
y = [90, 90,]
gt_size = 128
sigma = 1.1
print(sigma)
soft_gt = gaussian_heatmap(x, y, gt_size, sigma=sigma)
print(soft_gt.shape, soft_gt.dtype)
visual_gt, _ = torch.max(soft_gt, dim=0)
gt_mask = visual_gt.numpy()
plt.imshow(visual_gt.numpy(), cmap="coolwarm")
plt.savefig("datas/sigma_{}.png".format(sigma))
plt.close()