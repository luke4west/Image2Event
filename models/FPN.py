import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class FPN(nn.Module):
    """
    Feature Pyramid Network
    """
    def __init__(self, features, keypoint_num=17):
        super().__init__()

        self.features = features

        channels = [128, 256, 512]
        self.lateral3 = nn.Conv2d(channels[0], 256, 1)
        self.lateral4 = nn.Conv2d(channels[1], 256, 1)
        self.lateral5 = nn.Conv2d(channels[2], 256, 1)
        # self.pyramid6 = nn.Conv2d(channels[2], 256, 3, stride=2, padding=1)
        # self.pyramid7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        # self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        # self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        # self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)

        hidden_dim = 64
        # 中心点位置预测
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, keypoint_num, 1))
        
        # 宽高预测
        self.wh_head = nn.Sequential(
            nn.Conv2d(256, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1))

        # 中心点右下角偏移预测
        self.reg_head = nn.Sequential(
            nn.Conv2d(256, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1))

    def forward(self, x):
        c3, c4, c5 = self.features(x)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p4 = F.interpolate(p5, scale_factor=2) + p4
        p3 = self.lateral3(c3)
        p3 = F.interpolate(p4, scale_factor=2) + p3

        heatmap = self.cls_head(p3).sigmoid_()
        wh = self.wh_head(p3)
        offset = self.reg_head(p3)

        return heatmap, wh, offset


def ResNet18FPN():
     return FPN(timm.create_model("resnet18", in_chans=1, pretrained=True, features_only=True, out_indices=[2, 3, 4]))


def Res2NetFPN():
     return FPN(timm.create_model("dla60_res2net", in_chans=1, pretrained=True, features_only=True, out_indices=[2, 3, 4]))


if __name__ == '__main__':
    # model = timm.create_model("dla60_res2net", in_chans=1, pretrained=True, features_only=True, out_indices=[2, 3, 4])
    # output = model(torch.randn(1, 1, 448, 448))
    # for o in output:
    #     print(o.shape)
    #
    # model_list = timm.list_models("*res2*")
    # print(model_list)
    model = Res2NetFPN()
    output = model(torch.randn(1, 1, 448, 448))
    for o in output:
        print(o.shape)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
