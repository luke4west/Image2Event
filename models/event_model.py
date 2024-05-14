import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


# detection decoder
class FPN(nn.Module):
    """
    Feature Pyramid Network
    """
    def __init__(self, keypoint_num=17):
        super().__init__()
        
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
        c3, c4, c5 = x

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p4 = F.interpolate(p5, scale_factor=2) + p4
        p3 = self.lateral3(c3)
        p3 = F.interpolate(p4, scale_factor=2) + p3

        heatmap = self.cls_head(p3).sigmoid_()
        wh = self.wh_head(p3)
        offset = self.reg_head(p3)

        return heatmap, wh, offset


# generation decoder
class FCN(nn.Module):
    def __init__(self, inplanes, outplanes, bn_momentum=0.1):
        super(FCN, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = True

        # ----------------------------------------------------------#
        #   16,16,2048 -> 32,32,256 -> 64,64,128 -> 128,128,64
        #   利用ConvTranspose2d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
        # ----------------------------------------------------------#
        self.deconv_layers = self._make_deconv_layer(
            num_layers=4,
            num_filters=[256, 128, 64, 32],
            num_kernels=[4, 4, 4, 4,],
        )
        self.smooth = nn.Conv2d(32, outplanes, 1, 1, 0)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.smooth(x)
        return x


# loss functions
def generator_loss(fake):
    return -fake.mean()


def generator_loss_two_sensors(sensor_a=None, sensor_b=None):
    if sensor_a is None:
        return -sensor_b.mean()
    elif sensor_b is None:
        return sensor_a.mean()
    else:
        return sensor_a.mean() - sensor_b.mean()
    

def event_reconstruction_loss(gt_histogram, predicted_histogram):
    l1_distance = torch.abs(gt_histogram - predicted_histogram).sum(dim=1)
    bool_zero_cells = gt_histogram.sum(dim=1) > 0

    if torch.logical_not(bool_zero_cells).sum() == 0 or bool_zero_cells.sum() == 0:
        return l1_distance.mean()

    return l1_distance[bool_zero_cells].mean() + l1_distance[torch.logical_not(bool_zero_cells)].mean()


def discriminator_loss(real=torch.ones([1]), fake=-torch.ones([1])):
    real_loss = F.relu(1.0 - real).mean()
    fake_loss = F.relu(1.0 + fake).mean()
    return real_loss + fake_loss


def gradient(I):
    """
    Arguments:
    - I - shape N1,...,Nn,C,H,W
    Returns:
    - dx - shape N1,...,Nn,C,H,W
    - dy - shape N1,...,Nn,C,H,W
    """

    dy = I.new_zeros(I.shape)
    dx = I.new_zeros(I.shape)

    dy[...,1:,:] = I[...,1:,:] - I[...,:-1,:]
    dx[...,:,1:] = I[...,:,1:] - I[...,:,:-1]

    return dx, dy


def flow_smoothness(flow, mask=None):
    dx, dy = gradient(flow)
    
    if mask is not None:
        mask = mask.expand(-1,2,-1,-1)
        loss = (charbonnier_loss(dx[mask]) \
                + charbonnier_loss(dy[mask])) / 2.
    else:
        loss = (charbonnier_loss(dx) \
                + charbonnier_loss(dy)) / 2.
        
    return loss


def charbonnier_loss(error, alpha=0.45, mask=None):
    charbonnier = (error ** 2. + 1e-5 ** 2.) ** alpha
    if mask is not None:
        mask = mask.float()
        loss = torch.mean(torch.sum(charbonnier * mask, dim=(1, 2, 3)) / \
                          torch.sum(mask, dim=(1, 2, 3)))
    else:
        loss = torch.mean(charbonnier)
    return loss


def gradient_loss(event):
    dx, dy = gradient(event)
    
    gradient_magnitude = torch.sqrt(dx**2 + dy**2)
    
    threshold = 0.7
    mask = gradient_magnitude > threshold

    # 对梯度值大于 0.7 的位置进行梯度求和
    sum_of_gradients = torch.sum(gradient_magnitude[mask])
    
    return sum_of_gradients
    

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)
        
        
class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        if sn:
            model += [torch.nn.utils.spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride,
                                                             padding=padding, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        if 'norm' == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        #elif == 'Group'

    def forward(self, x):
        return self.model(x)
    

# feature discriminator
class ContentDiscriminator(nn.Module):
    def __init__(self, nr_channels, smaller_input=False):
        super(ContentDiscriminator, self).__init__()
        model = []
        model += [LeakyReLUConv2d(nr_channels, nr_channels, kernel_size=7, stride=2, padding=1, norm='Instance')]
        # model += [LeakyReLUConv2d(nr_channels, nr_channels, kernel_size=7, stride=2, padding=1, norm='Instance')]
        if smaller_input:
            model += [LeakyReLUConv2d(nr_channels, nr_channels, kernel_size=4, stride=1, padding=1, norm='Instance')]
        else:
            model += [LeakyReLUConv2d(nr_channels, nr_channels, kernel_size=7, stride=1, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(nr_channels, nr_channels, kernel_size=4, stride=1, padding=0)]
        model += [nn.Conv2d(nr_channels, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)

        return out
    

# refine discriminator
class CrossDiscriminator(nn.Module):
    def __init__(self, input_dim, n_layer=6, norm='None', sn=True):
        super(CrossDiscriminator, self).__init__()
        ch = 64
        self.model = self._make_net(ch, input_dim, n_layer, norm, sn)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] #16
        tch = ch

        for i in range(1, n_layer-1):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] # 8
            tch *= 2

        model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)] # 2
        tch *= 2
        if sn:
            model += [torch.nn.utils.spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]  # 1
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1

        return nn.Sequential(*model)

    def cuda(self, gpu):
        self.model.cuda(gpu)

    def forward(self, x_A):
        out_A = self.model(x_A)

        return out_A


class Frame2Event(nn.Module):
    def __init__(self, 
                 backbone="dla60_res2net",
                 keypoint_num=2, # num_class of bounding boxes
                 ):
        super().__init__()
        
        dim = 512
        hidden_dim = 64
        
        # ==================MAIN PART==================
        # frame encoder
        self.encoder_f = timm.create_model(backbone, in_chans=3, pretrained=True, features_only=True, out_indices=[2, 3, 4])
        # event encoder
        self.encoder_e = timm.create_model(backbone, in_chans=1, pretrained=True, features_only=True, out_indices=[2, 3, 4])
        # detection decoder
        self.decoder_det = FPN(keypoint_num)
        
        # ==================GEN PART==================
        # event-specific head --> get event-specific features
        self.transform_e = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1))
        # generation decoder --> upsample to input_size
        self.decoder_gen = FCN(inplanes=dim+hidden_dim, outplanes=2, )
        # refine module --> incorporate with input frame
        self.refine = nn.Sequential(
            nn.Conv2d(3+2, 24, 3, 1, 1),
            nn.GroupNorm(1, 24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 1, 1, 1, 0))

        # ==================DIS PART================== 
        self.content_disc = ContentDiscriminator(512)
        self.refine_disc = CrossDiscriminator(1)
        
    def illustration(self, x_img, x_event):
        # -----------Main Flow-----------
        # 1.1 feature extraction: get feature list
        features_img = self.encoder_f(x_img)
        features_event = self.encoder_e(x_event)
        
        # 1.2 detection on img
        hm_img, wh_img, offset_img = self.decoder_det(features_img)
        
        # 2.1 get event-specific features
        zeta_event = self.transform_e(features_event[-1])
        # 2.2 generation
        pseudo_flow = self.decoder_gen(torch.cat([zeta_event, features_img[-1]], dim=1))
        # 2.3 sample 
        # aug_flow = random_epipolar_augmentation(pseudo_flow)
        aug_flow = pseudo_flow
        # 2.4 refine generation using x_img
        # print(x_img.shape, aug_flow.shape)
        aug_pseudo_event = self.refine(torch.cat([x_img, aug_flow], dim=1))
        
        # operation on pseudo event
        # 3.1 feature extraction
        features_pseudo_event = self.encoder_e(aug_pseudo_event)
        # 3.2 detection
        hm_event, wh_event, offset_event = self.decoder_det(features_pseudo_event)
        # 3.3 get event-specific features
        zeta_pseudo_event = self.transform_e(features_pseudo_event[-1])
        # 3.4 augment reconstruction
        reconstructed_flow = self.decoder_gen(torch.cat([zeta_pseudo_event, features_img[-1]], dim=1))
        
        # -----------Extra Losses-----------
        # cycle loss 
        loss_cycle = F.l1_loss(features_img[-1], features_pseudo_event[-1]) + \
            F.l1_loss(zeta_event, zeta_pseudo_event)
        
        # # latent features
        # loss_lat = discriminator_loss(features_img[-1], features_event[-1]) + \
        #     generator_loss_two_sensors(features_img[-1], features_event[-1])
        # # events
        # loss_event = discriminator_loss(aug_pseudo_event, x_event) + \
        #     generator_loss(aug_pseudo_event)
        
        # generator
        loss_gen = generator_loss_two_sensors(features_img[-1], features_event[-1]) + generator_loss(aug_pseudo_event)
        
        # discriminator
        loss_disc = discriminator_loss(features_img[-1], features_event[-1]) + discriminator_loss(aug_pseudo_event, x_event)
        
        # augment
        loss_augment = F.l1_loss(reconstructed_flow, aug_flow)
        # smooth
        loss_smooth = flow_smoothness(pseudo_flow) + flow_smoothness(reconstructed_flow)
        
        # return [hm_img, wh_img, offset_img], [hm_event, wh_event, offset_event], loss_gen + loss_cycle + loss_augment + loss_smooth, loss_disc,
    
    def discrimination_step(self, x_img, x_event):
        with torch.no_grad():
            features_img = self.encoder_f(x_img)
            features_event = self.encoder_e(x_event)
        
        loss_content = discriminator_loss(self.content_disc(features_img[-1]), 
                                          self.content_disc(features_event[-1]))
        
        with torch.no_grad():
            # specific features
            specific_event = self.transform_e(features_event[-1])
            # flow generation
            pseudo_flow = self.decoder_gen(torch.cat([specific_event, features_img[-1]], dim=1))
            # 2.3 sample 
            # aug_flow = random_epipolar_augmentation(pseudo_flow)
            aug_flow = pseudo_flow
            # 2.4 refine generation using x_img
            # print(x_img.shape, aug_flow.shape)
            aug_pseudo_event = self.refine(torch.cat([x_img, aug_flow], dim=1))
        
        loss_event = discriminator_loss(self.refine_disc(aug_pseudo_event), 
                                        self.refine_disc(x_event))
        
        return loss_content + loss_event

    def generation_step(self, x_img, x_event):
        features_img = self.encoder_f(x_img)
        features_event = self.encoder_e(x_event)
        
        # detection on img
        hm_img, wh_img, offset_img = self.decoder_det(features_img)
        
        # generating fake event
        ## specific features
        specific_event = self.transform_e(features_event[-1])
        ## flow generation
        pseudo_flow = self.decoder_gen(torch.cat([specific_event, features_img[-1]], dim=1))
        ## sample 
        aug_flow = pseudo_flow
        ## refine generation using x_img
        aug_pseudo_event = self.refine(torch.cat([x_img, aug_flow], dim=1))
        
        # operation on fake event
        ## feature extraction
        features_pseudo_event = self.encoder_e(aug_pseudo_event)
        ## detection
        hm_event, wh_event, offset_event = self.decoder_det(features_pseudo_event)
        ## get event-specific features
        specific_pseudo_event = self.transform_e(features_pseudo_event[-1])
        ## augment reconstruction
        reconstructed_flow = self.decoder_gen(torch.cat([specific_pseudo_event, features_img[-1]], dim=1))
        
        # loss
        # print(features_img[-1].shape, features_event[-1].shape)
        loss_content = generator_loss_two_sensors(self.content_disc(features_img[-1]), self.content_disc(features_event[-1]))
        loss_event = generator_loss_two_sensors(self.refine_disc(aug_pseudo_event), None)
        loss_cycle = F.l1_loss(features_img[-1], features_pseudo_event[-1]) + \
            F.l1_loss(specific_event, specific_pseudo_event)
        loss_augment = F.l1_loss(reconstructed_flow, aug_flow)
        loss_smooth = flow_smoothness(pseudo_flow) + flow_smoothness(reconstructed_flow)
        
        return [hm_img, wh_img, offset_img], [hm_event, wh_event, offset_event], loss_content + loss_event + loss_cycle + loss_augment + loss_smooth
       
    def detection_step_img(self, x_img):
        features_img = self.encoder_f(x_img)
        hm_img, wh_img, offset_img = self.decoder_det(features_img)
        return hm_img, wh_img, offset_img
    
    def detection_step_fake_event(self, x_img, x_event):
        features_img = self.encoder_f(x_img)
        features_event = self.encoder_e(x_event)
        
        # detection on img
        hm_img, wh_img, offset_img = self.decoder_det(features_img)
        
        # generating fake event
        ## specific features
        specific_event = self.transform_e(features_event[-1])
        ## flow generation
        pseudo_flow = self.decoder_gen(torch.cat([specific_event, features_img[-1]], dim=1))
        ## sample 
        aug_flow = pseudo_flow
        ## refine generation using x_img
        aug_pseudo_event = self.refine(torch.cat([x_img, aug_flow], dim=1))
        
        # operation on fake event
        ## feature extraction
        features_pseudo_event = self.encoder_e(aug_pseudo_event)
        ## detection
        hm_event, wh_event, offset_event = self.decoder_det(features_pseudo_event)
        
        return [hm_img, wh_img, offset_img], [hm_event, wh_event, offset_event], aug_pseudo_event
        
    def detection_step_event(self, x_event):
        features_event = self.encoder_e(x_event)
        hm_event, wh_event, offset_event = self.decoder_det(features_event)
        return hm_event, wh_event, offset_event
    

if __name__ == '__main__':
    inputs_f = torch.randn(4, 3, 448, 448)
    inputs_e = torch.randn(4, 1, 448, 448)
    
    model = Frame2Event()
    o1, o2, loss_gen = model.generation_step(inputs_f, inputs_e)
    for o in o1:
        print(o.shape)
    print()
    for o in o2:
        print(o.shape)
    print()
    print(loss_gen)
    print("-----------------------------------")
    loss_disc = model.discrimination_step(inputs_f, inputs_e)
    print(loss_disc)