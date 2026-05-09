import copy
from typing import Optional, List
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat
from util.misc import NestedTensor, is_main_process

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            # [修改 1] 将 padding=0 改为 padding=1 (Same Padding)
            # 这样卷积后尺寸不变，不再需要巨大的外部 Padding
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            # [修改 1] 同样改为 padding=1
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_ch=1, out_ch=4):
        super(Unet, self).__init__()

        n1 = 64
        filters = [64, 128, 256, 512, 1024]

        # [修改 2] 删除了 self.Pad = nn.ConstantPad2d((92, 92, 92, 92), 0)
        # 不再需要这个巨大的 Padding，FLOPs 将降低约 3-4 倍
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up4 = up_conv(filters[4], 4)
        self.Up_conv4 = conv_block(516, filters[3])

        self.Up3 = up_conv(filters[3], 4)
        self.Up_conv3 = conv_block(260, filters[2])
        
        self.Up2 = up_conv(filters[2], filters[1])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Norm = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)

        self.active = torch.nn.Softmax(dim=1)

    def forward(self, tensor_list):
        x = tensor_list

        # [修改 3] 处理输入尺寸 (212x212)
        # 212 不能被 16 整除，会导致下采样和上采样尺寸对不上。
        # 我们临时 pad 到 224 (最接近的 16 倍数)，不仅解决了 bug，计算量也远小于之前的 396x396
        _, _, h, w = x.shape
        target_h = ((h - 1) // 16 + 1) * 16
        target_w = ((w - 1) // 16 + 1) * 16
        pad_h = (target_h - h) // 2
        pad_w = (target_w - w) // 2
        
        # 动态 Padding (上下左右补齐到 224)
        x = F.pad(x, (pad_w, target_w - w - pad_w, pad_h, target_h - h - pad_h), mode='constant', value=0)

        # [修改 4] 删除了 x1 = self.Pad(x)，直接使用 padding 后的 x
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # Decoder 部分
        d4 = self.Up4(e5)
        # [修改 5] 删除了 e4_cropped = e4[:,:,4:38...] 这种硬编码裁剪
        # 因为我们改用了 padding=1，特征图大小会自动对齐，直接 concat 即可
        d4 = torch.cat((d4, e4), dim=1) 
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # 直接 concat
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # 直接 concat
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        # 直接 concat
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.Up_conv1(d1)

        d0 = self.Conv(d1)
        norm_out = self.Norm(d0)
        out = self.active(norm_out)

        # [修改 6] 裁剪回原始尺寸 (212x212)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, pad_h:pad_h+h, pad_w:pad_w+w]

        return out

def build_UNet(args):
    return Unet(in_ch=1, out_ch=4)