import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from utils.highDHA_utils import initialize_weights
from .models.models_Swin_mae import MaskedAutoencoderViT, Swin_MAE_Segmenter
from einops import rearrange
from .models.SwinLSTM_B import SwinLSTM

BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


from .models.frft_layers import FrFTLayer

class PhaseAwareWeight(nn.Module):
    """相位感知权重生成（修复输入维度问题）"""
    def __init__(self, in_channels):
        super().__init__()
        self.phase_conv = nn.Sequential(
            # 输入通道数改为实际相位图的通道维度
            nn.Conv2d(in_channels, in_channels//8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels//8, 2, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, phase):
        """
        输入phase形状应为 [B, C, H, W]
        输出w_amp/w_phase形状为 [B, 1, H, W]
        """
        # 直接处理原始相位图（无需unsqueeze）
        weights = self.phase_conv(phase)  # [B,2,H,W]
        w_amp, w_phase = weights.chunk(2, dim=1)  # 沿通道维度分割
        return w_amp, w_phase

class FrFDLConv(nn.Module):
    """分数傅里叶动态大尺度卷积块（替换原DLK）"""
    def __init__(self, dim, h=128, w=128, order=0.5):
        super().__init__()
        # 可学习分数阶参数（二维分离变换）
        self.frft_h = FrFTLayer(order, dim=-2, trainable=True)
        self.frft_w = FrFTLayer(order, dim=-1, trainable=True)
        # self.frft_hb = FrFTLayer(order=-0.5, dim=-2, trainable=True)
        # self.frft_wb = FrFTLayer(order=-0.5, dim=-1, trainable=True)
        
        # 动态卷积层（匹配论文的5x5和7x7深度可分离卷积）
        self.att_conv1 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3)
        
        # 相位感知权重生成
        self.phase_att = PhaseAwareWeight(in_channels=dim) 
        
        # 空间注意力机制（AVP+MAP结构）
        self.spatial_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(2, 1, kernel_size=1),
            nn.Sigmoid()         
        )
        
        # 残差缩放因子（可学习）
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 分数傅里叶变换（二维分离变换）
        x_frft = self.frft_h(x)
        x_frft = self.frft_w(x_frft)
        
        # 分离幅度与相位（论文式2扩展）
        mag = torch.abs(x_frft)
        phase = torch.angle(x_frft)
        
        # 动态卷积处理（匹配论文T1/T2生成）
        T1 = self.att_conv1(mag)
        T2 = self.att_conv2(T1)
        
        # 相位权重调制（新增模块）
        w_amp, w_phase = self.phase_att(phase)
        T = w_amp * T1 + w_phase * T2
        
        # 空间注意力计算（融合实部与虚部）
        spatial_att = self.spatial_att(
            torch.cat([T.mean(dim=1, keepdim=True),  T.max(dim=1, keepdim=True).values], dim=1)  # 拼接后 [B,2,H,W]
        )  # 输出形状 [B,1,H,W]
        
        # 逆分数傅里叶变换（保持实数输出）
        output = self.frft_w(T * spatial_att).real
        output = self.frft_h(output).real
        
        # 残差连接（论文式2最后一项）
        return x + self.gamma * output


class FFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        
        return x
        
class IFFT(nn.Module):
    def __init__(self, dim, h=128, w=128):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(int((w+2)/2), h, dim, 2, dtype=torch.float32) * 0.02)
        self.W = w
        self.H = h

    def forward(self, x):

        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.W, self.H), dim=(1, 2), norm='ortho')
        return x 

class ComplexConv2d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    def apply_complex(self, fr, fi, input, dtype = torch.complex64):
        return (fr(input.real)-fi(input.imag)).type(dtype) \
                + 1j*(fr(input.imag)+fi(input.real)).type(dtype)
            
    def forward(self, input):    
        return self.apply_complex(self.conv_r, self.conv_i, input)


class FDLK(nn.Module):
    def __init__(self, dim, h=128, w=128):
        super().__init__()
        self.frft_conv = FrFDLConv(dim, h=h, w=w, order=0.4)
        
    def forward(self, x):
        return self.frft_conv(x)

class DLK(nn.Module):
    def __init__(self, dim, h=128, w=128):
        super().__init__()

        self.att_conv1 = ComplexConv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.att_conv2 = ComplexConv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)

        self.spatial_se = nn.Sequential(
            ComplexConv2d(in_channels=2, out_channels=2, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.fft = FFT()
        self.ifft = IFFT(dim, h=h, w=w)
        
    def forward(self, x):
        
        x_fft = self.fft(x)
        
        att1 = self.att_conv1(x_fft)
        att2 = self.att_conv2(att1)
        
        att = torch.cat([att1, att2], dim=1)
        
        realmean = torch.mean(att.real, dim=1, keepdim=True)
        imagmean = torch.mean(att.imag, dim=1, keepdim=True)
        avg_att = torch.complex(realmean, imagmean)
        
        realmax,_ = torch.max(att.real, dim=1, keepdim=True)
        imagmax,_ = torch.max(att.imag, dim=1, keepdim=True)
        max_att = torch.complex(realmax, imagmax)
        
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:,0,:,:].unsqueeze(1) + att2 * att[:,1,:,:].unsqueeze(1)
        
        output = rearrange(output, 'B C H W  -> B H W C')
        output = self.ifft(output)
        output = rearrange(output, 'B H W C  -> B C H W')
        
        output = output + x
        
        return output


class HMDADANet(nn.Module):

    def __init__(self, band, patchsize, num_classes, p_ratio, num_channels = 64):
        super(HMDADANet, self).__init__()
        
        self.patchsize = patchsize
        self.DLKModule = DLK(num_channels, h=patchsize, w=patchsize)
        self.FDLKModule = FDLK(num_channels, h=patchsize, w=patchsize)

        # stem net for hsi
        self.conv1 = nn.Conv2d(band, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        # stem net for msi
        self.conv_msi = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_msi = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        # stem net for sar
        self.conv_sar = nn.Conv2d(2, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_sar = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
                
        self.MAE_dim = 96
        window_size = 4
        self.patch_size = 2

        self.Swin_MAE = MaskedAutoencoderViT(img_size=(patchsize, patchsize), 
                        in_chans=2*num_channels, patch_size=self.patch_size, 
                        embed_dim=self.MAE_dim, decoder_embed_dim=self.MAE_dim, 
                        mlp_ratio=4.,p_ratio=p_ratio, memory_num = 3, 
                        window_size = window_size)
        
        self.Swin_MAE_Segmenter = Swin_MAE_Segmenter(self.Swin_MAE.Encoder, 
                        dim = self.MAE_dim)
        
        out_channel = 64 #self.MAE_dim
        self.convx1 = nn.Conv2d(self.MAE_dim, out_channel, 1, 1)              
        self.convx2 = nn.Conv2d(self.MAE_dim*2, out_channel, 1, 1)
        self.convx3 = nn.Conv2d(self.MAE_dim*4, out_channel, 1, 1)
        self.convy1 = nn.Conv2d(self.MAE_dim, out_channel, 1, 1)
        self.convy2 = nn.Conv2d(self.MAE_dim*2, out_channel, 1, 1)
        self.convy3 = nn.Conv2d(self.MAE_dim*4, out_channel, 1, 1)
        self.convz1 = nn.Conv2d(self.MAE_dim, out_channel, 1, 1)
        self.convz2 = nn.Conv2d(self.MAE_dim*2,out_channel, 1, 1)
        self.convz3 = nn.Conv2d(self.MAE_dim*4, out_channel, 1, 1)
   
 
        self.swinLSTM = SwinLSTM(img_size=(patchsize//self.patch_size, patchsize//self.patch_size), patch_size=1,
                         in_chans=out_channel, embed_dim=out_channel,
                         depths=(1, 1), num_heads=(1, 1),
                         window_size=window_size, drop_rate = 0.,
                         attn_drop_rate = 0., drop_path_rate=0.1)
   
        self.fuse_out = (out_channel)*6
        
        self.transconv = nn.Sequential(

            nn.ConvTranspose2d(self.fuse_out, 128, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),

        )
        self.final_conv = nn.Conv2d(128, num_classes, 1, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x, y, D, domain):
        _, _, height, width = x.shape #? 10 128 128

        ## 
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        y = self.relu(self.bn_msi(self.conv_msi(y)))

        x1 = self.DLKModule(x)
        x2 = self.FDLKModule(x)
        x = torch.cat([x1, x2], dim=1)  # [B,2C,H,W]

        y1 = self.DLKModule(y)
        y2 = self.FDLKModule(y)
        y = torch.cat([y1, y2], dim=1)  # [B,2C,H,W]
                
        loss = self.Swin_MAE(x, y, mask_ratio=0.75)
        x_list, y_list = self.Swin_MAE_Segmenter(x, y)# ? 8 8 384*3
        
        x0_h = self.patchsize//self.patch_size
        x0_w = self.patchsize//self.patch_size
        

        x1 = x_list[0]
        x1 = self.convx1(x1)
        x2 = F.interpolate(x_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = self.convx2(x2)
        x3 = F.interpolate(x_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = self.convx3(x3)

        y1 = y_list[0]
        y1 = self.convy1(y1)
        y2 = F.interpolate(y_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y2 = self.convy2(y2)
        y3 = F.interpolate(y_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y3 = self.convy3(y3)
        
        outputs = []
        states = [None] * 2

        x1 = rearrange(x1, 'B C H W -> B H W C')
        x2 = rearrange(x2, 'B C H W -> B H W C')
        x3 = rearrange(x3, 'B C H W -> B H W C')
        y1 = rearrange(y1, 'B C H W -> B H W C')
        y2 = rearrange(y2, 'B C H W -> B H W C')
        y3 = rearrange(y3, 'B C H W -> B H W C')
        
        firstinput = (x1, y1, x2, y2, x3)
        last_input = y3
        for i in range(5):
            output, states = self.swinLSTM(firstinput[i], states)
            outputs.append(output)

        for i in range(1):
            output, states = self.swinLSTM(last_input, states)
            outputs.append(output)

        x = torch.cat(outputs, 3)## 4 2016 32 32
        x = rearrange(x, 'B H W C -> B C H W')#'''
        ## domain adaptation
        if domain == 'source':
            xt = x
        if domain == 'target':
            xt = x
            
        ### head
        out = self.transconv(xt)
        out = self.final_conv(out)
        
        return x, x_list[2], y_list[2], out, loss
