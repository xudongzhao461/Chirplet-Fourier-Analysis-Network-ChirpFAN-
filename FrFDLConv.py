import torch
import torch.nn as nn
import torch.nn.functional as F
from models.frft_layers import FrFTLayer

class ComplexDynamicConv(nn.Module):
    """复数动态卷积（分离处理实部与虚部）"""
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.conv_real = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2, groups=in_ch)
        self.conv_imag = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2, groups=in_ch)
        
    def forward(self, x_real, x_imag):
        return self.conv_real(x_real), self.conv_imag(x_imag)

class PhaseAwareWeight(nn.Module):
    """相位感知权重生成"""
    def __init__(self, channels):
        super().__init__()
        self.phase_conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2, kernel_size=1)  # 输出幅度和相位权重
        )
        
    def forward(self, phase):
        weights = torch.sigmoid(self.phase_conv(phase.unsqueeze(1)))
        return weights[:,0], weights[:,1]

class FrFDLConv(nn.Module):
    """分数傅里叶动态大尺度卷积块（论文III-B节适配版）"""
    def __init__(self, dim, order=0, kernel_sizes=(5,7)):
        super().__init__()
        # 可学习分数阶参数（约束在0-2区间）
        self.frft_h = FrFTLayer(order, dim=-2, trainable=True)
        self.frft_w = FrFTLayer(order, dim=-1, trainable=True)
        
        # 动态卷积层（匹配论文的5x5和7x7 DWConv）
        self.dw_conv1 = ComplexDynamicConv(dim, dim, kernel_sizes)
        self.dw_conv2 = ComplexDynamicConv(dim, dim, kernel_sizes)
        
        # 相位感知权重生成
        self.phase_weight = PhaseAwareWeight(dim)
        
        # 空间注意力（论文中的AVP+MAP+Conv）
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
        
        # 分离幅度与相位
        mag = torch.abs(x_frft)
        phase = torch.angle(x_frft)
        
        # 动态卷积处理（实部与虚部分离）
        t1_real, t1_imag = self.dw_conv1(mag, phase)
        t2_real, t2_imag = self.dw_conv2(t1_real, t1_imag)
        
        # 相位感知权重融合
        w_amp, w_phase = self.phase_weight(phase)
        t_real = w_amp * t1_real + (1-w_amp) * t2_real
        t_imag = w_phase * t1_imag + (1-w_phase) * t2_imag
        
        # 空间注意力（论文式2）
        spatial_att = self.spatial_att(
            torch.cat([t_real.mean(dim=1, keepdim=True), 
                      t_imag.mean(dim=1, keepdim=True)], dim=1)
        )
        
        # 逆分数傅里叶变换（保持实数输出）
        output = self.frft_w(t_real + 1j*t_imag, inverse=True).real
        output = self.frft_h(output, inverse=True).real
        
        # 残差连接（论文式2最后一项）
        return x + self.gamma * (output * spatial_att)