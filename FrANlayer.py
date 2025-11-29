import torch
import torch.nn as nn

class ChirpFANLayer(nn.Module):
    """支持多维度输入的 Chirp-FAN 层"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 p_ratio=None, activation=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 傅里叶分量参数
        self.p_ratio = p_ratio
        self.p_dim = int(hidden_features * p_ratio)
        self.g_dim = hidden_features - 2 * self.p_dim
        
        # 调频参数投影
        self.proj_f = nn.Linear(in_features, self.p_dim)
        self.proj_beta = nn.Linear(in_features, self.p_dim)
        
        # 非周期分量投影
        self.proj_g = nn.Linear(in_features, self.g_dim)
        
        # 输出投影
        self.fc_out = nn.Linear(hidden_features, out_features)
        
        # 激活与正则化
        self.activation = activation()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        """支持输入形状: (B, H, W, C) 或 (B, N, C)"""
        # 统一输入为 (B, N, C)
        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.view(B, H*W, C)  # (B, H*W, C)
        elif x.dim() == 2:
            x = x.unsqueeze(1)     # (B, 1, C)
        B, N, C = x.shape
        
        # 生成空间坐标网格
        grid = self._get_spatial_grid(N, x.device)  # (N, 1)
        
        # 调频分量计算
        f = self.proj_f(x)       # (B, N, p_dim)
        beta = self.proj_beta(x) # (B, N, p_dim)
        chirp_arg = 2 * torch.pi * f * grid + beta * grid**2
        # chirp_arg = 2 * torch.pi * f * grid
        p_cos = torch.cos(chirp_arg)
        p_sin = torch.sin(chirp_arg)
        
        # 非周期分量
        g = self.activation(self.proj_g(x))  # (B, N, g_dim)
        
        # 拼接并投影
        x = torch.cat([p_cos, p_sin, g], dim=-1)
        x = self.dropout(x)
        x = self.fc_out(x)
        
        # 恢复原始形状（若输入为四维）
        if x.dim() == 3 and N != 1:
            x = x.view(B, int(N**0.5), int(N**0.5), -1)
        return x

    def _get_spatial_grid(self, num_patches, device):
        """生成归一化的空间坐标网格"""
        if num_patches == 1:
            return torch.zeros(1, 1, device=device)
        
        grid_size = int(num_patches**0.5)
        if grid_size**2 != num_patches:
            # 非正方形输入（如全连接层）
            grid = torch.linspace(-1, 1, num_patches, device=device).unsqueeze(-1)
            return grid
        else:
            # 正方形输入（如图像块）
            grid = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, grid_size, device=device),
                torch.linspace(-1, 1, grid_size, device=device)
            ), dim=-1).view(-1, 2)
            return grid.mean(dim=-1, keepdim=True)