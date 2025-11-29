# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import logging
import pdb
import numpy as np
from einops import rearrange

from timm.models.vision_transformer import Block
from ..models.swin_unet import PatchEmbedding, BasicBlock, PatchExpanding, BasicBlockUp

#from util.pos_embed import get_2d_sincos_pos_embed
from ..models.utils.pos_embed import get_2d_sincos_pos_embed
from ..models.corss_transformers import CMAttention
from ..models.SwinLSTM_B import SwinLSTM

# copied from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
        
        
class Swin_MAE_Encoder(nn.Module):
    """ 
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3,
                 embed_dim=1024, depth=(2, 2, 6, 2), num_heads=(2, 6, 12, 24), 
                 decoder_depth=(2, 2), decoder_num_heads=(6, 12), 
                 window_size = 8, norm_layer=nn.LayerNorm, p_ratio=0.25):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

       ## modality HSI
        self.patch_embed_x = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed_x.num_patches

        self.cls_token_x = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        ## modality MSI
        self.patch_embed_y = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        #num_patches = self.patch_embed.num_patches

        self.cls_token_y = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_y = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        ## modality SAR
        self.patch_embed_z = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        #num_patches = self.patch_embed.num_patches

        self.cls_token_z = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_z = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        '''
        self.blocks_x = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_x = norm_layer(embed_dim)
        self.blocks_y = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_y = norm_layer(embed_dim)
        self.blocks_z = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_z = norm_layer(embed_dim)#'''

        self.mask_token_x = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token_y = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token_z = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        '''
        Encoder Part  patch降维，通道数升维
        '''
        Encoder_spec_layer = 2
        self.blocks_x = self.build_layers_speci(embed_dim=embed_dim, window_size = window_size, 
                                                depths = depth, num_heads = num_heads,p_ratio=p_ratio,
                                                num_layers = Encoder_spec_layer)
        self.norm_x = norm_layer(embed_dim*Encoder_spec_layer)
        self.blocks_y = self.build_layers_speci(embed_dim=embed_dim, window_size = window_size, 
                                                depths = depth, num_heads = num_heads,p_ratio=p_ratio,
                                                num_layers = Encoder_spec_layer)
        self.norm_y = norm_layer(embed_dim*Encoder_spec_layer)
        self.blocks_z = self.build_layers_speci(embed_dim=embed_dim, window_size = window_size, 
                                                depths = depth, num_heads = num_heads,p_ratio=p_ratio,
                                                num_layers = Encoder_spec_layer)
                                                # 不论几层，只在最后一层通道数增加一倍
        self.norm_z = norm_layer(embed_dim*Encoder_spec_layer)
        

        Encoder_share_layer = 2
        self.blocks = self.build_layers_share(embed_dim=embed_dim*Encoder_spec_layer, 
                                              window_size = window_size, 
                                              depths = decoder_depth, num_heads = decoder_num_heads, 
                                              num_layers = Encoder_share_layer)
                                              # 不论几层，只在最后一层通道数增加一倍   
        self.norm = norm_layer(embed_dim*Encoder_spec_layer*Encoder_share_layer)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed_x = get_2d_sincos_pos_embed(self.pos_embed_x.shape[-1], self.patch_embed_x.grid_size, cls_token=True)  # modified
        self.pos_embed_x.data.copy_(torch.from_numpy(pos_embed_x).float().unsqueeze(0))
        ##
        pos_embed_y = get_2d_sincos_pos_embed(self.pos_embed_y.shape[-1], self.patch_embed_y.grid_size, cls_token=True)  # modified
        self.pos_embed_y.data.copy_(torch.from_numpy(pos_embed_y).float().unsqueeze(0))
        ##        
        pos_embed_z = get_2d_sincos_pos_embed(self.pos_embed_z.shape[-1], self.patch_embed_z.grid_size, cls_token=True)  # modified
        self.pos_embed_z.data.copy_(torch.from_numpy(pos_embed_z).float().unsqueeze(0))
        

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed_x.grid_size, cls_token=True)  # modified
        #self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        w = self.patch_embed_x.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        w = self.patch_embed_y.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        w = self.patch_embed_z.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token_x, std=.02)
        torch.nn.init.normal_(self.mask_token_x, std=.02)
        
        torch.nn.init.normal_(self.cls_token_y, std=.02)
        torch.nn.init.normal_(self.mask_token_y, std=.02)
        
        torch.nn.init.normal_(self.cls_token_z, std=.02)
        torch.nn.init.normal_(self.mask_token_z, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def window_masking(self, x: torch.Tensor, y: torch.Tensor, r: int = 4, flag: int = 0, mask_ratio: float = 0.75, 
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        #x = rearrange(x, 'B H W C -> B (H W) C')
        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0) 
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int_) 
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if flag ==0:
            mask_token = self.mask_token_x
        elif flag ==1:
            mask_token = self.mask_token_y
        else:
            mask_token = self.mask_token_z
            
        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)
            for i in range(B):   
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = mask_token
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            
            y_masked = torch.clone(y)
            for i in range(B):   
                y_masked[i, index_mask.cpu().numpy()[i, :], :] = mask_token
            y_masked = rearrange(y_masked, 'B (H W) C -> B H W C', H=int(y_masked.shape[1] ** 0.5))
            
            return x_masked, y_masked, mask
            
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_tokenizer(self, x, y, mask_ratio):
        # embed patches
        #print(11111, x.shape) # 4 64 128 128
        x = self.patch_embed_x(x)
        #print("mae的positional是："+str(x.shape))
        # add pos embed w/o cls token
        x = x + self.pos_embed_x[:, 1:, :]###维度  4 256 720
        # masking: length -> length * mask_ratio
        #x, mask_x, ids_restore_x = self.random_masking(x, mask_ratio)
        # x, mask_x = self.window_masking(x, r = 4, flag = 0, mask_ratio = mask_ratio, remove=False, mask_len_sparse=False)# B H W C
        # append cls token
        # cls_token_x = self.cls_token_x + self.pos_embed_x[:, :1, :]
        # cls_tokens_x = cls_token_x.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens_x, x), dim=1)

        y = self.patch_embed_y(y)
        #print("mae的positional是："+str(x.shape))
        # add pos embed w/o cls token
        y = y + self.pos_embed_y[:, 1:, :]###维度  4 256 720
        # print(1133, y.shape)  
        # masking: length -> length * mask_ratio
        #y, mask_y, ids_restore_y = self.random_masking(y, mask_ratio)
        # y, mask_y = self.window_masking(y, r = 4, flag = 1, mask_ratio = mask_ratio, remove=False, mask_len_sparse=False)# B H W C
        
        x, y, mask_x = self.window_masking(x, y, r = 4, flag = 0, mask_ratio = mask_ratio, remove=False, mask_len_sparse=False)# B H W C
        
        # append cls token
        # cls_token_y = self.cls_token_y + self.pos_embed_y[:, :1, :]
        # cls_tokens_y = cls_token_y.expand(y.shape[0], -1, -1)
        # y = torch.cat((cls_tokens_y, y), dim=1)###维度 4 65 720
        # print(1122, y.shape)  
        
        '''
        z = self.patch_embed_z(z)
        #print("mae的positional是："+str(x.shape))
        # add pos embed w/o cls token
        z = z + self.pos_embed_z[:, 1:, :]
        # masking: length -> length * mask_ratio
        #z, mask_z, ids_restore_z = self.random_masking(z, mask_ratio)
        z, mask_z = self.window_masking(z, r = 4, flag = 2, remove=False, mask_len_sparse=False)# B H W C'''
        # append cls token
        # cls_token_z = self.cls_token_z + self.pos_embed_z[:, :1, :]
        # cls_tokens_z = cls_token_z.expand(z.shape[0], -1, -1)
        # z = torch.cat((cls_tokens_z, z), dim=1)
        #print(0, x.shape, y.shape, z.shape)
        ## 16 16 64*4*4
        return x, y, mask_x, mask_x#, ids_restore_x, ids_restore_y, ids_restore_z
        
    def build_layers_speci(self, depths: tuple = (2, 2, 6, 2), embed_dim=96, 
                            num_heads: tuple = (2, 6, 12, 24), drop_path: float = 0.1, 
                            window_size: int = 8, mlp_ratio: float = 4.,p_ratio: float =0.25, qkv_bias: bool = True, 
                            drop_rate: float = 0., attn_drop_rate: float = 0., num_layers: int = 1, norm_layer=nn.LayerNorm):
        layers = nn.ModuleList()
        for i in range(num_layers):
            layer = BasicBlock(
                index=i,
                depths=depths,
                embed_dim=embed_dim,## 每个stage得通道数 C
                num_heads=num_heads,
                drop_path=drop_path,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                p_ratio=p_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                norm_layer=norm_layer,
                patch_merging=False if i == num_layers - 1 else True)
            layers.append(layer)
        return layers
        
    def forward_encoder_specific(self, x, y):
        # apply Transformer blocks
        for blk in self.blocks_x:
            x = blk(x)
        x = self.norm_x(x)
        
        for blk in self.blocks_y:
            y = blk(y)
        y = self.norm_y(y)
        
        '''
        for blk in self.blocks_z:
            z = blk(z)
        z = self.norm_z(z)#'''
      
        return x, y## 结构顺序：SwinT path merge SwinT    降维一次
        
    def build_layers_share(self, depths: tuple = (2, 2, 6, 2), embed_dim=96, 
                            num_heads: tuple = (2, 6, 12, 24), 
                            drop_path: float = 0.1, window_size: int = 8, mlp_ratio: float = 4., qkv_bias: bool = True, 
                            drop_rate: float = 0., attn_drop_rate: float = 0., num_layers: int = 2, norm_layer=nn.LayerNorm):
        layers = nn.ModuleList()
        for i in range(num_layers):
            layer = BasicBlock(
                index=i,
                depths=depths,
                embed_dim=embed_dim,## 每个stage得通道数 C
                num_heads=num_heads,
                drop_path=drop_path,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                norm_layer=norm_layer,
                patch_merging=False if i == num_layers - 1 else True)
            layers.append(layer)
        return layers
        
    def forward_encoder_share(self, x, y):
       
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        for blk in self.blocks:
            y = blk(y)
        y = self.norm(y) 
        
        '''
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z) #'''
        return x, y## 结构顺序：SwinT path merge SwinT   降维一次


    def forward(self, imgsx, imgsy, mask_ratio):

        latentx, latenty, mask_x, mask_y = self.forward_tokenizer(imgsx, imgsy, mask_ratio)
        
        x, y = self.forward_encoder_specific(latentx, latenty)
        x, y = self.forward_encoder_share(x, y)
        
        return x, y, mask_x, mask_y## 4 8 8 96*4  降维2次



class Swin_MAE_Dncoder(nn.Module):
    """ 
    """
    def __init__(self, imagesize, patch_size = 16, in_chans = 3, decoder_embed_dim=512, 
                 decoder_depth=(2, 2), decoder_num_heads=(6, 12),
                 mlp_ratio=4., memory_num = 3, window_size = 8, norm_layer=nn.LayerNorm):
        super().__init__()

        Encoder_number = 2 ## build_layers_speci + build_layers_share  增加一个Encoder，通道数增加一倍

        self.layers_up = self.build_layers_up_share(
                                    embed_dim=decoder_embed_dim*Encoder_number,
                                    ## 这里通道数指第一个Decoder中patch升维后的通道数，而非输入通道数
                                    window_size = window_size, 
                                    depths = decoder_depth, 
                                    num_heads = decoder_num_heads, 
                                    num_layers = Encoder_number)
                                                ## 这里的depths与前面的Encoder个数一致。
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.CMA_x = CMAttention(decoder_embed_dim, imagesize, heads = 12, 
                                dim_head = decoder_embed_dim, dropout = 0.1, memory_num = memory_num)
                                
        self.CMA_y = CMAttention(decoder_embed_dim, imagesize, heads = 12, 
                                dim_head = decoder_embed_dim, dropout = 0.1, memory_num = memory_num)
           
        self.decoder_pred_x = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.decoder_pred_y = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
       
    def build_layers_up_share(self, depths: tuple = (2, 2, 6, 2), embed_dim=96, 
                            num_heads: tuple = (2, 6, 12, 24), drop_path: float = 0.1, 
                            window_size: int = 8, mlp_ratio: float = 4., qkv_bias: bool = True, 
                            drop_rate: float = 0., attn_drop_rate: float = 0., num_layers: int = 2, 
                            norm_layer = nn.LayerNorm):
        layers = nn.ModuleList()
        self.first_patch_expanding = PatchExpanding(dim=embed_dim * 2 ** (len(depths) - 1), norm_layer=norm_layer)
        for i in range(num_layers-1):
            layer = BasicBlockUp(
                index=i,
                depths=depths,
                embed_dim=embed_dim,## 每个stage得通道数 C
                num_heads=num_heads,
                drop_path=drop_path,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                norm_layer=norm_layer,
                patch_expanding=True if i == num_layers - 2 else False)
            layers.append(layer)
        return layers
        
    def forward_decoder_share(self, x, y):
    
        x = self.first_patch_expanding(x)
        for layer in self.layers_up:
            x = layer(x)  
        x = self.decoder_norm(x)
        
        y = self.first_patch_expanding(y)
        for layer in self.layers_up:
            y = layer(y)
        y = self.decoder_norm(y)
        
        x = rearrange(x, 'B H W C -> B (H W) C')
        y = rearrange(y, 'B H W C -> B (H W) C')
        
        return x, y##先升维，后MAE（MAE结构顺序：SwinT+patch expand (再升维)+SwinT） 该部分升维2次
        
    def forward_decoder_speci(self, x, y):
    
        # predictor projection
        inp_x = x
        inp_y = y
        
        x = self.CMA_x(x, inp_x, inp_y)
        x = self.decoder_pred_x(x)
        
        y = self.CMA_y(y, inp_x, inp_y)
        y = self.decoder_pred_y(y)
        
        return x, y ## 维度不变
           
    def forward(self, x, y):
    
        x, y= self.forward_decoder_share(x, y)
        x, y= self.forward_decoder_speci(x, y)
        
        return x, y## 4 32 32 96  升维2次，恢复输入分辨率

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3,
                 embed_dim=1024, depth=(2, 2), num_heads=(2, 6), 
                 decoder_embed_dim=512, decoder_depth=(2, 2), decoder_num_heads=(6, 12),
                 mlp_ratio=4., p_ratio=0.25,memory_num = 3, window_size = 8, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.patch_size = patch_size
        self.Encoder = Swin_MAE_Encoder(img_size=img_size, patch_size=patch_size, 
                 in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads, 
                 decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads, 
                 window_size = window_size, norm_layer=norm_layer,p_ratio=p_ratio)
        
        self.Dncoder = Swin_MAE_Dncoder(img_size[0], patch_size = patch_size, in_chans = in_chans, 
                 decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, 
                 decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, 
                 memory_num = memory_num, window_size = window_size, norm_layer=norm_layer)
                 
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)  # modified
        # print(pos_embed.shape)
        # print(self.pos_embed.shape)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # h = w = imgs.shape[2] // p
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        #x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        #x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
        H = imgs.shape[2]
        W = imgs.shape[3]
        self.patch_info = (H, W, p, h, w)
        
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        #x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        #imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
        return imgs
     
        
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgsx, imgsy, mask_ratio=0.75):
    
        #latentx, latenty, latentz, mask_x, mask_y, mask_z = self.forward_tokenizer(imgsx, imgsy, imgsz, mask_ratio)
        ## 应该保持不同模态的mask不同，因为从所有令牌中统一选择可见令牌来调整MAE掩码采样策略，将导致大多数模态以相似的程度表示
        ## 即使来自一个模态的标记很少可见，由于跨模态的相互作用，所产生的预测也相对稳定和可信。
        ## 论文 MultiMAE: Multi-modal Multi-task Masked Autoencoders
        #latentx, latenty, latentz = self.forward_encoder_specific(latentx, latenty, latentz)
        #latentx, latenty, latentz, mask_x, mask_y, mask_z = self.forward_encoder(imgsx, imgsy, imgsz, mask_ratio)# ？ 65 720
        #pred_x, pred_y, pred_z = self.forward_decoder_share(latentx, latenty, latentz)#, ids_restore_x, ids_restore_y, ids_restore_z)  # [N, L, p*p*3]  ? 64 16*16*64
        ## 这里的latent指的是mask后剩余patch对应的特征。
        
        
        latentx, latenty, mask_x, mask_y = self.Encoder(imgsx, imgsy, mask_ratio)
        pred_x, pred_y  = self.Dncoder(latentx, latenty)
        
        loss_x = self.forward_loss(imgsx, pred_x, mask_x)
        loss_y = self.forward_loss(imgsy, pred_y, mask_y)
        
        #latent = self.forward_encoder_seg(imgs)
       
        H, W, p, h, w = self.patch_info
        
        # pred_x = pred_x.reshape(shape=(pred_x.shape[0], H, W, -1))
        # pred_x = torch.einsum('nhwc->nchw', pred_x)
        
        # pred_y = pred_y.reshape(shape=(pred_y.shape[0], H, W, -1))
        # pred_y = torch.einsum('nhwc->nchw', pred_y)
        
        # pred_z = pred_z.reshape(shape=(pred_z.shape[0], H, W, -1))
        # pred_z = torch.einsum('nhwc->nchw', pred_z)
        
        loss = loss_x + loss_y
        
        return loss#, pred_x, pred_y, pred_z

class Swin_MAE_Segmenter(torch.nn.Module):
    def __init__(self, encoder : Swin_MAE_Encoder, dim = 96, window_size=8, num_classes=10) -> None:
        super().__init__()

        self.forward_encoder_specific = encoder.forward_encoder_specific
        self.forward_encoder_share = encoder.forward_encoder_share
        
        self.patch_embed_x = encoder.patch_embed_x
        self.pos_embed_x   = encoder.pos_embed_x
        self.patch_embed_y = encoder.patch_embed_y
        self.pos_embed_y   = encoder.pos_embed_y
        
        # self.patch_embed_z = encoder.patch_embed_z
        # self.pos_embed_z   = encoder.pos_embed_z
        
        # self.decoder_pred_x = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # self.decoder_pred_y = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # self.decoder_pred_z = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        
        # self.transconv1 = nn.ConvTranspose2d(384*3, 384*2, kernel_size=2, stride=2, padding=0, output_padding=0)
        # self.transconv2 = nn.ConvTranspose2d(384*2, 384, kernel_size=2, stride=2, padding=0, output_padding=0)
        # self.transconv3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2, padding=0, output_padding=0)
        # self.transconv4 = nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        
        # self.transconv = nn.Sequential(
            # nn.ConvTranspose2d(384*3, 384*2, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(384*2, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(384*2, 384, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(384, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
        # )
        # self.final_conv = nn.Conv2d(64, num_classes, 1, 1)
        
        self.swinLSTM1 = SwinLSTM(img_size=(32, 32), patch_size=2,
                         in_chans=dim, embed_dim=dim,
                         depths=(2, 2), num_heads=(2, 2),
                         window_size=window_size, drop_rate = 0.,
                         attn_drop_rate = 0., drop_path_rate=0.1)
                         
        self.swinLSTM2 = SwinLSTM(img_size=(16, 16), patch_size=2,
                         in_chans=dim*2, embed_dim=dim*2,
                         depths=(2, 2), num_heads=(2, 2),
                         window_size=window_size, drop_rate = 0.,
                         attn_drop_rate = 0., drop_path_rate=0.1)
                         
        self.swinLSTM3 = SwinLSTM(img_size=(8, 8), patch_size=2,
                         in_chans=dim*4, embed_dim=dim*4,
                         depths=(2, 2), num_heads=(2, 2),
                         window_size=window_size, drop_rate = 0.,
                         attn_drop_rate = 0., drop_path_rate=0.1)
        
    def multimodal_fuse(self, latentx, latenty, flag=1):

        outputs = []
        states = [None] * 2

        #latentx = rearrange(latentx, 'B H W C -> B C H W')
        #latenty = rearrange(latenty, 'B H W C -> B C H W')
        #latentz = rearrange(latentz, 'B H W C -> B C H W')
        
        firstinput = (latentx, latenty)
        # last_input = latentz
        if flag == 1:
            swinLSTM = self.swinLSTM1 
        elif flag == 2:
            swinLSTM = self.swinLSTM2 
        elif flag == 3:
            swinLSTM = self.swinLSTM3 
        else:
            print('Please check the deimensions')
        # print(111111111, firstinput[0].shape)
        for i in range(2):
            output, states = swinLSTM(firstinput[i], states)
            outputs.append(output)

        # for i in range(1):
        #     output, states = swinLSTM(last_input, states)
        #     outputs.append(output)
        #     last_input = output
        
        latentx = rearrange(latentx, 'B H W C -> B C H W')
        latenty = rearrange(latenty, 'B H W C -> B C H W')
        #latentz = rearrange(latentz, 'B H W C -> B C H W')
        return latentx, latenty
        
    def forward(self, imgsx, imgsy):
        
        x = self.patch_embed_x(imgsx)
        #print("mae的positional是："+str(x.shape))
        # add pos embed w/o cls token
        x = x + self.pos_embed_x[:, 1:, :]

        y = self.patch_embed_y(imgsy)
        #print("mae的positional是："+str(x.shape))
        # add pos embed w/o cls token
        y = y + self.pos_embed_y[:, 1:, :]

        #z = self.patch_embed_z(imgsz)
        #print("mae的positional是："+str(x.shape))
        # add pos embed w/o cls token
        #z = z + self.pos_embed_z[:, 1:, :]#4 32*32 96
        
        
        x_list = []
        y_list = []
        #z_list = []## 不同分辨率的融合
        
        x = rearrange(x, 'B (H W) C -> B H W C', H=int(x.shape[1] ** 0.5))
        y = rearrange(y, 'B (H W) C -> B H W C', H=int(y.shape[1] ** 0.5))
        #z = rearrange(z, 'B (H W) C -> B H W C', H=int(z.shape[1] ** 0.5))
        
        x1, y1 = self.multimodal_fuse(x, y, flag = 1)
        
        x_list.append(x1)
        y_list.append(y1)
        
        x, y = self.forward_encoder_specific(x, y)# 4 16 16 192
        x2, y2 = self.multimodal_fuse(x, y, flag = 2)
        x_list.append(x2)
        y_list.append(y2)
        
        latentx, latenty = self.forward_encoder_share(x, y) 
        x3, y3 = self.multimodal_fuse(latentx, latenty, flag = 3)
        #fuse = torch.cat([x3, y3, z3], 1)# 8 8 384
        x_list.append(x3)
        y_list.append(y3)

        return x_list, y_list