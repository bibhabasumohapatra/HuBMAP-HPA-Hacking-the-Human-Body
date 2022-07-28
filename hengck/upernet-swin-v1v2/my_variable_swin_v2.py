import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_



#https://zhangtemplar.github.io/swin-transformer/
pretrain_dir ='/root/share/data/pretrain_model'
if 1:
	cfg = dict(
		
		#configs/_base_/models/upernet_swin.py
		basic = dict(
			swin=dict(
				pretrain_img_size=256,
				pretrained_window_size=[16,16,16,8],
				window_size=[16,16,16,8],
				embed_dim=96,
				depths=[2, 2, 6, 2],
				num_heads=[3, 6, 12, 24],
				mlp_ratio=4.,
				qkv_bias=True,
				drop_rate=0.,
				attn_drop_rate=0.,
				drop_path_rate=0.3,
				patch_norm=True,
				out_indices=(0, 1, 2, 3),
			),
		
		),
		#main/configs/swinv2/swinv2_tiny_patch4_window16_256.yaml
		swinv2_tiny_patch4_window16_256=dict(
			checkpoint = pretrain_dir+'/swinv2_tiny_patch4_window16_256.pth',
			
			swin = dict(
				pretrain_img_size=256,
				pretrained_window_size=[16,16,16,8], #add print(window.size) from original ms code to find this
				window_size=[16,16,16, 8], #you can modify this to use larger fine-tune window
				embed_dim=96,
				depths=[2, 2, 6, 2],
				num_heads=[3, 6, 12, 24],
				drop_path_rate=0.2,
				patch_norm=True,
			),
			upernet=dict(
				in_channels=[96, 192, 384, 768],
			),
		),
		
		#/main/configs/swinv2/swinv2_small_patch4_window16_256.yaml
		swinv2_small_patch4_window16_256=dict(
			checkpoint = pretrain_dir+'/swinv2_small_patch4_window16_256.pth',
			
			swin = dict(
				pretrain_img_size=256,
				pretrained_window_size=[16,16,16,8],
				window_size=[16,16,16, 8],
				embed_dim=96,
				depths=[2, 2, 18, 2],
				num_heads=[3, 6, 12, 24],
				drop_path_rate=0.3,
				patch_norm=True,
			),
			upernet=dict(
				in_channels=[96, 192, 384, 768],
			),
		),
		
		
		swinv2_base_patch4_window8_256=dict(
			checkpoint = pretrain_dir+'/swinv2_base_patch4_window8_256.pth',
			
			swin = dict(
				pretrain_img_size=256,
				pretrained_window_size=[8,8,8,8],
				window_size=[8,8,8,8],
				embed_dim=128,
				depths=[2, 2, 18, 2],
				num_heads=[4, 8, 16, 32 ],
				drop_path_rate=0.5,
				patch_norm=True,
			),
			upernet=dict(
				in_channels=[128, 256, 512, 1024],
			),
		),
	)

##############################################################################################################






#https://zhangtemplar.github.io/swin-transformer/
#<no-change>
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
#---

#<no-change>
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

#<no-change>
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

#<no-change>
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self,
                 dim,
                 window_size,
                 pretrained_window_size,
                 num_heads,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        super().__init__()
        #window_size = to_2tuple(window_size)
        #pretrained_window_size=to_2tuple(pretrained_window_size) if pretrained_window_size else None,
        
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        
        
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )
        
        # get relative_coords_table (rescale)
        relative_coords_h = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        
        relative_coords_table[:, :, :, 0] /= (self.pretrained_window_size - 1)
        relative_coords_table[:, :, :, 1] /= (self.pretrained_window_size - 1)
        #relative_coords_table[:, :, :, 0] /= (self.window_size - 1)
        #relative_coords_table[:, :, :, 1] /= (self.window_size - 1)
        
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer('relative_coords_table', relative_coords_table)
        
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    
    
    
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        
        B_, N, C = x.shape
        
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        #qkv = self.qkv(x)
        
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        
        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale.float(), max=np.log(1. / 0.01)).exp()
        attn = attn * logit_scale
        
        #q = q * self.scale
        #attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = \
            relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, self.num_heads)
        # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

# remove fixed calculate attention mask for SW-MSA
#
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """
    def __init__(self,
        dim,
        num_heads,
        window_size=7,
        pretrained_window_size=7,
        shift_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=window_size,
            pretrained_window_size=pretrained_window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
     
    def forward(self, x, H, W, mask_matrix):
        
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = x.view(B, H, W, C)
        
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        
        # cyclic shift ---
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
     
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        
        # reverse cyclic shift ---
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path( self.norm1(x) )
        
        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    
#---------
# H,W as input
# padding
# note : different swin v1: self.norm = norm_layer(2 * dim)
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)
    
    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
     
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
 
        
        x = x.view(B, H, W, C)
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        x = x.view(B, -1, 4 * C)  # B, H/2*W/2, 4*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}"




# compute  attention mask for SW-MSA
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """
    def __init__(self,
        dim,
        depth,
        num_heads,
        window_size,
        pretrained_window_size,
        mlp_ratio=4.,
        qkv_bias=True,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                pretrained_window_size=pretrained_window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
            
        self._init_respostnorm()
    
    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)
            
    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        
        # calculate attention mask for SW-MSA ----
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        #------
     
    
        for blk in self.blocks:
            x = blk(x, H, W, attn_mask)
            
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W
    
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


#padding
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None
                 ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # padding
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            
            
        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        
        return x
 

class SwinTransformerV2(nn.Module):
    no_weight_decay_keywords = {'cpb_mlp', 'logit_scale', 'relative_position_bias_table'}

    def __init__(self,
        pretrain_img_size=256,
        pretrained_window_size=[16,16,16,8],
        window_size=[16,16,16,8],
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        out_norm = nn.Identity, #use nn.Identity, nn.BatchNorm2d, LayerNorm2d
        **kwargs
    ):
        super().__init__()
        
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        
    
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # stochastic depth
        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist() # stochastic depth decay rule
    
        # build layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                pretrained_window_size=pretrained_window_size[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i < self.num_layers - 1) else None,
            )
            self.layers.append(layer)
            
        #---
        # add a norm layer for each output
        self.out_norm = nn.ModuleList(
            [ out_norm(int(embed_dim * 2 ** i)) for i in range(self.num_layers)]
        )

        #---
        self.apply(self._init_weights)
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
 
    
    def forward(self, x):
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        
        #remove positional encode?
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
	        x_out, H, W, x, Wh, Ww = self.layers[i](x, Wh, Ww)
	        out = x_out.view(-1, H, W, int(self.embed_dim * 2 ** i)).permute(0, 3, 1, 2).contiguous()
	        out = self.out_norm[i](out)
	        outs.append(out)
         
        return outs
    
    
##https://github.com/microsoft/Swin-Transformer/blob/main/optimizer.py
def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def set_weight_decay(model, skip=(), skip_keywords=()):
    has_decay = []
    no_decay  = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or (name in skip) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]

def get_swin_parameters(swin):
    no_weight_decay = {'absolute_pos_embed'}
    no_weight_decay_keywords = {'cpb_mlp', 'logit_scale', 'relative_position_bias_table'}
 
    parameters = set_weight_decay(swin, skip=no_weight_decay, skip_keywords=no_weight_decay_keywords)
 
    '''
    #example use
    optimizer = optim.AdamW(parameters, eps=eps, betas=betas, lr=lr, weight_decay=weight_decay)
    
    '''
    return parameters
    
    
#---

#debug
if 0:
    #https://github.com/microsoft/Swin-Transformer/blob/main/configs/swinv2/swinv2_tiny_patch4_window16_256.yaml
    m = SwinTransformerV2(
        pretrain_img_size=256,
        pretrained_window_size=[16,16,16,8],
        window_size=[16,16,16,8],
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
    )
    print(m)
    checkpoint = torch.load('/root/share/data/pretrain_model/swinv2_tiny_patch4_window16_256.pth', map_location=lambda storage, loc: storage)['model']
    print(m.load_state_dict(checkpoint,strict=False))  #True
    #ignore buffer: relative_coords_table, relative_position_index

    parameters = get_swin_parameters(m)
    #print(parameters)
    zz=0


if 0:
    swin = SwinTransformerV2()
    print(swin)
    xx = torch.rand(1,3,224,224)
    zz = swin(xx)
    print([z.shape for z in zz])
if 0:
    blk = SwinTransformerBlock(
        dim=32,
        input_resolution=(224,112),#H, W
        num_heads=5,
        window_size=7,
        shift_size=3,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    )

if 0:
    wmsa = WindowAttention( dim=32*5, window_size=(7,7), num_heads=5)
    xx =torch.rand(1,49,32*5)
    wmsa(xx)
 

