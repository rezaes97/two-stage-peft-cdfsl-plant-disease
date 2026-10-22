""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Overwriting .* in registry .*",  # regex to match timm's overwrite message
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.helpers is deprecated",  # message is treated as a regex
    category=FutureWarning,
)

import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import build_model_with_cfg, named_apply, adapt_input_conv, resolve_pretrained_cfg, checkpoint_seq
from timm.layers import DropPath, lecun_normal_, _assert
from timm.layers import to_2tuple
from timm.models import register_model
from timm.layers import trunc_normal_


import ipdb


_logger = logging.getLogger(__name__)

# --- Invertible adapter (NICE-style coupling) ---
class InvertibleAdapter(nn.Module):
    """
    NICE-style invertible adapter (coupling nets f,g).
    Works with inputs shaped (B, N, C) or (B, C) or (..., C).
    forward:  y1 = x1 + f(x2);  y2 = x2 + g(y1)
    inverse: x2 = y2 - g(y1);  x1 = y1 - f(x2)
    """
    def __init__(self, dim, hidden=None, non_linearity=nn.GELU):
        super().__init__()
        assert dim % 2 == 0, "InvertibleAdapter requires even embedding dim"
        self.dim = dim
        self.half = dim // 2
        h = hidden or max(self.half, 64)
        # small coupling networks
        self.f = nn.Sequential(
            nn.Linear(self.half, h),
            non_linearity(),
            nn.Linear(h, self.half)
        )
        self.g = nn.Sequential(
            nn.Linear(self.half, h),
            non_linearity(),
            nn.Linear(h, self.half)
        )
        # small init
        for m in (self.f, self.g):
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _flatten_last(self, x):
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])
        return flat, orig_shape

    def _restore(self, flat, orig_shape):
        return flat.reshape(*orig_shape)

    def forward(self, x):
        flat, orig_shape = self._flatten_last(x)
        x1 = flat[:, :self.half]
        x2 = flat[:, self.half:]
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        y = torch.cat([y1, y2], dim=-1)
        return self._restore(y, orig_shape)

    def inverse(self, y):
        flat, orig_shape = self._flatten_last(y)
        y1 = flat[:, :self.half]
        y2 = flat[:, self.half:]
        x2 = y2 - self.g(y1)
        x1 = y1 - self.f(x2)
        x = torch.cat([x1, x2], dim=-1)
        return self._restore(x, orig_shape)


class LoRAAdapterForActivation(nn.Module):
    """LoRA adapter operating on activation vector (in_dim) using down/up linear pair.
    Registers as .lora_A and .lora_B (ModuleDict['default']) so saved keys match your loader.
    """
    def __init__(self, in_dim, r=8, alpha=1.0, dropout=0.0):
        super().__init__()
        self.r = int(r)
        self.alpha = float(alpha)
        # keep naming compatible with many loaders: lora_A / lora_B, each a ModuleDict keyed 'default'
        self.lora_A = nn.ModuleDict({'default': nn.Linear(in_dim, self.r, bias=False)})
        self.lora_B = nn.ModuleDict({'default': nn.Linear(self.r, in_dim, bias=False)})
        self.dropout = nn.Dropout(dropout) if (dropout and dropout > 0.0) else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A['default'].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B['default'].weight)

    def forward(self, x):
        down = self.lora_A['default'](x)
        up = self.lora_B['default'](down)
        delta = self.dropout(up)
        return x + (self.alpha * delta)

class HoulsbyAdapter(nn.Module):
    """
    Houlsby adapter: LayerNorm -> down (d->r) -> act -> up (r->d) -> dropout -> residual add
    Inserted after sublayer outputs (attention output / mlp output).
    """
    def __init__(self, dim, reduction=16, non_linearity='gelu', dropout=0.0):
        super().__init__()
        # bottleneck size r
        r = max(1, dim // reduction)
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, r, bias=True)
        self.act = nn.GELU() if non_linearity == 'gelu' else (nn.ReLU() if non_linearity == 'relu' else nn.GELU())
        self.up = nn.Linear(r, dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # init small
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # adapter applied to sublayer output; apply LayerNorm inside adapter (Houlsby)
        z = self.ln(x)
        z = self.down(z)
        z = self.act(z)
        z = self.up(z)
        z = self.dropout(z)
        return x + z

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., tuning_mode=None, lora_targets=None, lora_r=4, lora_alpha=1.0, lora_dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
    
            
        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':        
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(hidden_features)
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(out_features)

        self.fc1_lora = None
        self.fc2_lora = None
        if lora_targets:
            lt = set([t.strip().lower() for t in (lora_targets if isinstance(lora_targets, (list,tuple)) else str(lora_targets).split(',')) if t.strip()])
            if 'fc1' in lt or 'mlp_fc1' in lt:
                self.fc1_lora = LoRAAdapterForActivation(hidden_features, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            if 'fc2' in lt or 'mlp_fc2' in lt:
                self.fc2_lora = LoRAAdapterForActivation(out_features, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)



    def forward(self, x):
        x = self.fc1(x)
        if self.tuning_mode == 'ssf':
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)

        if getattr(self, 'fc1_lora', None) is not None:
            x = self.fc1_lora(x)

        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x) 
        if self.tuning_mode == 'ssf':
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)

        if getattr(self, 'fc2_lora', None) is not None:
            x = self.fc2_lora(x)
        
        x = self.drop2(x)
        
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., tuning_mode=None, lora_targets=None, lora_r=4, lora_alpha=1.0, lora_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv_lora = None
        if lora_targets:
            lt = set([t.strip().lower() for t in (lora_targets if isinstance(lora_targets, (list,tuple)) else str(lora_targets).split(',')) if t.strip()])
            if 'qkv' in lt:
                # qkv activation dim = dim * 3
                self.qkv_lora = LoRAAdapterForActivation(dim * 3, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.proj_lora = None
        if lora_targets:
            lt = set([t.strip().lower() for t in (lora_targets if isinstance(lora_targets, (list,tuple)) else str(lora_targets).split(',')) if t.strip()])
            if 'proj' in lt:
                # proj activation dim = dim
                self.proj_lora = LoRAAdapterForActivation(dim, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':     
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim * 3)
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)




    def forward(self, x):
        B, N, C = x.shape
        if self.tuning_mode == 'ssf':
            qkv_out = ssf_ada(self.qkv(x), self.ssf_scale_1, self.ssf_shift_1)
        else:
            qkv_out = self.qkv(x)

        if getattr(self, 'qkv_lora', None) is not None:
            qkv_out = self.qkv_lora(qkv_out)
        
        qkv = qkv_out.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        if self.tuning_mode == 'ssf':
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)

        if getattr(self, 'proj_lora', None) is not None:
            x = self.proj_lora(x)

        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, tuning_mode=None,
            adapter_reduction=16, adapter_act='gelu', adapter_dropout=0.0,
            lora_targets=None, lora_r=4, lora_alpha=1.0, lora_dropout=0.0):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, tuning_mode=tuning_mode,
                              lora_targets=lora_targets, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if tuning_mode == 'houlsby':
            # adapter placed after attention output and after mlp output
            self.attn_adapter = HoulsbyAdapter(dim, reduction=adapter_reduction, non_linearity=adapter_act, dropout=adapter_dropout)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop, tuning_mode=tuning_mode,
                       lora_targets=lora_targets, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':     
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)
        if tuning_mode == 'houlsby':
            self.mlp_adapter = HoulsbyAdapter(dim, reduction=adapter_reduction, non_linearity=adapter_act, dropout=adapter_dropout)


    def forward(self, x):
        if self.tuning_mode == 'ssf':
            x = x + self.drop_path1(self.ls1(self.attn(ssf_ada(self.norm1(x), self.ssf_scale_1, self.ssf_shift_1))))
            x = x + self.drop_path2(self.ls2(self.mlp(ssf_ada(self.norm2(x), self.ssf_scale_2, self.ssf_shift_2))))
        elif self.tuning_mode == 'houlsby':
            # apply attention sublayer, then adapter, then residual
            attn_out = self.attn(self.norm1(x))
            attn_out = self.attn_adapter(attn_out)
            x = x + self.drop_path1(self.ls1(attn_out))

            mlp_out = self.mlp(self.norm2(x))
            mlp_out = self.mlp_adapter(mlp_out)
            x = x + self.drop_path2(self.ls2(mlp_out))
        else:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResPostBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x):
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelBlock(nn.Module):

    def __init__(
            self, dim, num_heads, num_parallel=2, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward_jit(self, x):
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, tuning_mode=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.norm_layer = norm_layer

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':     
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dim)

            if norm_layer:
                self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(embed_dim)



    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        x = self.proj(x) 
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.tuning_mode == 'ssf':  
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
            if self.norm_layer:
                x = ssf_ada(self.norm(x), self.ssf_scale_2, self.ssf_shift_2)
            else:
                x = self.norm(x)
        else:
            x = self.norm(x)
        return x



def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='',
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, tuning_mode=None,
            adapter_reduction=16, adapter_act='gelu', adapter_dropout=0.0, 
            lora_targets=None, lora_r=4, lora_alpha=1.0, lora_dropout=0.0): 
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 if class_token else 0
        self.grad_checkpointing = False 

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, tuning_mode=tuning_mode)
        num_patches = self.patch_embed.num_patches



        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.num_tokens > 0 else None
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + self.num_tokens, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # --- invertible adapter creation: enable by setting tuning_mode='invertible' manually ---
        invertible_aliases = set(['invertible', 'inverted', 'invert'])
        if isinstance(tuning_mode, str) and tuning_mode.lower() in invertible_aliases:
            # create single global invertible adapter used before blocks and inverted after
            # hidden can be changed manually here if you want a different bottleneck
            self.invertible_adapter = InvertibleAdapter(embed_dim, hidden=embed_dim//2)
        else:
            self.invertible_adapter = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.tuning_mode = tuning_mode
        tuning_mode_list = [tuning_mode] * depth 

        use_lora = (tuning_mode == 'lora')
        if lora_targets and not use_lora:
            _logger.info("lora_targets provided (%s) but tuning_mode != 'lora' (tuning_mode=%s). LoRA adapters will NOT be created.", lora_targets, tuning_mode)


        if tuning_mode == 'ssf':     
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(self.num_features)

        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, tuning_mode=tuning_mode_list[i],
                adapter_reduction=adapter_reduction, adapter_act=adapter_act, adapter_dropout=adapter_dropout, 
                lora_targets=(lora_targets if use_lora else None), lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        x = self.patch_embed(x)
        
        if getattr(self, 'invertible_adapter', None) is not None:
            x = self.invertible_adapter(x)
        
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        if getattr(self, 'invertible_adapter', None) is not None:
            x = self.invertible_adapter.inverse(x)

        x = self.norm(x)
        if self.tuning_mode == 'ssf': 
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
            
        return x 

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        
        return x 


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()



def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

