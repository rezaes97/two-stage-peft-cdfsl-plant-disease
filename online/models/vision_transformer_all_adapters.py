# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from SSF repository:
https://github.com/dongzelian/SSF
"""

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint

from functools import partial
from utils import trunc_normal_
from models.vision_transformer import DropPath



def _adapter_list(adapters):
    """Normalize adapter-like args to a list. Accepts None, single string, or list/tuple."""
    if adapters is None:
        return []
    if isinstance(adapters, str):
        # Accept comma-separated string optionally too:
        if ',' in adapters:
            return [s.strip() for s in adapters.split(',') if s.strip()]
        return [adapters]
    return list(adapters)

def ssf_init_scale_shift(dim, ssf_init=True):
    # initialize parameters with constant initialization (Zhou et al. 2022)
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    if ssf_init:
        # initialize parameters with a mean value of one and zero (Zhou et al. 2022)
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

# --- Invertible adapter (NICE-style coupling) ---
class InvertibleAdapter(nn.Module):
    """
    NICE-style invertible adapter used by MAD-X:
    Splits last dim into two halves and applies two coupling nets f,g.
    forward:  y1 = x1 + f(x2);  y2 = x2 + g(y1)
    inverse: x2 = y2 - g(y1);  x1 = y1 - f(x2)
    Works with inputs shaped (..., C) or (B, N, C).
    """
    def __init__(self, dim, hidden=None, non_linearity=nn.GELU):
        super().__init__()
        assert dim % 2 == 0, "InvertibleAdapter requires even embedding dim"
        self.dim = dim
        self.half = dim // 2
        h = hidden or max( self.half, 64 )
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
        # convert (..., C) -> (B*N, C) robustly (works even if x is non-contiguous)
        orig_shape = x.shape
        # use reshape which handles non-contiguous tensors
        flat = x.reshape(-1, orig_shape[-1])
        return flat, orig_shape
    
    def _restore(self, flat, orig_shape):
        # restore using reshape too
        return flat.reshape(*orig_shape)

    def forward(self, x):
        # supports (B, N, C) or (B, C) or (batch*, C)
        flat, orig_shape = self._flatten_last(x)
        c = self.dim
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


class LoRAAdapter(nn.Module):
    def __init__(self, dim, r=8, alpha=1.0, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.lora_A = nn.ModuleDict({'default': nn.Linear(dim, r, bias=False)})
        self.lora_B = nn.ModuleDict({'default': nn.Linear(r, dim, bias=False)})
        self.dropout = nn.Dropout(dropout) if (dropout is not None and dropout > 0.0) else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A['default'].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B['default'].weight)

    def forward(self, x):
        delta = self.lora_B['default'](self.lora_A['default'](x))
        delta = self.dropout(delta)
        return x + (self.alpha * delta)


class HoulsbyAdapter(nn.Module):
    def __init__(self, dim, reduction=32, non_linearity='gelu', dropout=0.0):
        super().__init__()
        r = max(1, dim // reduction)
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, r, bias=True)
        self.act = nn.GELU() if non_linearity == 'gelu' else nn.ReLU()
        self.up = nn.Linear(r, dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)
    def forward(self, x):
        z = self.ln(x)
        z = self.down(z)
        z = self.act(z)
        z = self.up(z)
        z = self.dropout(z)
        return x + z


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 base_adapters=None, ssf_init=True, lora_target_modules=None, lora_r=8, lora_alpha=1.0, lora_dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.base_adapters = _adapter_list(base_adapters)
        if 'ssf' in self.base_adapters:
            self.ssf_scale_1, self.ssf_shift_1 = ssf_init_scale_shift(hidden_features, ssf_init=ssf_init)
            self.ssf_scale_2, self.ssf_shift_2 = ssf_init_scale_shift(out_features, ssf_init=ssf_init)

        self.fc1_lora = None
        self.fc2_lora = None
        if 'lora' in self.base_adapters and lora_target_modules:
            lora_target_modules = _adapter_list(lora_target_modules)
            if 'fc1' in lora_target_modules or 'mlp_fc1' in lora_target_modules:
                # LoRA applied to fc1 output (hidden dim)
                self.fc1_lora = LoRAAdapter(hidden_features, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            if 'fc2' in lora_target_modules or 'mlp_fc2' in lora_target_modules:
                # LoRA applied to fc2 output (out_features)
                self.fc2_lora = LoRAAdapter(out_features, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)


    def forward(self, x):
        x = self.fc1(x)
        if 'ssf' in self.base_adapters:
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
        if getattr(self, 'fc1_lora', None) is not None:
            x = self.fc1_lora(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if 'ssf' in self.base_adapters:
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)
        if getattr(self, 'fc2_lora', None) is not None:
            x = self.fc2_lora(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., base_adapters=None,
                 ssf_init=True, lora_target_modules=None, lora_r=8, lora_alpha=1.0, lora_dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.base_adapters = _adapter_list(base_adapters)
        if 'ssf' in self.base_adapters:
            self.ssf_scale_1, self.ssf_shift_1 = ssf_init_scale_shift(dim * 3, ssf_init=ssf_init)
            self.ssf_scale_2, self.ssf_shift_2 = ssf_init_scale_shift(dim, ssf_init=ssf_init)

        self.qkv_lora = None
        self.proj_lora = None
        if 'lora' in self.base_adapters and lora_target_modules:
            lora_target_modules = _adapter_list(lora_target_modules)
            if 'qkv' in lora_target_modules:
                self.qkv_lora = LoRAAdapter(dim * 3, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            if 'proj' in lora_target_modules:
                self.proj_lora = LoRAAdapter(dim, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv_out = self.qkv(x)
        if 'ssf' in self.base_adapters:
            qkv_out = ssf_ada(qkv_out, self.ssf_scale_1, self.ssf_shift_1)
        if getattr(self, 'qkv_lora', None) is not None:
            qkv_out = self.qkv_lora(qkv_out)
        qkv = qkv_out.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if 'ssf' in self.base_adapters:
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)
        if getattr(self, 'proj_lora', None) is not None:
            x = self.proj_lora(x)
        x = self.proj_drop(x)
        return x, attn


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0,
                 base_adapters=None, houl_adapter_reduction=32, houl_act='gelu', houl_adapter_dropout=0.0,
                 lora_r=8, lora_alpha=1.0, lora_dropout=None, lora_target_modules=None,
                 ssf_init=True):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, base_adapters=base_adapters, ssf_init=ssf_init,
            lora_target_modules=lora_target_modules, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout 
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       base_adapters=base_adapters, ssf_init=ssf_init,
                       lora_target_modules=lora_target_modules, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)


        # Adapter containers
        self.has_houlsby = 'houlsby' in (base_adapters or [])
        self.has_ssf = 'ssf' in (base_adapters or [])
        self.has_lora = 'lora' in (base_adapters or [])

        if self.has_houlsby:
            self.attn_adapter = HoulsbyAdapter(dim, reduction=houl_adapter_reduction, non_linearity=houl_act, dropout=houl_adapter_dropout)
            self.mlp_adapter = HoulsbyAdapter(dim, reduction=houl_adapter_reduction, non_linearity=houl_act, dropout=houl_adapter_dropout)
        if self.has_ssf:
            self.ssf_scale_1, self.ssf_shift_1 = ssf_init_scale_shift(dim, ssf_init=ssf_init)
            self.ssf_scale_2, self.ssf_shift_2 = ssf_init_scale_shift(dim, ssf_init=ssf_init)


    def forward(self, x, return_attention=False, return_logits=True):
        y, attn = self.attn(self.norm1(x))
        if self.has_houlsby:
            y = self.attn_adapter(y)
        x = x + self.drop_path(y)

        mlp_in = self.norm2(x)
        if self.has_ssf:
            mlp_in = ssf_ada(mlp_in, self.ssf_scale_2, self.ssf_shift_2)
        mlp_out = self.mlp(mlp_in)
        if self.has_houlsby:
            mlp_out = self.mlp_adapter(mlp_out)
        x = x + self.drop_path(mlp_out)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, add_ssf_to_patch=False, ssf_init=True):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.add_ssf_to_patch = add_ssf_to_patch

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if self.add_ssf_to_patch:
            self.ssf_scale_1, self.ssf_shift_1 = ssf_init_scale_shift(embed_dim, ssf_init=ssf_init)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.add_ssf_to_patch:
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
        return x


class VisionTransformerExtended(nn.Module):
    """ Vision Transformer """

    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), return_all_tokens=False,
                 init_values=0, use_mean_pooling=False, masked_im_modeling=False,
                 base_adapters=None, tuning_adapters=None, tuning_depth=0,
                 houl_adapter_reduction=32, houl_act='gelu', houl_adapter_dropout=0.0,
                 lora_r=8, lora_alpha=1.0, lora_dropout=0.0, lora_target_modules=None,
                 ssf_init=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens

        if 'ssf' in base_adapters or ('ssf' in tuning_adapters and tuning_depth==12):
            add_ssf_to_patch = True
        else:
            add_ssf_to_patch = False


        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            add_ssf_to_patch=add_ssf_to_patch, ssf_init=ssf_init
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        tuning_adapters_list = _adapter_list(tuning_adapters)

        # Accept several synonyms for "invertible" tuning flag (e.g., "invertible", "inverted", "invert")
        invertible_aliases = set(['invertible', 'inverted', 'invert'])
        base_adapters_list = _adapter_list(base_adapters)
        # Create invertible adapter if any alias is present in tuning_adapters_list
        if any(a.lower() in invertible_aliases for a in (tuning_adapters_list + base_adapters_list)):
            # single module used for forward and inverse
            self.invertible_adapter = InvertibleAdapter(embed_dim, hidden=embed_dim//2)
        else:
            self.invertible_adapter = None        

        # Assign adapters to each block
        self.blocks = nn.ModuleList()
        
        td = int(tuning_depth or 0)
        depth_int = int(depth)
        for i in range(depth_int):
            in_last_tuning = (td > 0) and (i >= depth_int - td)
            block_adapters = list(base_adapters_list)
            if in_last_tuning:
                for a in tuning_adapters_list:
                    if a not in block_adapters:
                        block_adapters.append(a)

            self.blocks.append(Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, base_adapters=block_adapters,
                houl_adapter_reduction=houl_adapter_reduction, houl_act=houl_act, houl_adapter_dropout=houl_adapter_dropout,
                lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules, ssf_init=ssf_init
            ))

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # Freeze all parameters by default
        for p in self.parameters():
            p.requires_grad = False

        # If invertible adapter exists and the tuning_adapters requested it (any alias), make it trainable
        if getattr(self, 'invertible_adapter', None) is not None:
            if any(a.lower() in invertible_aliases for a in tuning_adapters_list):
                for p in self.invertible_adapter.parameters():
                    p.requires_grad = True

        if td == 12 and 'ssf' in tuning_adapters_list:
            pe = getattr(self, 'patch_embed', None)
            if pe is not None:
                # unfreeze SSF scale/shift on patch_embed if present
                for t in ('ssf_scale_1', 'ssf_shift_1', 'ssf_scale_2', 'ssf_shift_2'):
                    v = getattr(pe, t, None)
                    if v is not None:
                        # these may be nn.Parameter (1D tensors) — allow direct flagging
                        try:
                            v.requires_grad = True
                        except Exception:
                            # if it's a module or container, try to set its params
                            for p in getattr(v, 'parameters', lambda: [])():
                                p.requires_grad = True

        # Unfreeze only the adapters specified in tuning_adapters for last tuning_depth blocks
        if tuning_adapters_list and td > 0:
            for blk in self.blocks[-td:]:
                # Houlsby unfreeze
                if 'houlsby' in tuning_adapters_list:
                    if hasattr(blk, 'attn_adapter'):
                        for p in blk.attn_adapter.parameters():
                            p.requires_grad = True
                    if hasattr(blk, 'mlp_adapter'):
                        for p in blk.mlp_adapter.parameters():
                            p.requires_grad = True
                # SSF unfreeze (parameters are simple tensors)
                if 'ssf' in tuning_adapters_list:
                    # ssf might exist in block (either from base_adapters or created only for these last blocks)
                    if hasattr(blk, 'ssf_scale_1'):
                        for p in [getattr(blk, 'ssf_scale_1', None), getattr(blk, 'ssf_shift_1', None),
                                  getattr(blk, 'ssf_scale_2', None), getattr(blk, 'ssf_shift_2', None)]:
                            if p is not None:
                                p.requires_grad = True
                    # also check attn and mlp internal ssf
                    if hasattr(blk, 'attn'):
                        for t in ['ssf_scale_1', 'ssf_shift_1', 'ssf_scale_2', 'ssf_shift_2']:
                            v = getattr(blk.attn, t, None)
                            if v is not None:
                                v.requires_grad = True
                    if hasattr(blk, 'mlp'):
                        for t in ['ssf_scale_1', 'ssf_shift_1', 'ssf_scale_2', 'ssf_shift_2']:
                            v = getattr(blk.mlp, t, None)
                            if v is not None:
                                v.requires_grad = True
                # LoRA unfreeze
                if 'lora' in tuning_adapters_list:
                    if hasattr(blk, 'attn') and getattr(blk.attn, 'qkv_lora', None) is not None:
                        for p in blk.attn.qkv_lora.parameters():
                            p.requires_grad = True
                    if hasattr(blk, 'attn') and getattr(blk.attn, 'proj_lora', None) is not None:
                        for p in blk.attn.proj_lora.parameters():
                            p.requires_grad = True
                    if getattr(blk, 'mlp', None) is not None:
                        if getattr(blk.mlp, 'fc1_lora', None) is not None:
                            for p in blk.mlp.fc1_lora.parameters():
                                p.requires_grad = True
                        if getattr(blk.mlp, 'fc2_lora', None) is not None:
                            for p in blk.mlp.fc2_lora.parameters():
                                p.requires_grad = True

        # masked image modeling
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, mask=None):
        B, nc, w, h = x.shape
        # patch linear embedding
        x = self.patch_embed(x)

        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)

        if getattr(self, 'invertible_adapter', None) is not None:
            x = self.invertible_adapter(x)   # maps token embeddings into adapted space

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, return_all_tokens=None, mask=None):
        # mim
        if self.masked_im_modeling:
            assert mask is not None
            x = self.prepare_tokens(x, mask=mask)
        else:
            x = self.prepare_tokens(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))

        return_all_tokens = self.return_all_tokens if \
            return_all_tokens is None else return_all_tokens
        if return_all_tokens:
            return x
        
        if getattr(self, 'invertible_adapter', None) is not None:
            # apply inverse to restore original embedding space
            x = self.invertible_adapter.inverse(x)

        return x[:, 0]

    def get_intermediate_selfattentions(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x, return_attention=True, return_logits=True)
            if len(self.blocks) - i <= n:
                output.append(attn)
        return self.norm(x[:, 0]), output

    def get_intermediate_selfattentions_all(self, x, n=1):
        x = self.prepare_tokens(x)
        output_attn = []
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x, return_attention=True, return_logits=True)
            if len(self.blocks) - i <= n:
                output_attn.append(attn)
        return self.norm(x), output_attn

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True, return_logits=False)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def get_intermediate_layers_and_attentions(self, x, n_emb=1, n_attn=1):
        x = self.prepare_tokens(x)
        output_attn = []
        output_embedding = []
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x, return_attention=True, return_logits=True)
            if len(self.blocks) - i <= n_attn:
                output_attn.append(attn)
            if len(self.blocks) - i <= n_emb:
                output_embedding.append(self.norm(x))
        return output_embedding, output_attn

    def get_num_layers(self):
        return len(self.blocks)

    def mask_model(self, x, mask):
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x


# def vit_tiny_extended(patch_size=16, **kwargs):
#     model = VisionTransformerExtended(
#         patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
#         qkv_bias=True, **kwargs)
#     return model


def vit_small_extended(patch_size=16, **kwargs):
    model = VisionTransformerExtended(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


# def vit_base_extended(patch_size=16, **kwargs):
#     model = VisionTransformerExtended(
#         patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
#         qkv_bias=True, **kwargs)
#     return model


# def vit_large_extended(patch_size=16, **kwargs):
#     model = VisionTransformerExtended(
#         patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
#         qkv_bias=True, **kwargs)
#     return model
