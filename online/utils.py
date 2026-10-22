# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

import os
import sys
import time
import math
import json
import random
import datetime
import subprocess
import numpy as np
import torch
import torch.distributed as dist
import re

from collections import defaultdict, deque
from pathlib import Path
from torch import nn
from PIL import ImageFilter, ImageOps, Image, ImageDraw

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # enforce deterministic CuDNN (may slow down and sometimes change numeric results)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def check_dir(dirname, verbose=True):
    """This function creates a directory
    in case it doesn't exist"""
    Path(dirname).mkdir(parents=True, exist_ok=True)
    print("Directory ", dirname, " was created/ already exists")
    return dirname


class Accumulator():
    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.ac = np.empty(0)

    def append(self, v):
        self.ac = np.append(self.ac[-self.max_size:], v)

    def reset(self):
        self.ac = np.empty(0)

    def mean(self, last=None):
        last = last if last else self.max_size
        return self.ac[-last:].mean()


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class PermutePatch(object):
    """
    Apply Patch permutation to the PIL image.
    """

    def __init__(self, psz):
        self.psz = psz

    def __call__(self, img):
        imgs = []
        imgwidth, imgheight = img.size
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                box = (j, i, j + self.psz, i + self.psz)
                imgs.append(img.crop(box))
        random.shuffle(imgs)
        new_img = Image.new('RGB', (imgwidth, imgheight))
        k = 0
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                new_img.paste(imgs[k], (j, i))
                k += 1
        return new_img


class HideAndSeek(object):
    """
    Apply Patch permutation to the PIL image.
    """

    def __init__(self, ratio, psz):
        self.ratio = ratio
        self.psz = psz

    def __call__(self, img):
        imgwidth, imgheight = img.size
        numw, numh = imgwidth // self.psz, imgheight // self.psz
        mask_num = int(numw * numh * self.ratio)
        mask_patch = np.random.choice(np.arange(numw * numh), mask_num, replace=False)
        mask_w, mask_h = mask_patch % numh, mask_patch // numh
        # img.save('test1.png')
        draw = ImageDraw.Draw(img)
        for mw, mh in zip(mask_w, mask_h):
            draw.rectangle((mw * self.psz,
                            mh * self.psz,
                            (mw + 1) * self.psz,
                            (mh + 1) * self.psz), fill="black")
        # img.save('test2.png')
        return img

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size, verbose=False):
    """
    Replacement loader with extended normalization/mapping and detailed printing.
    - prints all checkpoint keys + shapes
    - strips wrappers (module., backbone., base_model.model., base_model., model.)
    - drops .base_layer.
    - tries many remaps for LoRA naming variants (qkv <-> qkv_lora, lora_A<->lora_down, etc.)
    - attempts tiling (3x) for common qkv vs per-dim LoRA mismatch
    Returns msg from model.load_state_dict(...)
    """
    EPS = 1e-12

    if os.path.isfile(pretrained_weights):
        raw = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        if checkpoint_key == "auto" and isinstance(raw, dict):
            # Offline training with --model-ema stores both state_dict and state_dict_ema.
            # Prefer EMA for evaluation when present; pure state_dict checkpoints remain supported.
            for candidate_key in ("state_dict_ema", "state_dict", "model", "model_state_dict"):
                if isinstance(raw.get(candidate_key), dict):
                    if verbose:
                        print(f"Auto-selected checkpoint key: {candidate_key}")
                    raw = raw[candidate_key]
                    break
        elif checkpoint_key is not None and isinstance(raw, dict) and checkpoint_key in raw:
            raw = raw[checkpoint_key]

        # First normalize/checkpoint keys (strip common wrappers)
        proc_state_raw = {}
        for k, v in raw.items():
            nk = k
            nk = nk.replace("module.", "")
            nk = nk.replace("backbone.", "")
            # common wrappers from different saving flows
            if nk.startswith("base_model.model."):
                nk = nk.replace("base_model.model.", "", 1)
            elif nk.startswith("base_model."):
                nk = nk.replace("base_model.", "", 1)
            elif nk.startswith("model."):
                nk = nk.replace("model.", "", 1)
            # unwrap PEFT .base_layer wrapper if present
            if ".base_layer." in nk:
                nk = nk.replace(".base_layer.", ".")
            proc_state_raw[nk] = v

        # # Print all keys present in checkpoint (after initial normalization)
        # if verbose:
        #     print("\n=== CHECKPOINT KEYS (processed) ===")
        #     for k in sorted(proc_state_raw.keys()):
        #         try:
        #             s = tuple(proc_state_raw[k].shape)
        #         except Exception:
        #             s = "(non-tensor)"
        #         print(f"  {k:120s}  shape={str(s)}")
        #     print("=== end checkpoint keys ===\n")

        ckpt_keys = sorted(proc_state_raw.keys())
        model_state = model.state_dict()
        model_keys = sorted(model_state.keys())
        model_key_set = set(model_keys)

        proc_state = {}
        mapping_info = []
        diffs = []
        matched_equal_shape = []
        matched_shape_mismatch = []
        skipped_ckpt_keys = []

        # helper to generate candidate remaps for a checkpoint key
        def generate_candidates(ck):
            cands = [ck]

            # swap qkv <-> qkv_lora and proj <-> proj_lora
            if ".qkv." in ck and ck.replace(".qkv.", ".qkv_lora.") not in cands:
                cands.append(ck.replace(".qkv.", ".qkv_lora."))
            if ".qkv_lora." in ck and ck.replace(".qkv_lora.", ".qkv.") not in cands:
                cands.append(ck.replace(".qkv_lora.", ".qkv."))
            if ".proj." in ck and ck.replace(".proj.", ".proj_lora.") not in cands:
                cands.append(ck.replace(".proj.", ".proj_lora."))
            if ".proj_lora." in ck and ck.replace(".proj_lora.", ".proj.") not in cands:
                cands.append(ck.replace(".proj_lora.", ".proj."))

            # insert "_lora" before ".lora_A" / ".lora_B" if missing
            m = re.search(r'(.+?)(\.(lora_[AB]\.default.*))', ck)
            if m:
                cand = m.group(1) + "_lora" + m.group(2)
                if cand not in cands:
                    cands.append(cand)
            # also try the reverse: remove "_lora" if present
            if "_lora." in ck:
                cand = ck.replace("_lora.", ".")
                if cand not in cands:
                    cands.append(cand)

            # map names lora_A <-> lora_down and lora_B <-> lora_up (and reverse)
            if "lora_A" in ck:
                cands.append(ck.replace("lora_A", "lora_down"))
                cands.append(ck.replace("lora_A", "lora_down").replace("_lora.", "."))
            if "lora_B" in ck:
                cands.append(ck.replace("lora_B", "lora_up"))
                cands.append(ck.replace("lora_B", "lora_up").replace("_lora.", "."))
            if "lora_down" in ck:
                cands.append(ck.replace("lora_down", "lora_A"))
            if "lora_up" in ck:
                cands.append(ck.replace("lora_up", "lora_B"))

            # Alias support for checkpoints saved with invertible/inverted naming.
            if "inverted" in ck and "invertible" not in ck:
                cands.append(ck.replace("inverted", "invertible"))
                cands.append(ck.replace("inverted", "invertible_adapter"))
            elif "invert" in ck and "invertible" not in ck:
                cands.append(ck.replace("invert", "invertible"))
                cands.append(ck.replace("invert", "invertible_adapter"))
            elif "invertible" in ck and "invertible_adapter" not in ck:
                cands.append(ck.replace("invertible", "invertible_adapter"))

            # ensure uniqueness
            uniq = []
            for x in cands:
                if x not in uniq:
                    uniq.append(x)
            return uniq

        for ck in ckpt_keys:
            c_v = proc_state_raw[ck]
            effective_model_key = None
            chosen_candidate = None

            # generate candidates and pick the first that exists in model_state
            for cand in generate_candidates(ck):
                if cand in model_state:
                    effective_model_key = cand
                    chosen_candidate = cand
                    break

            if effective_model_key is not None:
                m_v = model_state[effective_model_key]
                m_shape = tuple(m_v.shape)
                c_shape = tuple(c_v.shape) if isinstance(c_v, torch.Tensor) else ()
                c_tensor = c_v
                adjusted = False

                # If shapes differ, try intelligent LoRA tiling:
                if m_shape != c_shape and isinstance(c_v, torch.Tensor):
                    # lora_A: common pattern (r, in_dim) -> model expects (r, in_dim*3)
                    if ("lora_A" in effective_model_key or "lora_A" in ck or "lora_down" in effective_model_key or "lora_down" in ck) and len(c_shape) == 2 and len(m_shape) == 2:
                        r_ck, in_ck = c_shape
                        r_m, in_m = m_shape
                        if r_ck == r_m and in_m == in_ck * 3:
                            c_tensor = torch.cat([c_v.detach().cpu(), c_v.detach().cpu(), c_v.detach().cpu()], dim=1)
                            adjusted = True
                    # lora_B: common pattern (in_dim, r) -> model expects (in_dim*3, r)
                    if not adjusted and ("lora_B" in effective_model_key or "lora_B" in ck or "lora_up" in effective_model_key or "lora_up" in ck) and len(c_shape) == 2 and len(m_shape) == 2:
                        in_ck, r_ck = c_shape
                        in_m, r_m = m_shape
                        if r_ck == r_m and in_m == in_ck * 3:
                            c_tensor = torch.cat([c_v.detach().cpu(), c_v.detach().cpu(), c_v.detach().cpu()], dim=0)
                            adjusted = True
                    # bias-like 1D case: allow 3x tiling
                    if not adjusted and len(c_shape) == 1 and len(m_shape) == 1:
                        if m_shape[0] == c_shape[0] * 3:
                            c_tensor = torch.cat([c_v.detach().cpu(), c_v.detach().cpu(), c_v.detach().cpu()], dim=0)
                            adjusted = True

                # store the tensor to proc_state (use adjusted if applied)
                proc_state[effective_model_key] = c_tensor

                # compute norms/diffs if shapes now equal
                entry = {'ckpt_key': ck, 'model_key': effective_model_key, 'm_shape': m_shape, 'c_shape': c_shape, 'adjusted': adjusted}
                if tuple(proc_state[effective_model_key].shape) == m_shape:
                    try:
                        m_cpu = m_v.detach().cpu().float()
                        c_cpu = proc_state[effective_model_key].float()
                        abs_diff = (m_cpu - c_cpu).norm().item()
                        m_norm = m_cpu.norm().item()
                        c_norm = c_cpu.norm().item()
                        rel_to_model = abs_diff / (m_norm + EPS)
                        rel_to_ckpt  = abs_diff / (c_norm + EPS)
                    except Exception:
                        abs_diff = float('inf'); m_norm = float('inf'); c_norm = float('inf')
                        rel_to_model = float('inf'); rel_to_ckpt = float('inf')
                    entry.update({'matched': True, 'abs_diff': abs_diff, 'm_norm': m_norm, 'c_norm': c_norm,
                                  'rel_to_model': rel_to_model, 'rel_to_ckpt': rel_to_ckpt})
                    diffs.append((effective_model_key, rel_to_model, rel_to_ckpt, abs_diff, m_norm, c_norm, m_shape))
                    matched_equal_shape.append(effective_model_key)
                else:
                    entry.update({'matched': True, 'shape_mismatch': True})
                    matched_shape_mismatch.append((effective_model_key, m_shape, tuple(proc_state[effective_model_key].shape)))
                mapping_info.append(entry)
            else:
                # couldn't map this checkpoint key to model key -> keep original (will be unexpected)
                proc_state[ck] = c_v
                mapping_info.append({'ckpt_key': ck, 'model_key': None, 'skipped': True, 'c_shape': tuple(c_v.shape) if isinstance(c_v, torch.Tensor) else ()})
                skipped_ckpt_keys.append(ck)

        # Now load the remapped state dict non-strictly
        msg = model.load_state_dict(proc_state, strict=False)

        # # Extract missing/unexpected lists from PyTorch msg
        # try:
        #     missing_keys = msg.missing_keys if hasattr(msg, 'missing_keys') else (msg.get('missing_keys', []) if isinstance(msg, dict) else [])
        # except Exception:
        #     missing_keys = []
        # try:
        #     unexpected_keys = msg.unexpected_keys if hasattr(msg, 'unexpected_keys') else (msg.get('unexpected_keys', []) if isinstance(msg, dict) else [])
        # except Exception:
        #     unexpected_keys = []

        # # Trainable-but-not-in-checkpoint
        # trainable_missing = []
        # for n, p in model.named_parameters():
        #     if p.requires_grad and n not in proc_state:
        #         trainable_missing.append((n, tuple(p.shape)))

        # # Print mapping lines
        # print("\n=== CHECKPOINT -> MODEL KEY MAPPINGS ===")
        # for entry in mapping_info:
        #     if entry.get('skipped', False):
        #         print(f"  {entry['ckpt_key']:100s}  -> SKIPPED  (ckpt_shape={entry['c_shape']})")
        #     else:
        #         if entry.get('shape_mismatch', False):
        #             print(f"  {entry['ckpt_key']:100s}  -> {entry['model_key']:100s}  (SHAPE MISMATCH ckpt={entry['c_shape']} model={entry['m_shape']})")
        #         else:
        #             m_norm = entry.get('m_norm', float('inf')); c_norm = entry.get('c_norm', float('inf'))
        #             abs_diff = entry.get('abs_diff', float('inf'))
        #             r_model = entry.get('rel_to_model', float('inf')); r_ckpt = entry.get('rel_to_ckpt', float('inf'))
        #             shape_str = str(entry['m_shape'])
        #             adj_str = " (ADJUSTED x3)" if entry.get('adjusted', False) else ""
        #             print(f"  {entry['ckpt_key']:100s}  -> {entry['model_key']:100s}  shape={shape_str:20s}  abs_diff={abs_diff:.6g}  rel_to_model={r_model:.6g}  rel_to_ckpt={r_ckpt:.6g}{adj_str}")

        # # Summaries
        # print("\n=== SUMMARY ===")
        # print(f"  model keys total: {len(model_keys)}")
        # print(f"  checkpoint keys total (processed): {len(ckpt_keys)}")
        # print(f"  matched (equal-shape) keys: {len(matched_equal_shape)}")
        # print(f"  matched but shape-mismatch keys: {len(matched_shape_mismatch)}")
        # print(f"  skipped checkpoint keys (not used by model): {len(skipped_ckpt_keys)}")
        # print(f"  missing keys in checkpoint (model expects but not found): {len(missing_keys)}")
        # print(f"  unexpected keys in checkpoint (loaded but unused): {len(unexpected_keys)}")
        # print(f"  trainable params not present in checkpoint: {len(trainable_missing)}")

        # # Show top differences
        # diffs.sort(key=lambda x: x[1], reverse=True)
        # topN = min(len(diffs), 40)
        # if topN > 0:
        #     print(f"\nTop {topN} parameters by rel_to_model (||m-c|| / ||m||):")
        #     for k, rel_m, rel_c, abs_d, m_n, c_n, shape in diffs[:topN]:
        #         tag = " <-- LARGE DELTA" if rel_m > 1e-3 else ""
        #         print(f"  {k:80s} rel_to_model={rel_m:.6g} rel_to_ckpt={rel_c:.6g} abs_diff={abs_d:.6g}  model_norm={m_n:.6g}  ckpt_norm={c_n:.6g}  shape={shape}{tag}")

        #     diffs_abs = sorted(diffs, key=lambda x: x[3], reverse=True)
        #     print(f"\nTop {topN} parameters by absolute difference (||m-c||):")
        #     for k, rel_m, rel_c, abs_d, m_n, c_n, shape in diffs_abs[:topN]:
        #         print(f"  {k:80s} abs_diff={abs_d:.6g}  model_norm={m_n:.6g}  ckpt_norm={c_n:.6g}  rel_to_model={rel_m:.6g}  rel_to_ckpt={rel_c:.6g}  shape={shape}")

        # if matched_shape_mismatch:
        #     print(f"\nShape mismatches (first {min(len(matched_shape_mismatch),40)}):")
        #     for k, mshape, cshape in matched_shape_mismatch[:40]:
        #         print(f"  {k:80s} model={mshape}  ckpt={cshape}")

        # if missing_keys:
        #     print(f"\nMissing keys (model expects but not found in ckpt) [{min(len(missing_keys),80)} shown]:")
        #     for k in missing_keys[:80]:
        #         print("  -", k)
        # if unexpected_keys:
        #     print(f"\nUnexpected keys in checkpoint (not used by model) [{min(len(unexpected_keys),80)} shown]:")
        #     for k in unexpected_keys[:80]:
        #         print("  -", k)

        # if trainable_missing:
        #     print(f"\nTrainable parameters not loaded from checkpoint (first {min(len(trainable_missing),80)}):")
        #     for n, s in trainable_missing[:80]:
        #         print(f"  - {n:70s}  {s}")

        # if verbose:
        #     print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

        return msg

    # (download / supervised branches unchanged from your original function)
    elif pretrained_weights == 'download':
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            if verbose:
                print("Since no pretrained weights are provided, we load the pretrained weights from {}.".format(url))
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
            return
    elif pretrained_weights == 'supervised':
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "deit_small_patch16_224-cd65a155.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "deit_base_patch16_224-b5f2ef4d.pth"
        if url is not None:
            if verbose:
                print("Since no pretrained weights are provided, we load the pretrained weights from {}.".format(url))
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/deit/" + url)
            msg = model.load_state_dict(state_dict['model'], strict=False)
            if verbose:
                print('Supervised weights found at {} and loaded with msg: {}'.format(url, msg))
            return
    print("There is no reference weights available for this model => We use random weights.")


# def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size, verbose=False):
#     if os.path.isfile(pretrained_weights):
#         state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False )
#         if checkpoint_key is not None and checkpoint_key in state_dict:
#             if verbose:
#                 print(f"Take key {checkpoint_key} in provided checkpoint dict")
#             state_dict = state_dict[checkpoint_key]
#         # Normalize keys: remove common wrapper prefixes
#         new_state = {}
#         for k, v in state_dict.items():
#             # remove common prefixes added by ddp/multicrop wrappers
#             nk = k.replace("module.", "")
#             nk = nk.replace("backbone.", "")

#             # remove peft/wrapper prefixes seen in practice
#             # try most specific first
#             if nk.startswith("base_model.model."):
#                 nk = nk.replace("base_model.model.", "", 1)
#             elif nk.startswith("base_model."):
#                 nk = nk.replace("base_model.", "", 1)
#             elif nk.startswith("model."):
#                 nk = nk.replace("model.", "", 1)

#             new_state[nk] = v

#         # Optionally print a small debug summary
#         if verbose:
#             # keys before/after examples (up to 10)
#             sample_before = list(state_dict.keys())[:10]
#             sample_after = list(new_state.keys())[:10]
#             print(f"Normalized checkpoint keys (showing up to 10):")
#             for a, b in zip(sample_before, sample_after):
#                 print(f"  {a}  ->  {b}")
#             if len(state_dict) != len(new_state):
#                 print(f"Warning: key count changed from {len(state_dict)} -> {len(new_state)}")
        
#         msg = model.load_state_dict(state_dict, strict=False)
#         if verbose:
#             print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
#         return msg
#     elif pretrained_weights == 'download':
#         url = None
#         if model_name == "vit_small" and patch_size == 16:
#             url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
#         elif model_name == "vit_small" and patch_size == 8:
#             url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
#         elif model_name == "vit_base" and patch_size == 16:
#             url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
#         elif model_name == "vit_base" and patch_size == 8:
#             url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
#         if url is not None:
#             if verbose:
#                 print("Since no pretrained weights are provided, we load the pretrained weights from {}.".format(url))
#             state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
#             model.load_state_dict(state_dict, strict=True)
#             return
#     elif pretrained_weights == 'supervised':
#         url = None
#         if model_name == "vit_small" and patch_size == 16:
#             url = "deit_small_patch16_224-cd65a155.pth"
#         elif model_name == "vit_base" and patch_size == 16:
#             url = "deit_base_patch16_224-b5f2ef4d.pth"
#         if url is not None:
#             if verbose:
#                 print("Since no pretrained weights are provided, we load the pretrained weights from {}.".format(url))
#             state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/deit/" + url)
#             msg = model.load_state_dict(state_dict['model'], strict=False)
#             if verbose:
#                 print('Supervised weights found at {} and loaded with msg: {}'.format(url, msg))
#             return
#     print("There is no reference weights available for this model => We use random weights.")


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    import argparse
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


# def fix_random_seeds(seed=42):
#     """
#     Fix random seeds.
#     """
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """

    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False,
                **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx: end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx: end_idx])
                kwargs.update(dict(mask=inp_m))

            _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class PCA():
    """
    Class to  compute and apply PCA.
    """

    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1. / d ** self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]);
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def print_trainable_parameters(model, loss_func=None):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    if loss_func is not None:
        try:
            trainable_params = 0
            all_param = 0
            for param in loss_func.parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(
                f"Loss_function trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )
        except:
            print(
                f"Loss_function trainable params: trainable%: 0"
            )
