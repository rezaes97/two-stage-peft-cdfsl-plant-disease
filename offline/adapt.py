#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
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

#region Imports
import argparse
import time
import yaml
import os
import logging
# import peft
# import wandb
# import numpy as np
import re
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.utils
from timm.data import resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
     model_parameters
from timm.layers import convert_splitbn_model
from timm.utils import *
from timm.loss import *
from timm.scheduler import create_scheduler
from data import create_loader, create_dataset
from optim_factory import create_optimizer_v2, optimizer_kwargs
from proj_models import vision_transformer
import gc


has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

#endregion

#region Parsers
# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--grad-checkpointing', action='store_true', default=False,
                    help='Enable gradient checkpointing through model blocks/stages')


# finetuning
parser.add_argument('--tuning-mode', default=None, type=str,
                    help='Method of fine-tuning (default: None)')
parser.add_argument('--adapter-reduction', default=16, type=int,
                    help='Reduction factor for adapter (default: 16)')
parser.add_argument('--adapter-act', default='gelu', type=str,
                    help='Activation function for adapter (default: gelu)')
parser.add_argument('--adapter-dropout', default=0.0, type=float,
                    help='Dropout rate for adapter (default: 0.0)')
parser.add_argument('--lora-r', type=int, default=4, metavar='R',)
parser.add_argument('--lora-alpha', type=int, default=16, metavar='A',
                    help='Lora alpha (default: 16)')
parser.add_argument('--lora-dropout', type=float, default=0.05, metavar='PCT',
                    help='Lora dropout (default: 0.05)')
parser.add_argument('--target-modules', type=str, nargs='+', default=None, metavar='MODULE',
                    help='Target modules for LoRA, e.g. "qkv proj fc1" (default: ["qkv"])')

parser.add_argument('--evaluate', action='store_true', default=False,
                    help='evaluate')


# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
parser.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=5, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--simple-aug', action='store_true', default=False,
                    help='Only randomresize and flip training augmentation, override other train aug args')
parser.add_argument('--direct-resize', action='store_true', default=False,
                    help='Direct resize image in validation')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.5,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
parser.add_argument('--aug-repeats', type=float, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
parser.add_argument('--patience', type=int, default=5,
                    help='epochs with no val improvement to wait before early stopping')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        # _logger.info('found conf files')
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

#endregion

def maybe_release_cache(threshold_mb=4000):  # threshold check
    # If reserved > threshold, free cache
    if not torch.cuda.is_available():
        return
    res = torch.cuda.memory_reserved()
    if res / 1024**2 > threshold_mb:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{time.strftime('%H:%M:%S')}] empty_cache() called: reserved before > {threshold_mb} MB")

def freeze_except_keywords(model,
                           keep_keywords=('lora', 'lora_A', 'lora_B', 'lora_dropout',
                                          'ssf', 'ssf_scale', 'ssf_shift',
                                          'head', 'classifier',
                   # Houlsby / adapter keywords (kept trainable)
                   'houlsby', 'adapter', 'attn_adapter', 'mlp_adapter'),
    extra_keep=None
):
    if extra_keep:
        # allow passing additional keywords at call-time
        keep_keywords = tuple(list(keep_keywords) + list(extra_keep))
    kept = []
    frozen = []
    for n, p in model.named_parameters():
        do_keep = any(kw in n for kw in keep_keywords)
        p.requires_grad = do_keep
        if do_keep:
            kept.append((n, p.numel()))
        else:
            frozen.append((n, p.numel()))
    # summary
    kept_count = sum(sz for _, sz in kept)
    frozen_count = sum(sz for _, sz in frozen)
    print(f"Kept (trainable) params: {kept_count:,}  |  Frozen params: {frozen_count:,}")
    # optionally list small sample:
    print("Sample trainable params:")
    for n, sz in kept[:30]:
        print("  ", n, "(", f"{sz:,}", ")")
    return kept_count, frozen_count

class WeightedSoftTargetCrossEntropy(nn.Module):
    """
    Weighted cross-entropy for soft targets.
    - accepts logits (B, C) and target either:
        * LongTensor (B,) of hard labels
        * FloatTensor (B, C) of soft targets (e.g. mixup)
    - class_weights: 1D tensor of size C (CPU or GPU). Registered as buffer so .cuda() works.
    """
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
            self.register_buffer('class_weights', torch.ones(1))
        else:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            # ensure 1D
            if cw.dim() != 1:
                cw = cw.view(-1)
            self.register_buffer('class_weights', cw)

    def forward(self, input_logits, target):
        """
        input_logits: (B, C)
        target: (B,) long OR (B, C) float (soft)
        """
        log_probs = torch.nn.functional.log_softmax(input_logits, dim=1)  # (B, C)

        # convert hard labels to one-hot if needed
        if target.dtype == torch.long or target.dim() == 1:
            # create one-hot on same device/dtype as log_probs
            y = torch.zeros_like(log_probs).scatter_(1, target.view(-1, 1), 1.0)
        else:
            y = target.type_as(log_probs)

        # ensure class_weights length matches num classes
        cw = self.class_weights
        if cw.numel() != log_probs.size(1):
            # fallback: try to broadcast/truncate or raise informative error
            raise ValueError(f"class_weights length ({cw.numel()}) != num classes ({log_probs.size(1)})")

        # weighted per-class target
        weighted_target = y * cw.unsqueeze(0)  # (B, C)
        # loss per sample
        loss_per_sample = -(weighted_target * log_probs).sum(dim=1)
        return loss_per_sample.mean()

def _summarize_loss_module(loss_mod):
    name = loss_mod.__class__.__name__
    print(f"== Selected train loss: {name}")
    if isinstance(loss_mod, WeightedSoftTargetCrossEntropy):
        cw = loss_mod.class_weights
        print(f"  WeightedSoftTargetCrossEntropy.class_weights -> device={cw.device}, dtype={cw.dtype}, len={cw.numel()}")
        print(f"   stats: min={cw.min().item():.6f}, mean={cw.mean().item():.6f}, max={cw.max().item():.6f}")
    elif isinstance(loss_mod, nn.CrossEntropyLoss):
        w = loss_mod.weight
        if w is None:
            print("  CrossEntropyLoss: weight=None (no class weighting)")
        else:
            print(f"  CrossEntropyLoss.weight -> device={w.device}, dtype={w.dtype}, len={w.numel()}")
            print(f"   stats: min={w.min().item():.6f}, mean={w.mean().item():.6f}, max={w.max().item():.6f}")
    else:
        # fallback for SoftTargetCrossEntropy from timm etc.
        print(f"  Loss module attrs: {list(loss_mod.__dict__.keys())}")

def load_backbone_weights(model, ckpt_path, verbose=True):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)

    if verbose:
        print("==> checkpoint keys:")
        for k in list(sd.keys()):
            print("  ", k)
        print("\n==> model keys:")
        for k in list(model.state_dict().keys()):
            print("  ", k)

    model_sd = model.state_dict()
    new_sd = {}
    used_ck = []

    # Which substrings should try the .base_layer wrapper variant
    wrap_candidates = ('attn.qkv', 'attn.proj', 'patch_embed.proj')


    def should_map_base(base):  # base is checkpoint key without leading 'backbone.'
        # Always allow cls_token, pos_embed, patch_embed
        if base.startswith('cls_token') or base.startswith('pos_embed') or base.startswith('patch_embed'):
            return True
        # Blocks: only map block indices <= 11
        m = re.match(r'blocks\.(\d+)\b', base)
        if m:
            idx = int(m.group(1))
            return idx <= 11
        # explicitly block common head keys
        if base.startswith('head') or base.startswith('classifier') or base.startswith('last_layer'):
            return False
        # otherwise allow
        return True

    def candidate_targets_from_ckey(ckey):
        # produce a list of possible model keys to try (ordered by preference)
        if ckey.startswith('backbone.'):
            base = ckey[len('backbone.'):]
        else:
            base = ckey

        if not should_map_base(base):
            return []  # skip mapping this key

        base_tgt = 'base_model.model.' + base
        cands = []

        # if the target ends with .weight or .bias and one of wrap_candidates appears, add .base_layer variant first
        if base_tgt.endswith('.weight') or base_tgt.endswith('.bias'):
            for p in wrap_candidates:
                if p in base_tgt:
                    if base_tgt.endswith('.weight'):
                        alt = base_tgt[:-len('.weight')] + '.base_layer.weight'
                    else:
                        alt = base_tgt[:-len('.bias')] + '.base_layer.bias'
                    cands.append(alt)
                    break
        # primary candidate (plain)
        cands.append(base_tgt)

        # also add a suffix-only fallback candidate (we try matches that end with the same suffix)
        suffix = base_tgt.split('base_model.model.')[-1]
        # we'll handle suffix-fallback outside this function
        return cands

    # Build mapping: try direct candidate keys first, then suffix fallback (same-shape)
    for ck, val in sd.items():
        # skip adapter-specific or obvious head keys in checkpoint (we don't want to load them)
        if any(x in ck for x in ('lora', 'ssf', 'adapter', 'head.mlp', 'head.last_layer', 'head.')):
            continue

        cands = candidate_targets_from_ckey(ck)
        if not cands:
            continue

        mapped = False
        # 1) try explicit candidates in order
        for tgt in cands:
            if tgt in model_sd and model_sd[tgt].shape == val.shape:
                new_sd[tgt] = val
                used_ck.append((ck, tgt))
                mapped = True
                break
        if mapped:
            continue

        # 2) suffix fallback: find any model key ending with same suffix and same shape
        suffix = cands[-1].split('base_model.model.')[-1]
        matches = [mk for mk in model_sd.keys() if mk.endswith(suffix) and model_sd[mk].shape == val.shape]
        if matches:
            sel = matches[0]
            new_sd[sel] = val
            used_ck.append((ck, sel))
            continue

        # 3) last-chance: for keys like patch_embed.proj.weight vs patch_embed.proj.base_layer.weight
        # attempt an explicit base_layer/plain toggle even if wrap wasn't detected above
        if cands:
            alt = None
            tgt_plain = cands[-1]
            if tgt_plain.endswith('.weight'):
                alt = tgt_plain[:-len('.weight')] + '.base_layer.weight'
            elif tgt_plain.endswith('.bias'):
                alt = tgt_plain[:-len('.bias')] + '.base_layer.bias'
            if alt and alt in model_sd and model_sd[alt].shape == val.shape:
                new_sd[alt] = val
                used_ck.append((ck, alt))
                continue
        # else: skipped

    # Snapshot pre-load for avg-delta computation but only for keys we will set
    before = {k: v.clone().detach() for k, v in model.state_dict().items()}

    # perform the in-place load
    res = model.load_state_dict(new_sd, strict=False)

    # Snapshot after and compute avg abs diff for keys we actually loaded
    after = model.state_dict()
    diffs = []
    for _, tgt in used_ck:
        if tgt in before and before[tgt].shape == after[tgt].shape:
            diffs.append((after[tgt] - before[tgt]).abs().mean().item())
    avg_change = sum(diffs) / len(diffs) if diffs else 0.0

    # Print diagnostics
    if verbose:
        print(f"Checkpoint keys total: {len(sd)}")
        print(f"Mapped checkpoint->model pairs: {len(used_ck)}")
        if used_ck:
            print("Examples of mappings (ckpt_key -> model_key):")
            for a, b in used_ck:
                print("  ", a, "->", b)
        print(f"Backbone loaded.  Avg |deltaweight| (on mapped keys) = {avg_change:.6f}")
        print(f"Missing keys (model keys not found in ckpt mapping): {len(res.missing_keys)}\n  Sample: {res.missing_keys}")
        print(f"Unexpected keys (in ckpt but not used/dropped): {len(res.unexpected_keys)}\n  Sample: {res.unexpected_keys}")

def _unwrap_model(m):
    # common wrappers
    if hasattr(m, 'module'):       # DDP / DataParallel
        return m.module
    if hasattr(m, 'base_model'):    # some wrappers use base_model
        return m.base_model
    return m

def main():
    setup_default_logging()
    args, args_text = _parse_args()
    args.distributed=False
    args.prefetcher = not args.no_prefetcher
    args.world_size = 1
    args.device = 'cuda:0'
    args.rank = 0

    if args.amp:
        if has_native_amp:
            args.native_amp = True

    random_seed(args.seed, args.rank)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        tuning_mode=args.tuning_mode,
        adapter_reduction=args.adapter_reduction,
        adapter_act=args.adapter_act,
        adapter_dropout=args.adapter_dropout,
        lora_targets=args.target_modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    load_backbone_weights(model, args.initial_checkpoint)

    # if args.tuning_mode == 'lora':
    #     added = vision_transformer.attach_lora_adapters(model, targets=args.target_modules, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, verbose=False)
    #     print(f"Attached adapters with {added:,} params")

    freeze_except_keywords(model)

    trainable = [(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    print("Trainable adapters:", trainable)
    
    #region Restore/Cuda/Optimizer/Scheduler
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')
        _logger.info(f"number of params for requires grad: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    from contextlib import nullcontext
    amp_autocast = nullcontext
    loss_scaler = None

    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        try:
            from timm.utils import NativeScaler
            loss_scaler = NativeScaler()
        except Exception:
            _logger.info('Cant use nativescaler')
            loss_scaler = None
        if args.local_rank == 0:
            _logger.info('Using Native AMP.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        ema_device = 'cpu' if args.model_ema_force_cpu else args.device
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device=ema_device)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
        ema_internal = getattr(model_ema, 'module', getattr(model_ema, 'ema', model_ema))
        try:
            ema_internal.to(ema_device)
        except Exception:
            pass








    # DEBUG DIAGNOSTIC: compare model vs EMA keys / counts
    def _sd_keys(m):
        try:
            return set(m.state_dict().keys())
        except Exception:
            return set()

    real_model = _unwrap_model(model)
    ema_internal = getattr(model_ema, 'module', getattr(model_ema, 'ema', model_ema)) if model_ema is not None else None

    print(">>> DEBUG: model device:", next(real_model.parameters()).device)
    if ema_internal is not None:
        try:
            print(">>> DEBUG: ema device:", next(ema_internal.parameters()).device)
        except Exception:
            print(">>> DEBUG: ema device: (couldn't inspect)")

    model_count = sum(p.numel() for p in real_model.parameters())
    ema_count = sum(p.numel() for p in ema_internal.parameters()) if ema_internal is not None else 0
    print(f">>> DEBUG: model parameter count: {model_count}, ema parameter count: {ema_count}")

    mkeys = _sd_keys(real_model)
    ekeys = _sd_keys(ema_internal) if ema_internal is not None else set()
    missing = sorted(list(mkeys - ekeys))
    if missing:
        print(">>> WARNING: keys present in model but missing in EMA (first 40):")
        for k in missing[:40]:
            print("  -", k)
    else:
        print(">>> DEBUG: No missing keys in EMA.")







    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    #endregion

    #region Dataloader/Loss/Saver
    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)

    #new 
    from pathlib import Path
    # import torch.nn.functional as F
    import torch.nn as nn

    # compute per-class counts from directory layout 
    train_root = Path(args.data_dir) / args.train_split
    counts_list = []
    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir():
            continue
        counts_list.append(len(list(class_dir.glob('*'))))
    if len(counts_list) == 0:
        raise RuntimeError(f"No classes found under {train_root!s}. Check args.data_dir and args.train_split.")
    # counts on CPU (easier to pass into loss modules)
    counts = torch.tensor(counts_list, dtype=torch.float32)
    num_classes_from_counts = counts.numel()
    num_classes_expected = args.num_classes if args.num_classes is not None else int(num_classes_from_counts)

    # safe inverse-frequency weights (avoid div-by-zero)
    counts_safe = counts.clone()
    counts_safe[counts_safe == 0] = 1.0
    class_weights = 1.0 / counts_safe
    class_weights[counts == 0] = 0.0

    # normalize mean over present classes so scale remains similar to unweighted CE
    present_mask = counts > 0
    if present_mask.any():
        class_weights[present_mask] = class_weights[present_mask] / class_weights[present_mask].mean()

    # Ensure cw length matches expected num_classes: pad/truncate with warnings if mismatch
    if class_weights.numel() != num_classes_expected:
        print(f"WARNING: computed class_weights length {class_weights.numel()} != expected num_classes {num_classes_expected}")
        if class_weights.numel() < num_classes_expected:
            # pad with ones (no extra weighting) for missing classes
            pad = torch.ones(num_classes_expected - class_weights.numel(), dtype=class_weights.dtype)
            class_weights = torch.cat([class_weights, pad], dim=0)
            print(f" -> Padded class_weights to length {class_weights.numel()} (missing classes set weight=1.0).")
        else:
            # truncate
            class_weights = class_weights[:num_classes_expected]
            print(f" -> Truncated class_weights to length {class_weights.numel()}.")

    # Print diagnostics
    if args.local_rank == 0:
        print("== class counts summary ==")
        print(f" num_classes_from_counts: {num_classes_from_counts}, num_classes_expected: {num_classes_expected}")
        print(f" counts: (len={counts.numel()}) min={counts.min().item():.0f} mean={counts.mean().item():.1f} max={counts.max().item():.0f}")
        cw = class_weights
        print("== class_weights (normalized) summary ==")
        print(f" len={cw.numel()}, min={cw.min().item():.6f}, mean={cw.mean().item():.6f}, max={cw.max().item():.6f}")
        # print small sample (first 20)
        print(" sample class_weights (first 20):", cw[:20].tolist())


    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        simple_aug=args.simple_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        direct_resize=args.direct_resize,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    cw_for_loss = class_weights.clone().detach()


    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = WeightedSoftTargetCrossEntropy(class_weights=cw_for_loss)
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        # Override the loss
        train_loss_fn = torch.nn.CrossEntropyLoss(weight=cw_for_loss)
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # print summary (only on rank 0)
    if args.local_rank == 0:
        print(f"mixup_active = {mixup_active}, prefetcher = {args.prefetcher}, num_aug_splits = {num_aug_splits}")
        _summarize_loss_module(train_loss_fn)
        # Also print whether you're using sampler vs weighted-loss. If you have a sampler variable set earlier, print it:
        use_sampler = getattr(globals(), 'sampler', None)
        if use_sampler is not None:
            print("Using sampler (WeightedRandomSampler) in training loader.")
        else:
            print("Not using WeightedRandomSampler (using loss weighting or default shuffle).")
       
        
    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    if args.evaluate:
        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            if args.local_rank == 0:
                _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

        eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

        if model_ema is not None and not args.model_ema_force_cpu:
            # pick the internal EMA module whether attribute name is `.module` or `.ema`
            ema_model_for_eval = getattr(model_ema, 'module', None) or getattr(model_ema, 'ema', None) or model_ema
            # ensure on same device and eval mode
            try:
                ema_model_for_eval.to(args.device)
            except Exception:
                pass
            ema_model_for_eval.eval()
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(ema_model_for_eval, args.world_size, args.dist_bn == 'reduce')
            ema_eval_metrics = validate(
                ema_model_for_eval, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
            eval_metrics = ema_eval_metrics
        if saver is not None:
            # save proper checkpoint with eval metric
            save_metric = eval_metrics[eval_metric]
            best_metric, best_epoch = saver.save_checkpoint(start_epoch, metric=save_metric)

        return

    #endregion

    #MAIN LOOP    
    try:
        for epoch in range(start_epoch, num_epochs):

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
            
            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
            
            if model_ema is not None and not args.model_ema_force_cpu:
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics
            
            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
            
            maybe_release_cache(threshold_mb=0.9 * torch.cuda.get_device_properties(0).total_memory / 1024**2)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)

            loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            src_for_ema = _unwrap_model(model)
            model_ema.update(src_for_ema)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)

            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    
    return metrics

if __name__ == '__main__':
    main()
