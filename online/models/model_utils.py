import torch
from torch import nn
import torch.nn.functional as F

from utils import load_pretrained_weights
from utils import device
from models.vision_transformer import vit_small
from models.vision_transformer_all_adapters import vit_small_extended

sigmoid = nn.Sigmoid()

def count_and_list_trainable(model, tag):
    print("\n=== All model parameters ===")
    for name, p in model.named_parameters():
        shape = tuple(p.shape)
        print(f"  {name:60s}  {shape}")

    trainable = [(n, p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    total = sum(size for _,size in trainable)
    print(f"\n>>> {tag}:")
    print(f"  trainable param count = {total:,}")
    print("  trainable params:")
    for name, size in trainable:
        print(f"    {name:60s}  ({size:,})")
    return total

def print_trainable_params(model):
    total = 0
    trainable_total = 0
    print("Trainable parameters:")
    for name, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable_total += num
            print(f"  + {name}  ({num})")
    print(f"Total params: {total:,}, trainable: {trainable_total:,} ({100*trainable_total/total:.3f}%)")

def cosine_sim(embeds, prots):
    prots = prots.unsqueeze(0)
    embeds = embeds.unsqueeze(1)
    return F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30)


def get_pre_trained_vit(ckpt_path):
    model = vit_small(patch_size=16, num_classes=0)
    load_pretrained_weights(model, ckpt_path, 'state_dict', 'vit_small', 16)
    model.to(device)

    return model


def get_vit_with_adapter_model(ckpt_path, checkpoint_key='state_dict', **kwargs):
    """Build the online adapter model and load an offline checkpoint.

    checkpoint_key='auto' selects state_dict_ema when available, then state_dict.
    """
    kwargs.setdefault('num_classes', 0)
    model = vit_small_extended(**kwargs)
    hidden_dim = model.num_features
    load_pretrained_weights(model, ckpt_path, checkpoint_key, 'vit_small', 16, verbose=True)
    model.to(device)
    return model, hidden_dim


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def feature_extraction_and_fusion(model, images, feature_depth):

    features = model.get_intermediate_layers(images, feature_depth)
    features = [x[:, 0] for x in features]
    features = torch.cat(features, dim=-1)

    features = F.normalize(features, dim=1)  # l2 normalize the features
    return features
