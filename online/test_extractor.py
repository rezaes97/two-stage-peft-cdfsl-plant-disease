import os
import sys
import logging
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from utils import device, set_seed
from test_configs import get_test_configs
from models.proxy_tuning import proxy_tuning_streaming 

class ImageFolderSplit(Dataset):
    """
    Given a dataset_root that contains:
      train/  (class folders) and validation/ or val/ (class folders)
    builds a streaming dataset that yields (image_tensor, label)
    """
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(.5, .5)
        ])
        # locate split dir
        entries = sorted([e for e in os.listdir(root) if os.path.isdir(os.path.join(root, e))])
        low_map = {e.lower(): e for e in entries}
        if split == 'train':
            if 'train' in low_map:
                split_root = os.path.join(root, low_map['train'])
            else:
                # if no train dir, assume root contains class folders
                split_root = root
        else:
            # validation split
            if 'validation' in low_map:
                split_root = os.path.join(root, low_map['validation'])
            elif 'val' in low_map:
                split_root = os.path.join(root, low_map['val'])
            else:
                # no validation dir -> empty dataset
                split_root = None

        self.class_names = []
        self.class_to_idx = {}
        self.samples = []  # list of (fullpath, label)

        if split_root and os.path.isdir(split_root):
            classes = sorted([d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d))])
            self.class_names = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            for cls in classes:
                cls_dir = os.path.join(split_root, cls)
                for fn in sorted(os.listdir(cls_dir)):
                    full = os.path.join(cls_dir, fn)
                    self.samples.append((full, self.class_to_idx[cls]))
        else:
            # empty
            self.class_names = []
            self.class_to_idx = {}
            self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, lbl = self.samples[idx]
        img = Image.open(fp).convert('RGB')
        t = self.transform(img)
        return t, torch.tensor(lbl, dtype=torch.long)


def _get_seed_and_flags(args):
    """Support args that are Namespace-like or dict-like"""
    # seed
    seed = None
    deterministic = False
    if hasattr(args, "seed"):
        seed = getattr(args, "seed")
    else:
        try:
            seed = args.get("seed", None)
        except Exception:
            seed = None
    if seed is None:
        seed = 42

    if hasattr(args, "deterministic"):
        deterministic = getattr(args, "deterministic")
    else:
        try:
            deterministic = bool(args.get("deterministic", False))
        except Exception:
            deterministic = False

    return int(seed), bool(deterministic)


def main():
    args = get_test_configs()
    seed_val, deterministic_flag = _get_seed_and_flags(args)
    set_seed(seed_val, deterministic=deterministic_flag)
    print(f"Using seed={seed_val}, deterministic={deterministic_flag}")

    for name in ("PIL.TiffTags", "PIL.TiffImagePlugin", "PIL.ImageFile"):
        logging.getLogger(name).setLevel(logging.WARNING)
    
    # setup output directory
    out_dir = args['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    partial_dir = os.path.join(out_dir, 'partial')
    os.makedirs(partial_dir, exist_ok=True)

    # configure logging
    log_file = os.path.join(out_dir, 'debug.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Starting meta-testing run")

    dataset_root = args['dataset_root']

    # --- Build transforms depending on args ---
    try:
        from timm.data import create_transform as timm_create_transform
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        _HAVE_TIMM = True
    except Exception:
        _HAVE_TIMM = False
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    normalization = args.get('normalization', 'legacy-half')
    if normalization == 'imagenet':
        mean, std = tuple(IMAGENET_DEFAULT_MEAN), tuple(IMAGENET_DEFAULT_STD)
    else:
        # Retained as the default to avoid silently changing previously reported results.
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    normalize = transforms.Normalize(mean=mean, std=std)

    def _to_tuple_float(v):
        if v is None:
            return None
        return tuple(float(x) for x in v)

    input_size = int(args.get('input_size', 224))
    aug_preset = args.get('aug_preset', 'legacy')
    no_aug = bool(args.get('no_aug', False))
    simple_aug = bool(args.get('simple_aug', False))
    direct_resize = bool(args.get('direct_resize', False))
    scale = _to_tuple_float(args.get('scale', (0.08, 1.0)))
    ratio = _to_tuple_float(args.get('ratio', (3. / 4., 4. / 3.)))
    hflip = float(args.get('hflip', 0.5))
    vflip = float(args.get('vflip', 0.0))
    color_jitter = float(args.get('color_jitter', 0.4))
    aa_policy = args.get('aa', None)
    re_prob = float(args.get('reprob', 0.0))
    re_mode = args.get('remode', 'pixel')
    re_count = int(args.get('recount', 1))
    aug_splits = int(args.get('aug_splits', 0))
    train_interp = args.get('train_interpolation', 'random')
    bilinear = transforms.InterpolationMode.BILINEAR

    if aug_preset == 'none' or (aug_preset == 'legacy' and no_aug):
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])
        print("Training augmentation: none")
    elif aug_preset == 'simple' or (aug_preset == 'legacy' and simple_aug):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio, interpolation=bilinear),
            transforms.RandomHorizontalFlip(p=hflip),
            transforms.ToTensor(),
            normalize,
        ])
        print("Training augmentation: simple random crop + horizontal flip")
    elif aug_preset == 'plant-light':
        # Conservative geometric changes and mild color jitter preserve symptom/color cues.
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.65, 1.0), ratio=(0.80, 1.25), interpolation=bilinear),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(degrees=15, interpolation=bilinear, fill=128),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.10, hue=0.015
            )], p=0.7),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.08),
            transforms.ToTensor(),
            normalize,
        ])
        print("Training augmentation: plant-light")
    elif aug_preset == 'plant-strong':
        # Stronger domain robustness ablation; inspect for loss of small-lesion/detail cues.
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.50, 1.0), ratio=(0.75, 1.3333333333), interpolation=bilinear),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomRotation(degrees=25, interpolation=bilinear, fill=128),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.25, contrast=0.20, saturation=0.15, hue=0.02
            )], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.15),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.10),
            transforms.ToTensor(),
            normalize,
        ])
        print("Training augmentation: plant-strong")
    elif aug_preset == 'randaugment':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.65, 1.0), ratio=(0.75, 1.3333333333), interpolation=bilinear),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=5),
            transforms.ToTensor(),
            normalize,
        ])
        print("Training augmentation: RandAugment (2 operations, magnitude 5)")
    else:
        # "legacy": preserve the prior TIMM transform path and its --no_aug/--simple_aug/--aa flags.
        if _HAVE_TIMM:
            try:
                train_transform = timm_create_transform(
                    input_size=input_size,
                    is_training=True,
                    use_prefetcher=False,
                    no_aug=False,
                    simple_aug=False,
                    direct_resize=False,
                    scale=scale,
                    ratio=ratio,
                    hflip=hflip,
                    vflip=vflip,
                    color_jitter=color_jitter,
                    auto_augment=aa_policy,
                    interpolation=train_interp,
                    mean=mean,
                    std=std,
                    re_prob=re_prob,
                    re_mode=re_mode,
                    re_count=re_count,
                    re_num_splits=aug_splits,
                    separate=False
                )
                print("Training augmentation: legacy TIMM transforms")
            except Exception as exc:
                print(f"Warning: timm.create_transform failed ({exc}); using torchvision fallback.")
                fallback = [
                    transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio, interpolation=bilinear),
                    transforms.RandomHorizontalFlip(p=hflip),
                    transforms.RandomVerticalFlip(p=vflip),
                    transforms.ColorJitter(
                        brightness=color_jitter, contrast=color_jitter, saturation=color_jitter
                    ),
                ]
                if aa_policy:
                    fallback.append(transforms.RandAugment(num_ops=2, magnitude=5)
                                    if str(aa_policy).lower().startswith('rand')
                                    else transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET))
                train_transform = transforms.Compose(fallback + [transforms.ToTensor(), normalize])
        else:
            fallback = [
                transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio, interpolation=bilinear),
                transforms.RandomHorizontalFlip(p=hflip),
                transforms.RandomVerticalFlip(p=vflip),
                transforms.ColorJitter(
                    brightness=color_jitter, contrast=color_jitter, saturation=color_jitter
                ),
            ]
            if aa_policy:
                fallback.append(transforms.RandAugment(num_ops=2, magnitude=5)
                                if str(aa_policy).lower().startswith('rand')
                                else transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET))
            train_transform = transforms.Compose(fallback + [transforms.ToTensor(), normalize])
            print("Training augmentation: torchvision fallback")

    # Validation/test transforms remain deterministic and never apply random augmentation.
    if direct_resize:
        val_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        crop_pct = 0.875
        scale_size = int(input_size / crop_pct)
        val_transform = transforms.Compose([
            transforms.Resize(scale_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])

    # build train / val streaming datasets


    train_ds = ImageFolderSplit(dataset_root, split='train', transform=train_transform)
    val_ds   = ImageFolderSplit(dataset_root, split='validation', transform=val_transform)
    
    if len(train_ds) == 0:
        raise RuntimeError(f"No training images found under {dataset_root} (expecting 'train/' subfolder or classes directly)")
    
    batch_size = int(args.get('batch_size', 128))
    num_workers = max(0, int(args.get('num_workers', 8)))
    pin_memory = bool(args.get('pin_memory', False))

    gen = torch.Generator()
    gen.manual_seed(seed_val)

    train_sampler = None
    if bool(args.get('class_balanced_sampling', False)):
        train_labels = torch.as_tensor([label for (_path, label) in train_ds.samples], dtype=torch.long)
        class_counts = torch.bincount(train_labels)
        per_sample_weights = class_counts[train_labels].to(torch.float64).reciprocal()
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(seed_val + 1)
        train_sampler = WeightedRandomSampler(
            weights=per_sample_weights,
            num_samples=len(per_sample_weights),
            replacement=True,
            generator=sampler_generator,
        )
        print("Training sampler: inverse-frequency class-balanced sampling")

    worker_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        worker_kwargs.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        generator=gen,
        **worker_kwargs,
    )

    val_workers = max(0, num_workers // 2)
    val_worker_kwargs = {"num_workers": val_workers, "pin_memory": pin_memory}
    if val_workers > 0:
        val_worker_kwargs.update(persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **val_worker_kwargs
    ) if len(val_ds) > 0 else None

    # Deterministic class-accuracy reporting must not reuse the weighted/random training sampler.
    eval_batch_size = min(batch_size, 256)
    eval_workers = max(0, num_workers // 2)
    eval_worker_kwargs = {"num_workers": eval_workers, "pin_memory": pin_memory}
    if eval_workers > 0:
        eval_worker_kwargs.update(persistent_workers=False, prefetch_factor=2)
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        **eval_worker_kwargs,
    )

    # run streaming proxy tuning
    save_dir = os.path.join(args['out_dir'], 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    train_args = {
        'optimizer_type': args['test_optimizer'],
        'max_iter': args['max_iter'],
        'feature_depth': args['feature_depth'],
        'lr': args['lr'],
        'lr_proxy': args['lr_anchors'],
        'proxy_acc': args['use_proxy_acc'],
        'embedding_loss_type': args['embedding_loss_type'],
        'aug_preset': args.get('aug_preset', 'legacy'),
        'normalization': args.get('normalization', 'legacy-half'),
        'class_balanced_sampling': bool(args.get('class_balanced_sampling', False)),
        'checkpoint_freq': args.get('checkpoint_freq', 50),
        'keep_last_k': args.get('keep_last_k', 5),
        'model_init_kwargs': {
            # forward through other model init args
            'tuning_depth': args.get('tuning_depth'),
            'ssf_init': args.get('ssf_init'),
            'houl_adapter_reduction': args.get('houl_adapter_reduction'),
            'houl_adapter_dropout': args.get('houl_adapter_dropout'),
            'lora_r': args.get('lora_r'),
            'lora_alpha': args.get('lora_alpha'),
            'lora_dropout': args.get('lora_dropout'),
            'lora_target_modules': args.get('lora_target_modules'),
            'base_adapters': args.get('base_adapters'),
            'tuning_adapters': args.get('tuning_adapters'),
        }
    }

    # call streaming proxy tuning
    proxy_tuning_streaming(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=args.get('checkpoint_path'),
        optimizer_type=args['test_optimizer'],
        max_iter=int(args['max_iter']),
        feature_depth=int(args['feature_depth']),
        lr=float(args['lr']),
        lr_proxy=float(args['lr_anchors']),
        proxy_acc=bool(args['use_proxy_acc']),
        embedding_loss_type=args['embedding_loss_type'],
        save_dir=save_dir,
        checkpoint_freq=int(args.get('checkpoint_freq', 50)),
        keep_last_k=int(args.get('keep_last_k', 5)),
        train_args=train_args,
        device=device,
        patience_iters=100,
        eval_every=args['eval_every'],
        eval_train_loader=train_eval_loader,
    )

    print("Streaming proxy tuning finished.")


if __name__ == '__main__':
    main()
