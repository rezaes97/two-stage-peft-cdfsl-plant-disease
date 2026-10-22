import argparse
import utils


def get_test_configs():
    parser = argparse.ArgumentParser('CDFSL online adaptation')

    # test args
    parser.add_argument('--test_size', type=int, default=600, metavar='TEST_SIZE',
                        help='The number of test episodes sampled')
    parser.add_argument('--test_type', type=str, choices=['standard', '5shot', '1shot'], default='standard',
                        help="meta-test type, standard varying number of ways and shots as in Meta-Dataset, 1shot for "
                             "five-way-one-shot and 5shot for varying-way-five-shot evaluation.")
    parser.add_argument('--pretrained_setting', type=str, default='MDL', choices=['MDL', 'SDL','SDL_E'],
                        help="The pre-training setting. Will be used to determine the tuning depth depending on "
                             "whether a dataset is in-domain or out-of-domain.")
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    parser.add_argument("--deterministic", action="store_true",
                    help="set cudnn deterministic=True (may slow/affect perf)")

    parser.add_argument('--max_iter', type=int, default=80, help="The number of fine-tuning iterations")
    parser.add_argument('--feature_depth', type=int, default=4, help="The feature fusion depth")
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate for SSF parameters")
    parser.add_argument('--lr_anchors', type=float, default=5.0, help="learning rate for anchors in proxy-anchor")
    parser.add_argument('--test_optimizer', default='nadam', type=str,
                        help="The type of optimizer that is used for fine-tuning. Here, we use the same type of "
                             "optimizer for both SSF and proxy-anchor")
    parser.add_argument('--embedding_loss_type', default='proxy_anchor', type=str,
                        choices=['proxy_anchor', 'proxy_anchor_custom'], help="Choice of fine-tuning objective")
    parser.add_argument('--use_proxy_acc', default=False, type=utils.bool_flag,
                        help="Using proxy-anchor for Query classification. Default is False and we instead use NCC "
                             "for query classification. Morevoer, if using NCC for fine-tuning, NCC will "
                             "automatically be used for query classification as well, regardless of the value of this "
                             "flag.")
    parser.add_argument('--ssf_init', default=False,
                        type=utils.bool_flag, help="Whether SSF parameters have constant or random initializations. "
                                                   "Setting this to True means parameters will have random "
                                                   "initialization while False meanse no randomizations will be "
                                                   "performed.")

    # path args
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="path to the Stage 1/pretrained feature extractor checkpoint")
    parser.add_argument('--out_dir', default='./', type=str, metavar='PATH',
                        help='directory to output the result')
    parser.add_argument('--checkpoint_freq', type=int, default=50,
                        help='The frequency (measured in number of fine-tuning iterations) for saving '
                             'intermediate checkpoints. Default is 20.')
    parser.add_argument('--keep_last_k', type=int, default=5,
                        help='The number of latest checkpoints to keep. Default is 5.')

    # ─── Custom‑extractor flags ────────────────────────────────
    parser.add_argument(
        '--dataset_root',    type=str, required=True,
        help='Root folder containing each dataset\'s image directories'
    )
    parser.add_argument(
        '--splits_root',     type=str, required=False,
        help='Folder containing per-dataset splits JSON'
    )
    parser.add_argument('--episodic',    type=utils.bool_flag, default=True,
                        help='Use episodic data sampling (otherwise random)')
    parser.add_argument('--n_way',      type=int, default=5,
                        help='Number of classes per episode')
    parser.add_argument('--n_shot',     type=int, default=5,
                        help='Support examples per class')
    parser.add_argument('--n_query',    type=int, default=10,
                        help='Query examples per class')
    parser.add_argument('--tuning_depth', type=int, default=7,
                        help='Depth of SSF layers to fine-tune')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='PyTorch DataLoader workers')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--pin_memory', type=utils.bool_flag, default=False,
                        help='Pin memory for DataLoader (default: False to reduce host memory pressure)')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate every N iterations (default: 10)')
    parser.add_argument('--no_aug', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--simple_aug', action='store_true',
                        help='Use simple data augmentation (random crop, horizontal flip)')
    parser.add_argument('--aa', type=str, default='',
                        help='Use AutoAugment with the given policy (e.g., "rand-m9-mstd0.5-inc1")')
    
    # Augmentation presets affect training images only; validation remains deterministic.
    parser.add_argument('--aug_preset', '--aug-preset', dest='aug_preset', type=str,
                        default='legacy',
                        choices=['legacy', 'none', 'simple', 'plant-light', 'plant-strong', 'randaugment'],
                        help='Training augmentation preset. legacy preserves the previous transform flags.')
    parser.add_argument('--normalization', type=str, default='legacy-half',
                        choices=['legacy-half', 'imagenet'],
                        help='Normalization. legacy-half preserves old experiments; imagenet uses ImageNet statistics.')
    parser.add_argument('--class-balanced-sampling', '--class_balanced_sampling',
                        dest='class_balanced_sampling', action='store_true',
                        help='Use inverse-class-frequency sampling for training (not a weighted proxy loss).')
    parser.add_argument('--input_size', '--input-size', dest='input_size', type=int, default=224)
    parser.add_argument('--direct_resize', '--direct-resize', dest='direct_resize', action='store_true')
    parser.add_argument('--scale', type=float, nargs=2, default=[0.08, 1.0], metavar=('MIN', 'MAX'))
    parser.add_argument('--ratio', type=float, nargs=2, default=[0.75, 1.3333333333], metavar=('MIN', 'MAX'))
    parser.add_argument('--hflip', type=float, default=0.5)
    parser.add_argument('--vflip', type=float, default=0.0)
    parser.add_argument('--color-jitter', '--color_jitter', dest='color_jitter', type=float, default=0.4)
    parser.add_argument('--reprob', type=float, default=0.0, help='Random erasing probability in legacy mode.')
    parser.add_argument('--remode', type=str, default='pixel', choices=['pixel', 'const', 'rand'])
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--aug_splits', '--aug-splits', dest='aug_splits', type=int, default=0)
    parser.add_argument('--train_interpolation', '--train-interpolation', dest='train_interpolation',
                        type=str, default='random')

    # Adapter selection for ViT
    parser.add_argument('--base_adapters', type=str, default='',
                        help="Comma-separated list of adapters to include in the base model (houlsby, ssf, lora)")
    parser.add_argument('--tuning_adapters', type=str, default='ssf',
                        help="Comma-separated list of adapters to train in the last tuning_depth layers (houlsby, ssf, lora)")
    
    # Houlsby-specific
    parser.add_argument('--houl_adapter_reduction', type=int, default=32, help="Houlsby adapter reduction ratio")
    parser.add_argument('--houl_adapter_dropout', type=float, default=0.0, help="Houlsby adapter dropout")

    # LoRA-specific
    parser.add_argument('--lora_r', type=int, default=4, help="LoRA rank")
    parser.add_argument('--lora_alpha', type=float, default=1, help="LoRA scaling factor")
    parser.add_argument('--lora_target_modules', type=str, default='qkv', help="Comma-separated modules to apply LoRA (attn, mlp, etc)")
    parser.add_argument('--lora_dropout', type=float, default=0.0, help="LoRA dropout")

 
    args = vars(parser.parse_args())

    args['base_adapters'] = [a.strip() for a in args['base_adapters'].split(',') if a.strip()]
    args['tuning_adapters'] = [a.strip() for a in args['tuning_adapters'].split(',') if a.strip()]
    args['lora_target_modules'] = [m.strip() for m in args['lora_target_modules'].split(',') if m.strip()]
    
    return args
