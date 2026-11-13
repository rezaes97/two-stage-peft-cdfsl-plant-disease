# Two-Stage PEFT for Cross-Domain Few-Shot Plant Disease Recognition

Official implementation of:

**A Two-Stage Parameter-Efficient Framework for Cross-Domain Few-Shot Plant Disease Recognition**

*Saeed Khankalantary and Mohammad Reza Eskandari*  
*Computers and Electronics in Agriculture, 2026*

[Paper DOI: 10.1016/j.compag.2026.111691](https://doi.org/10.1016/j.compag.2026.111691)

---

## Overview

Plant disease recognition in real-world field images is challenging because the available labeled data can be limited and the visual distribution can differ substantially between training and deployment environments.

This repository contains the implementation of a **two-stage parameter-efficient framework for cross-domain few-shot learning (CDFSL)** for plant disease recognition.

The framework separates adaptation into two stages:

1. **Offline domain adaptation**  
   A pretrained visual model is adapted using domain-relevant datasets to improve its representation for the target visual environment.

2. **Online few-shot adaptation**  
   Lightweight task-specific adaptation is performed using only a small number of labeled examples from the target task.

The approach aims to make adaptation more data-efficient and parameter-efficient while retaining useful representations from the pretrained visual backbone.

## Method

The overall pipeline can be summarized as:

```text
Pretrained Vision Transformer
          |
          v
Offline Domain Adaptation
(Domain-relevant datasets)
          |
          v
Adapted Visual Representation
          |
          v
Online Few-Shot Adaptation
(Limited target-domain labels)
          |
          v
Target Disease Classification
```

The implementation uses **Vision Transformers** together with **parameter-efficient adaptation** rather than relying only on full-model fine-tuning.

## Datasets

The experiments use plant-disease and domain-relevant visual datasets including:

- [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset)
- PlantSeg
- PlantWild

The datasets are **not distributed with this repository**. Please obtain each dataset from its original source and follow the corresponding dataset license and terms of use.

Dataset directory structure and preprocessing requirements are described in the relevant dataset/configuration files and in [Data Preparation](#data-preparation) below.

## Results

The paper reports the following accuracies:

| Setting | Accuracy |
| --- | ---: |
| 5-way 1-shot | **76.13%** |
| 5-way 5-shot | **89.00%** |

The launcher defaults documented below were selected to resemble the reported 5-way, 5-shot setup. The 89% figure is the reported experimental result on the combined PlantSeg, PlantDoc, and PlantWild target test datasets; it is not a guarantee that every environment or independently prepared split will reproduce the same value.

Please refer to the paper for the complete experimental results, comparisons, ablation studies, and evaluation protocol.

## Requirements

The implementation is based on:

- Python 3.10 (recommended for this checkout)
- PyTorch and a compatible `torchvision` build
- CUDA-enabled PyTorch for NVIDIA GPU acceleration (recommended, not required for every operation)
- `timm`
- `pytorch-metric-learning`
- `numpy`, Pillow, PyYAML, `scikit-learn`, and SciPy

The complete dependency list is provided in [`requirements.txt`](requirements.txt). Install a PyTorch/torchvision pair appropriate for your operating system and CUDA setup before installing the remaining requirements.

## Installation

Clone the repository and enter its root directory:

```powershell
git clone https://github.com/rezaes97/two-stage-peft-cdfsl-plant-disease.git
cd two-stage-peft-cdfsl-plant-disease
```

Create the default virtual environment in the repository root. The launchers look for this environment next to `run_cdfsl.py`:

```powershell
py -3.10 -m venv .\venv
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

Install a PyTorch/torchvision build that matches your machine. For NVIDIA GPUs, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) and follow its generated command. Then install the repository dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

A CUDA-enabled GPU is recommended for training and repeated episodic evaluation. If you use a different Python environment, the PowerShell launchers also accept `-VenvPath` or `-PythonExe`.

## Data Preparation

1. Download the required datasets from their official sources.
2. Prepare the offline source-class split and target evaluation split following the directory requirements below.
3. Place pretrained checkpoints in a local directory, for example `.\checkpoints`, or provide another path when invoking a script.
4. Keep the saved split JSON files from the experiment you intend to reproduce. The launchers do not regenerate or rewrite class splits.
5. Do not commit original datasets, large checkpoints, or generated experiment outputs to the repository.

### Offline data layout

The offline launcher uses an image-folder layout. For the saved PlantDoc split used by the provided launcher, `train` and `validation` (or `val`) must contain the same **23 source classes**. The five held-out target classes must not occur in either offline folder.

```text
datasets/
└── plantdoc_source_split/
    ├── train/
    │   ├── source_class_1/
    │   └── ...
    └── validation/
        ├── source_class_1/
        └── ...
checkpoints/
└── pretrained_checkpoint.pth
```

Use the clean pretrained checkpoint expected by the model, not a checkpoint trained on the held-out target classes.

### Online target data and splits

The episodic test script takes a target `DatasetRoot` and a `SplitsRoot`. The split location should contain the saved split JSON used by the experiment (for example, `PlantDoc_splits.json`), or it can be the JSON file itself. Follow the layout expected by your target dataset and keep the support, validation, and held-out test classes separated as intended by the experiment. Use the exact split files you used for the reported result.

Example local layout (names are illustrative; paths are supplied to the launcher):

```text
datasets/
└── plantdoc_target/
    ├── data/
    │   └── PlantDoc/
    │       ├── target_class_1/
    │       └── ...
    └── PlantDoc_splits.json
```

The testing code expects the dataset and split layout of the selected experiment; do not assume that a differently organized third-party download can be used without preparing it first.

## Running the Code

The repository contains a unified Python entry point and PowerShell launchers for the saved PlantDoc one-target workflow. Run the commands from the repository root. All example paths are relative to that directory; replace folder and checkpoint names to match your local data.

### Stage 1 - Offline Domain Adaptation

The default PowerShell launcher trains offline LoRA on the 23-class saved source split. It uses ViT-S/16, rank-8 LoRA on QKV, alpha 1, dropout 0.05, AdamW with learning rate `0.001`, 35 epochs, batch size 32, 3 warm-up epochs, seed 42, and simple augmentation enabled. These are launcher defaults and can be overridden with parameters.

```powershell
.\scripts\train_plantdoc_lora.ps1 `
  -DataRoot ".\datasets\plantdoc_source_split" `
  -InitialCheckpoint ".\checkpoints\pretrained_checkpoint.pth" `
  -OutputDir ".\outputs\plantdoc_offline_lora"
```

The launcher checks that the `train` and `validation` folders contain matching source-class names and that there are 23 classes by default. It does not construct new splits. Use `-NoSimpleAug` to disable the default simple augmentation, or pass different values such as `-Epochs`, `-BatchSize`, `-LoraRank`, `-LoraAlpha`, `-LoraDropout`, and `-LoraTargetModules` to change training parameters.

The default offline optimizer is AdamW because this stage uses supervised classification training. The NAdam / ProxyAnchor iteration parameters listed for Stage 2 apply to online episodic adaptation; they are not options for the offline classifier loop.

### Stage 2 - Online Few-Shot Adaptation

After offline training, select the best checkpoint produced by Stage 1 and evaluate the target task. The default evaluation is **5-way, 5-shot, 600 episodes**, with 10 queries per class, ProxyAnchor, NAdam, 80 online iterations, feature depth 4, tuning depth 5, learning rate 0.005, anchor learning rate 5, the offline LoRA adapter loaded as the base adapter, and SSF tuned online. All are overridable.

```powershell
.\scripts\test_plantdoc_1t.ps1 `
  -DatasetRoot ".\datasets\plantdoc_target\data" `
  -SplitsRoot ".\datasets\plantdoc_target" `
  -Checkpoint ".\outputs\plantdoc_offline_lora\model_best.pth.tar" `
  -OutputDir ".\outputs\plantdoc_1t_5shot" `
  -TargetDataset "PlantDoc"
```

Adjust `-DatasetRoot`, `-SplitsRoot`, and `-Checkpoint` to match your local layout. The test launcher consumes the existing split JSON files and does not generate new splits. To change the number of support examples, use `-Shots 1` or `-Shots 5`; episode count, query count, tuning depth, learning rates, adapter options, and seed are also exposed as parameters.

For example, to evaluate a 1-shot setting, add `-Shots 1` to the command. For other target datasets, point the two data/split parameters to the appropriate prepared data and split files and set `-TargetDataset` accordingly. The reported 89% result refers to the combined PlantSeg, PlantDoc, and PlantWild target test evaluation; a single run on one target dataset should not be presented as the combined result.

### Alternative unified entry point

The lower-level stages can also be called using `run_cdfsl.py` and the options supported by the corresponding Python script:

```powershell
# Show options for the offline or online stage
.\venv\Scripts\python.exe .\run_cdfsl.py offline --help
.\venv\Scripts\python.exe .\run_cdfsl.py online --help
```

The unified entry point can chain both stages with `all` and `--online` as a separator; see its command help and the stage-specific arguments before using it for a custom experiment. The PowerShell episodic launcher above is the more direct route for the saved 5-way/5-shot one-target evaluation setup.

## Repository Structure

```text
two-stage-peft-cdfsl-plant-disease/
├── offline/
│   ├── adapt.py
│   ├── data/
│   ├── proj_models/
│   └── proj_utils/
├── online/
│   ├── data/
│   ├── models/
│   ├── test_configs.py
│   ├── test_extractor.py
│   ├── test_extractor_episodic.py
│   └── utils.py
├── scripts/
│   ├── train_plantdoc_lora.ps1
│   └── test_plantdoc_1t.ps1
├── checkpoints/                 # local pretrained weights (not tracked)
├── datasets/                    # local datasets and splits (not tracked)
├── outputs/                     # generated checkpoints and logs (not tracked)
├── requirements.txt
├── run_cdfsl.py
├── README.md
└── LICENSE
```

## Reproducibility

The reported results depend on dataset versions, preprocessing, train/test splits, random seeds, checkpoint source, model configuration, and optimization settings.

- Use the exact prepared split JSON files from the experiment being reproduced; the launchers do not generate splits.
- The offline launcher checks the 23-class source split by default and validates the class names between training and validation.
- The 5-way, 5-shot test defaults reflect the reported online settings but cannot guarantee the same accuracy for a different checkpoint or target split.
- The launcher defaults resemble the reported **89% accuracy on the combined PlantSeg, PlantDoc, and PlantWild 5-way, 5-shot target test evaluation**. This is a reported result, not a newly verified reproduction in every environment.
- Pretrained weights, target datasets, and generated outputs are external and should not be committed to Git.
- The code has been consolidated into a single offline/online implementation. Verify the PyTorch, torchvision, TIMM, CUDA, and checkpoint combination in your environment before treating any new run as an exact reproduction.

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{khankalantary2026twostage,
  title   = {A Two-Stage Parameter-Efficient Framework for Cross-Domain Few-Shot Plant Disease Recognition},
  author  = {Khankalantary, Saeed and Eskandari, Mohammad Reza},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026},
  doi     = {10.1016/j.compag.2026.111691}
}
```

## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE).

Third-party datasets, pretrained models, and external code remain subject to their respective licenses and terms of use.
