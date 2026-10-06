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

Dataset directory structure and preprocessing requirements are described in the relevant dataset/configuration files.

## Results

The proposed framework achieves:

| Setting | Accuracy |
| --- | ---: |
| 5-way 1-shot | **76.13%** |
| 5-way 5-shot | **89.00%** |

Please refer to the paper for the complete experimental results, comparisons, ablation studies, and evaluation protocol.

## Requirements

The implementation is based on:

- Python
- PyTorch
- CUDA
- `torchvision`
- `timm`
- `scikit-learn`
- Pillow

A complete dependency list is provided in [`requirements.txt`](requirements.txt).

## Installation

Clone the repository:

```bash
git clone https://github.com/rezaes97/two-stage-peft-cdfsl-plant-disease.git
cd two-stage-peft-cdfsl-plant-disease
```

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended for training.

## Data Preparation

1. Download the required datasets from their official sources.
2. Organize them according to the directory structure expected by the repository.
3. Update the corresponding dataset paths/configuration files.
4. Generate or provide the required training/evaluation splits before running the experiments.

> **Important:** Do not commit the original datasets, large checkpoints, or generated experiment outputs to the repository.

## Running the Code

The repository is organized around the two stages of the proposed framework.

### Stage 1 - Offline Domain Adaptation

Use the Stage 1 training scripts/configuration to adapt the pretrained visual representation using the domain-relevant training data.

```bash
# Not implemented yet
```

### Stage 2 - Online Few-Shot Adaptation

After Stage 1, use the Stage 2 implementation to perform lightweight adaptation and evaluate the target few-shot tasks.

```bash
# Not implemented yet
```

See the source files and configuration options in the repository for the exact experiment commands.

## Repository Structure

The exact structure may evolve as the implementation is cleaned up, but the main components are organized around:

```text
two-stage-peft-cdfsl-plant-disease/
├── datasets/
├── models/
├── configs/
├── scripts/
├── utils/
├── train/
├── test/
├── requirements.txt
├── README.md
└── LICENSE
```

## Reproducibility

The reported results depend on the dataset versions, preprocessing, train/test splits, random seeds, model configuration, and training settings used in the paper.

For faithful reproduction, use the configurations and splits provided with the repository and consult the paper for the full experimental protocol.

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
