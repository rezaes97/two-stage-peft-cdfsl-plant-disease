Two-Stage CDFSL for Plant Disease Recognition

Official implementation of a two-stage parameter-efficient framework for cross-domain few-shot plant disease recognition.

# Paper

A Two-Stage Parameter-Efficient Framework for Cross-Domain Few-Shot Plant Disease Recognition

Saeed Khankalantary, Mohammad Reza Eskandari
Computers and Electronics in Agriculture, 2026

DOI: 10.1016/j.compag.2026.111691

# Overview

Recognizing plant diseases in real-world environments is challenging because available labeled data can be limited and visual conditions can differ substantially across domains.

This repository contains the implementation of a two-stage parameter-efficient adaptation framework for cross-domain few-shot plant disease recognition. The framework first adapts a pretrained visual model to relevant visual domains and subsequently performs lightweight task-specific adaptation using only a small number of labeled examples from the target task.

The approach is designed to reduce the number of trainable parameters while retaining useful representations from the pretrained backbone.

# Framework

The proposed approach consists of two main stages:

## Stage 1 — Offline Domain Adaptation

A pretrained visual backbone is adapted using domain-relevant datasets to obtain representations that are better suited to the visual characteristics of the target problem.

## Stage 2 — Online Few-Shot Adaptation

Lightweight task-specific adaptation is performed using the limited labeled examples available for the target classification task.

This separation allows most of the adaptation to be performed before deployment while keeping the online adaptation stage parameter-efficient.

# Datasets

The experiments use multiple plant-disease and domain-relevant datasets, including:

PlantDoc
PlantSeg
PlantWild

Please refer to the paper for the complete experimental setup, dataset splits, and preprocessing details.

Dataset files are not included in this repository. Please obtain each dataset from its original source and follow the corresponding dataset license and terms of use.

# Results

The proposed framework achieves:

Setting	Accuracy
5-way 1-shot	76.13%
5-way 5-shot	89.00%

See the paper for complete results, comparisons, ablations, and experimental details.

# Installation

Clone the repository:

git clone https://github.com/<USERNAME>/two-stage-peft-cdfsl-plant-disease.git
cd two-stage-peft-cdfsl-plant-disease

Create the required Python environment and install the dependencies:

pip install -r requirements.txt

The code is intended to run with PyTorch and CUDA-enabled hardware.

# Data Preparation

Download the required datasets from their official sources and organize them according to the expected directory structure.

Update the dataset paths in the corresponding configuration files or command-line arguments before running the experiments.

# Training
Stage 1

Run the offline/domain-adaptation stage using:

* Under Construction

Run the few-shot adaptation/evaluation stage using:

* Under construction

The exact commands and configuration options are provided in the repository scripts.

# Repository Structure
two-stage-peft-cdfsl-plant-disease/
├── datasets/
├── models/
├── scripts/
├── configs/
├── utils/
├── train/
├── test/
├── requirements.txt
├── LICENSE
└── README.md
# Reproducibility

Experiments reported in the paper depend on the dataset versions, preprocessing, training configuration, and random seeds used during experimentation.

Additional implementation details and configuration files are included in this repository where applicable.

# Citation

If you use this code or find this work useful, please cite:

@article{khankalantary2026twostage,
  title   = {A Two-Stage Parameter-Efficient Framework for Cross-Domain Few-Shot Plant Disease Recognition},
  author  = {Khankalantary, Saeed and Eskandari, Mohammad Reza},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026},
  doi     = {10.1016/j.compag.2026.111691}
}
# License

This repository is released under the MIT License. See LICENSE.

Third-party datasets, pretrained models, and external code remain subject to their respective licenses and terms of use.
