# SelfMedMAE: Medical Image Analysis with Masked Autoencoders

This project was developed for **CS Special Topics: Image Analysis** at the University of Virginia (UVA). We adapted the Masked Autoencoder (MAE) architecture for self-supervised pre-training on medical imaging datasets, specifically focusing on brain tumor segmentation and medical image analysis tasks.

## Project Overview

This repository implements a self-supervised learning framework for medical image analysis using Masked Autoencoders. The project takes the existing MAE architecture from self-supervised pre-training literature and modifies it to work effectively with medical imaging data, including 2D and 3D medical volumes.

### Key Features

- **Adapted MAE Architecture**: Modified the original Masked Autoencoder architecture to handle medical imaging data formats (2D slices and 3D volumes)
- **Medical Image Preprocessing**: Custom data transformations and normalization tailored for medical datasets
- **Segmentation Pipeline**: Fine-tuning capabilities for downstream segmentation tasks using UNETR
- **Distributed Training**: Support for multi-GPU training on UVA's Rivanna HPC cluster

## Architecture Modifications

We made several key modifications to the standard MAE architecture:

- **Patch Embedding**: Custom patch embedding layers for 2D and 3D medical images
- **Positional Encoding**: Adapted sinusoidal positional embeddings for medical image dimensions
- **Loss Functions**: Custom Fourier-based loss functions for reconstruction
- **Data Handling**: Medical image-specific data loaders and preprocessing pipelines

## Training Infrastructure

All model training was performed on **UVA's Rivanna HPC cluster** using NVIDIA GPUs. The training pipeline supports:

- Self-supervised pre-training with MAE (Stage 1)
- Supervised fine-tuning for segmentation tasks with UNETR (Stage 2)
- Distributed training across multiple GPUs
- Wandb integration for experiment tracking and visualization

## Datasets

The project was tested on several medical imaging datasets including:
- **BTCV**: Multi-organ abdominal CT scans
- **MSD BraTS**: Brain tumor segmentation challenge dataset
- **Task01_BrainTumour**: Brain tumor segmentation dataset

## Repository Structure

```
├── lib/
│   ├── models/          # MAE and UNETR model implementations
│   ├── datasets.py      # Medical image dataset loaders
│   ├── data/            # Medical image transforms and preprocessing
│   └── trainers/        # Training loops for MAE and segmentation
├── configs/             # Configuration files for different experiments
├── main.py             # Main training script
└── requirements.txt    # Python dependencies
```

## Setup

See `SelfMedMAE/README.md` for detailed setup instructions and configuration details.

## Results

The pre-trained models can be fine-tuned for downstream segmentation tasks, demonstrating the effectiveness of self-supervised pre-training on medical imaging data.

## Acknowledgments

- Based on the Masked Autoencoder (MAE) architecture by He et al.
- Developed for CS Special Topics: Image Analysis at UVA
- Training infrastructure provided by UVA Research Computing (Rivanna HPC)

