# SelfMedMAE: Medical Image Analysis with Masked Autoencoders

This project was developed for **CS Special Topics: Image Analysis** at the University of Virginia (UVA). We adapted the Masked Autoencoder (MAE) architecture for self-supervised pre-training on medical imaging datasets, specifically focusing on brain tumor segmentation and medical image analysis tasks.

## Project Overview

This repository implements a self-supervised learning framework for medical image analysis using Masked Autoencoders. The project takes the existing MAE architecture from self-supervised pre-training literature and modifies it to work effectively with medical imaging data, including 2D and 3D medical volumes.

### Key Features

- **3D to 2D Conversion Pipeline**: Developed a comprehensive preprocessing pipeline to convert 3D medical volumes (CT scans, MRIs) into 2D slices for efficient MAE training, including medical format support (`.mhd`, `.nii`), slice extraction, and dual processing paths (true 2D and pseudo-3D)
- **Adapted MAE Architecture**: Modified the original Masked Autoencoder architecture to handle medical imaging data formats (2D slices and 3D volumes)
- **Medical Image Preprocessing**: Custom data transformations and normalization tailored for medical datasets, including intensity scaling, spatial resampling, and anatomical-aware data augmentation
- **Segmentation Pipeline**: Fine-tuning capabilities for downstream segmentation tasks using UNETR with both 2D and 3D model variants
- **Distributed Training**: Support for multi-GPU training on UVA's Rivanna HPC cluster

## Architecture Modifications

We made several key modifications to the standard MAE architecture:

- **Patch Embedding**: Custom patch embedding layers for 2D and 3D medical images
- **Positional Encoding**: Adapted sinusoidal positional embeddings for medical image dimensions
- **Loss Functions**: Custom Fourier-based loss functions for reconstruction
- **Data Handling**: Medical image-specific data loaders and preprocessing pipelines

## 3D to 2D Processing Pipeline

A critical component of this project was developing a comprehensive pipeline to process 3D medical volumes into 2D slices suitable for MAE pre-training. Our implementation supports multiple approaches:

### Pipeline Overview

1. **3D Volume Loading**:

   - Loads medical imaging formats (`.mhd`, `.mhd/.raw`, `.nii`, `.nii.gz`) using SimpleITK
   - Handles multi-modal medical data (e.g., multi-channel brain MRIs)

2. **Slice Extraction**:

   - For 2D MAE training, extracts 2D slices from 3D volumes (middle slice extraction for initial preprocessing)
   - Normalizes intensity values to 0-255 range for grayscale conversion
   - Maintains proper orientation using MONAI's orientation transforms (RAS standard)

3. **Dual Processing Paths**:

   - **True 2D Pipeline**: Processes 2D slices directly for 2D MAE models, preserving computational efficiency
   - **Pseudo-3D Pipeline**: Converts 2D images to pseudo-3D volumes by replicating along depth dimension (to patch_size) for compatibility with 3D models when needed

4. **Medical-Specific Transformations**:
   - Intensity normalization tailored for medical imaging ranges
   - Spatial resampling with configurable spacing parameters
   - Foreground cropping to remove background noise
   - Data augmentation (random flips, rotations, intensity scaling) while preserving anatomical validity

This pipeline enables efficient training on 2D slices while maintaining compatibility with both 2D and 3D model architectures, significantly reducing computational requirements compared to full 3D volume processing.

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
