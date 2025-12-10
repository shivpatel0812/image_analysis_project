# Setup Guide for SelfMedMAE

This guide will walk you through setting up the SelfMedMAE repository for medical image analysis using Masked Autoencoders (MAE).

## Prerequisites

- **Python**: 3.8+ (The README shows PyTorch 1.7.1 built for Python 3.8 based on the package version strings `py3.8`/`py38`, but newer Python versions should work with newer PyTorch versions)
- **CUDA**: 10.1+ (for GPU support)
- **Git**: To clone/download the repository

## Step 1: Create a Python Environment

It's recommended to use a virtual environment or conda environment:

### Option A: Using Conda (Recommended)

```bash
# Create a new conda environment
# Note: The original package versions specify Python 3.8, but you can use Python 3.8-3.11
conda create -n selfmedmae python=3.8  # or python=3.9, python=3.10, python=3.11
conda activate selfmedmae
```

### Option B: Using venv

```bash
# Create a virtual environment
python3 -m venv venv  # or python3.8, python3.9, etc.
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

## Step 2: Install PyTorch

Install PyTorch with CUDA support (matching the versions in README):

```bash
# For CUDA 10.1 (as specified in README)
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=10.1 -c pytorch

# OR if you have a newer CUDA version, install compatible PyTorch:
# For CUDA 11.x:
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch
# For CUDA 11.8+:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Step 3: Install Required Packages

Install the core dependencies:

```bash
# Install MONAI (weekly development version as specified)
pip install monai-weekly==0.9.dev2152

# Install other required packages
pip install nibabel==3.2.1
pip install omegaconf==2.1.1
pip install timm==0.4.12
pip install wandb  # For logging and visualizations
pip install scikit-learn  # For sklearn.metrics
pip install numpy
pip install tqdm  # For progress bars
```

**Note**: If `monai-weekly==0.9.dev2152` is not available, try:

```bash
pip install monai
```

## Step 4: Verify Installation

Test that PyTorch can access your GPU:

```python
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

## Step 5: Download Medical Imaging Datasets

The project requires medical imaging datasets. Download and prepare:

### For 3D Medical Image Segmentation:

1. **BTCV Dataset** (for abdominal organ segmentation):

   - Register and download from: https://www.synapse.org/#!Synapse:syn3193805/wiki/217752
   - Extract the dataset to a directory (e.g., `/path/to/BTCV`)

2. **MSD_BraTS Dataset** (for brain tumor segmentation):
   - Download from: http://medicaldecathlon.com/
   - Extract the dataset to a directory (e.g., `/path/to/MSD_BraTS`)

### For 2D Medical Image Classification:

- **CXR14** (Chest X-ray dataset)
- **ImageNet-100** (subset of ImageNet)

## Step 6: Update Configuration Files

Update the configuration files in the `configs/` directory with your data paths:

### For BTCV (3D):

Edit `configs/mae3d_btcv_1gpu.yaml`:

```yaml
data_path: /path/to/your/BTCV # Update this path
output_dir: /path/to/your/output/${run_name} # Update this path
```

### For MSD_BraTS (3D):

Edit `configs/mae3d_msdbrats_1gpu.yaml`:

```yaml
data_path: /path/to/your/MSD_BraTS # Update this path
output_dir: /path/to/your/output/${run_name} # Update this path
```

### For 2D datasets:

Update `configs/mae_cxr14.yaml` and `configs/mae_im100.yaml` similarly.

## Step 7: Prepare Dataset JSON Files

The project uses JSON files to specify dataset splits. For BTCV, it references `transunet.json`. You may need to:

1. Create a JSON file listing your training/validation data
2. Update the `json_list` parameter in the config files
3. The JSON format should match MONAI's expected format (see MONAI documentation)

Example structure:

```json
{
  "training": [
    {"image": "path/to/image1.nii.gz", "label": "path/to/label1.nii.gz"},
    ...
  ],
  "validation": [...]
}
```

## Step 8: Set Up Wandb (Optional but Recommended)

1. Create a Wandb account at https://wandb.ai
2. Login from command line:
   ```bash
   wandb login
   ```
3. Or disable Wandb by setting `disable_wandb: true` in config files

## Step 9: Test the Setup

Run a quick test to verify everything works:

```bash
# Test import
python -c "import torch; import monai; import timm; import wandb; print('All imports successful!')"
```

## Step 10: Run Training (Example)

Once everything is set up, you can start training:

### Stage 1: MAE Pre-training

```bash
python main.py \
    configs/mae3d_btcv_1gpu.yaml \
    --mask_ratio=0.125 \
    --run_name='mae3d_sincos_vit_base_btcv_mr125'
```

### Stage 2: UNETR Fine-tuning (after pre-training)

```bash
python main.py \
    configs/unetr_btcv_1gpu.yaml \
    --lr=3.44e-2 \
    --batch_size=6 \
    --run_name=unetr3d_vit_base_btcv_lr3.44e-2_mr125_10ke_pretrain_5000e \
    --pretrain=/path/to/your/pretrained/checkpoint.pth.tar
```

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce `batch_size` in config files
2. **MONAI version issues**: Try installing the latest stable MONAI: `pip install monai`
3. **Import errors**: Make sure you're in the repository root directory and `lib/` is in Python path
4. **Dataset path errors**: Verify all paths in config files are correct and accessible
5. **Wandb errors**: If Wandb fails, you can disable it with `--disable_wandb` flag

### GPU Selection:

To use a specific GPU, set `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py configs/mae3d_btcv_1gpu.yaml ...
```

Or modify the `gpu: 0` parameter in the config files.

## Additional Notes

- The project uses distributed training support (can be configured in YAML files)
- Checkpoint files will be saved in `{output_dir}/ckpts/`
- Wandb logs and visualizations will be stored in `{output_dir}/wandb/`
- Adjust `workers` parameter in config files based on your system's CPU cores

## Next Steps

1. Review the configuration files in `configs/` to understand all available options
2. Check the scripts in `scripts/` directory for example run commands
3. Modify hyperparameters in config files as needed for your use case
4. Monitor training progress via Wandb dashboard
