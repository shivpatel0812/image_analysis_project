# 2D Medical Image Segmentation Setup Guide

This guide provides instructions for two scenarios:

- **Option A**: Using the existing shared environment at `/standard/mlia/MLIA_Team14/SelfMedMAE`
- **Option B**: Setting up a fresh personal environment

---

## Option A: Using Existing Shared Environment

If you have access to the shared environment at `/standard/mlia/MLIA_Team14/SelfMedMAE`:

### Step 1: SSH to Rivanna

```bash
ssh user@login.hpc.virginia.edu

```

### Step 2: Setup Python Environment

```bash
cd /standard/mlia/MLIA_Team14/SelfMedMAE
python3 -m venv venv_selfmedmae
source venv_selfmedmae/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai timm wandb SimpleITK omegaconf scikit-learn numpy tqdm
pip install -r requirements.txt
```

### Step 3: Request GPU Node

```bash
export USER=njt4xc
salloc --partition=gpu-mig \
       --gres=gpu:1g.10gb:1 \
       --time=48:00:00 \
       --mem=32G \
       --cpus-per-task=4
```

Once allocated, connect to the compute node:

```bash
srun --cpu-bind=none --pty bash
```

### Step 4: Prepare Data

```bash
cd /standard/mlia/MLIA_Team14/SelfMedMAE
source venv_selfmedmae/bin/activate
python create_list_files.py /standard/mlia/MLIA_Team14/SelfMedMAE
```

This creates `train_list.txt` and `val_list.txt` for MAE pre-training.

### Step 5: MAE Pre-training (2D)

```bash
cd /standard/mlia/MLIA_Team14/SelfMedMAE
source venv_selfmedmae/bin/activate
module load cuda
python main.py configs/mae2d_btcv_1gpu.yaml
```

**Output:** Checkpoints saved to `/standard/mlia/MLIA_Team14/outputs/mae2d_vit_base_btcv/ckpts/`

### Step 6: UNETR Fine-tuning (With Pre-training)

```bash
cd /standard/mlia/MLIA_Team14/SelfMedMAE
source venv_selfmedmae/bin/activate
module load cuda
python main.py configs/unetr2d_btcv_true2d.yaml
```

**Note:** Uses pre-trained weights from `checkpoint_0799.pth.tar` (or latest available). The config file already points to `/standard/mlia/MLIA_Team14/outputs/mae2d_vit_base_btcv/ckpts/checkpoint_0799.pth.tar` If needed please change the file path of the checkpoint in that file if the weight is located in a different folder.

### Step 7: UNETR Baseline (Without Pre-training)

```bash
cd /standard/mlia/MLIA_Team14/SelfMedMAE
source venv_selfmedmae/bin/activate
module load cuda
python main.py configs/unetr2d_btcv_true2d_scratch.yaml
```

**Output:** Checkpoints saved to `/standard/mlia/MLIA_Team14/outputs/UNETR2D_vit_base_btcv_scratch/ckpts/`

### Step 8: Segmentation Inference

#### With Pre-trained UNETR

```bash
cd /standard/mlia/MLIA_Team14/SelfMedMAE
source venv_selfmedmae/bin/activate
module load cuda
python run_segmentation_true2d.py \
  --checkpoint /standard/mlia/MLIA_Team14/outputs/UNETR2D_vit_base_btcv/ckpts/checkpoint_0199.pth.tar \
  --data_path /standard/mlia/MLIA_Team14/SelfMedMAE/Segmentation_data\ 2 \
  --output_dir /standard/mlia/MLIA_Team14/segmentation_results_unetr2d_pretrained \
  --num_classes 14 \
  --gpu 0 \
  --save_predictions
```

**Note:** Update `checkpoint_0199.pth.tar` to the actual checkpoint you want to use (e.g., `checkpoint_0499.pth.tar`, `checkpoint_0799.pth.tar`, etc., along wiht change the file path as needed )

#### Baseline UNETR (Scratch)

```bash
cd /standard/mlia/MLIA_Team14/SelfMedMAE
source venv_selfmedmae/bin/activate
module load cuda
python run_segmentation_true2d.py \
  --checkpoint /standard/mlia/MLIA_Team14/outputs/UNETR2D_vit_base_btcv_scratch/ckpts/checkpoint_0199.pth.tar \
  --data_path /standard/mlia/MLIA_Team14/SelfMedMAE/Segmentation_data\ 2 \
  --output_dir /standard/mlia/MLIA_Team14/segmentation_results_unetr2d_scratch \
  --num_classes 14 \
  --gpu 0 \
  --save_predictions
```

**Results:** Saved to `--output_dir` with Dice scores in `results.txt`

---

## Option B: Fresh Personal Environment Setup

If you want to set up your own environment from scratch:

### Step 1: Transfer Code to Rivanna

From your local machine:

```bash

scp -r /path/to/your/local/SelfMedMAE user@login.hpc.virginia.edu:~/SelfMedMAE
```

### Step 2: Transfer Data to Rivanna

From your local machine:

```bash
#
scp -r /path/to/your/Segmentation_data\ 2 user@login.hpc.virginia.edu:~/SelfMedMAE/
```

### Step 3: SSH and Setup Environment

```bash
ssh user@login.hpc.virginia.edu
cd ~/SelfMedMAE
python3 -m venv venv_selfmedmae
source venv_selfmedmae/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai timm wandb SimpleITK omegaconf scikit-learn numpy tqdm
pip install -r requirements.txt
```

### Step 4: Update Config Files

Update the following config files to use your paths:

**`configs/mae2d_btcv_1gpu.yaml`:**

```yaml
data_path: ~/SelfMedMAE
tr_listfile: ~/SelfMedMAE/train_list.txt
va_listfile: ~/SelfMedMAE/val_list.txt
output_dir: ~/outputs/${run_name}
```

**`configs/unetr2d_btcv_true2d.yaml`:**

```yaml
data_path: ~/SelfMedMAE
output_dir: ~/outputs/${run_name}
pretrain: ~/outputs/mae2d_vit_base_btcv/ckpts/checkpoint_0799.pth.tar
```

**`configs/unetr2d_btcv_true2d_scratch.yaml`:**

```yaml
data_path: ~/SelfMedMAE
output_dir: ~/outputs/${run_name}
```

### Step 5: Request GPU Node

```bash
salloc --partition=gpu-mig \
       --gres=gpu:1g.10gb:1 \
       --time=48:00:00 \
       --mem=32G \
       --cpus-per-task=4
```

Once allocated, connect to the compute node:

```bash
srun --cpu-bind=none --pty bash
```

### Step 6: Prepare Data

```bash
cd ~/SelfMedMAE
source venv_selfmedmae/bin/activate
python create_list_files.py ~/SelfMedMAE
```

This creates `train_list.txt` and `val_list.txt` for MAE pre-training.

### Step 7: MAE Pre-training (2D)

```bash
cd ~/SelfMedMAE
source venv_selfmedmae/bin/activate
module load cuda
python main.py configs/mae2d_btcv_1gpu.yaml
```

**Output:** Checkpoints saved to `~/outputs/mae2d_vit_base_btcv/ckpts/`

### Step 8: UNETR Fine-tuning (With Pre-training)

```bash
cd ~/SelfMedMAE
source venv_selfmedmae/bin/activate
module load cuda
python main.py configs/unetr2d_btcv_true2d.yaml
```

**Note:** Make sure the `pretrain:` path in the config file points to your MAE checkpoint.

### Step 9: UNETR Baseline (Without Pre-training)

```bash
cd ~/SelfMedMAE
source venv_selfmedmae/bin/activate
module load cuda
python main.py configs/unetr2d_btcv_true2d_scratch.yaml
```

**Output:** Checkpoints saved to `~/outputs/UNETR2D_vit_base_btcv_scratch/ckpts/`

### Step 10: Segmentation Inference

#### With Pre-trained UNETR

```bash
cd ~/SelfMedMAE
source venv_selfmedmae/bin/activate
module load cuda
python run_segmentation_true2d.py \
  --checkpoint ~/outputs/UNETR2D_vit_base_btcv/ckpts/checkpoint_XXXX.pth.tar \
  --data_path ~/SelfMedMAE/Segmentation_data\ 2 \
  --output_dir ~/segmentation_results_unetr2d_pretrained \
  --num_classes 14 \
  --gpu 0 \
  --save_predictions
```

**Note:** Replace `checkpoint_XXXX.pth.tar` with your actual checkpoint filename (e.g., `checkpoint_0199.pth.tar`, `checkpoint_0499.pth.tar`, etc.)

#### Baseline UNETR (Scratch)

```bash
cd ~/SelfMedMAE
source venv_selfmedmae/bin/activate
module load cuda
python run_segmentation_true2d.py \
  --checkpoint ~/outputs/UNETR2D_vit_base_btcv_scratch/ckpts/checkpoint_XXXX.pth.tar \
  --data_path ~/SelfMedMAE/Segmentation_data\ 2 \
  --output_dir ~/segmentation_results_unetr2d_scratch \
  --num_classes 14 \
  --gpu 0 \
  --save_predictions
```

**Results:** Saved to `--output_dir` with Dice scores in `results.txt`
