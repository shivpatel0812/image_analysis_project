#!/usr/bin/env python3
"""
Script to check checkpoint contents and verify resume functionality
"""
import torch
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python check_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]

if not os.path.exists(checkpoint_path):
    print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
    sys.exit(1)

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*60)
print("CHECKPOINT CONTENTS:")
print("="*60)
print(f"Keys in checkpoint: {list(checkpoint.keys())}")

if 'epoch' in checkpoint:
    print(f"\n✓ Epoch stored in checkpoint: {checkpoint['epoch']}")
    print(f"  → Training should resume from epoch {checkpoint['epoch']}")
else:
    print("\n✗ ERROR: No 'epoch' key in checkpoint!")

if 'state_dict' in checkpoint:
    print(f"\n✓ State dict found: {len(checkpoint['state_dict'])} keys")
    # Check for encoder keys
    encoder_keys = [k for k in checkpoint['state_dict'].keys() if 'encoder' in k.lower() or not k.startswith('decoder')]
    print(f"  → Found {len(encoder_keys)} encoder-related keys")
else:
    print("\n✗ ERROR: No 'state_dict' key in checkpoint!")

if 'optimizer' in checkpoint:
    print(f"\n✓ Optimizer state found")
else:
    print("\n✗ WARNING: No 'optimizer' key in checkpoint (resume may not restore optimizer state)")

if 'scaler' in checkpoint:
    print(f"\n✓ Scaler state found")
else:
    print("\n✗ WARNING: No 'scaler' key in checkpoint")

print("\n" + "="*60)
print("RESUME VERIFICATION:")
print("="*60)

# Check what the filename suggests
filename = os.path.basename(checkpoint_path)
if 'checkpoint_' in filename:
    # Extract epoch from filename (e.g., checkpoint_0044.pth.tar -> 44)
    try:
        epoch_from_filename = int(filename.split('_')[1].split('.')[0])
        print(f"Filename suggests epoch: {epoch_from_filename}")
        if 'epoch' in checkpoint:
            if checkpoint['epoch'] == epoch_from_filename + 1:
                print(f"✓ Epoch in checkpoint ({checkpoint['epoch']}) matches filename ({epoch_from_filename + 1})")
            else:
                print(f"✗ MISMATCH: Epoch in checkpoint ({checkpoint['epoch']}) != filename ({epoch_from_filename + 1})")
    except:
        pass

print("\n" + "="*60)
print("RECOMMENDATION:")
print("="*60)
if 'epoch' in checkpoint:
    print(f"To resume from this checkpoint, use:")
    print(f"  python main.py configs/unetr_btcv_1gpu.yaml \\")
    print(f"      --resume={checkpoint_path}")
    print(f"\nExpected behavior:")
    print(f"  - Training should start from epoch {checkpoint['epoch']}")
    print(f"  - You should see: '=> loaded checkpoint ... (epoch {checkpoint['epoch']})'")
else:
    print("This checkpoint may not be suitable for resuming training.")
    print("It may be a pre-trained model checkpoint, not a training checkpoint.")

