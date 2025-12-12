#!/usr/bin/env python3
import os
import glob
import random
from pathlib import Path

def create_list_files(data_path, train_ratio=0.8):
    data_path = Path(data_path)
    
    image_dir = data_path / "Segmentation_data 2" / "Training" / "Brains"
    
    if not image_dir.exists():
        print(f"ERROR: Directory not found: {image_dir}")
        return False
    
    image_files = sorted(glob.glob(str(image_dir / "*.mhd")))
    
    if not image_files:
        print(f"ERROR: No .mhd files found in {image_dir}")
        return False
    
    print(f"Found {len(image_files)} images")
    
    rel_paths = [os.path.relpath(f, data_path) for f in image_files]
    
    random.seed(42)
    random.shuffle(rel_paths)
    
    split_idx = int(len(rel_paths) * train_ratio)
    train_files = rel_paths[:split_idx]
    val_files = rel_paths[split_idx:]
    
    train_list_path = data_path / "train_list.txt"
    val_list_path = data_path / "val_list.txt"
    
    with open(train_list_path, 'w') as f:
        for path in train_files:
            f.write(f"{path}\n")
    
    with open(val_list_path, 'w') as f:
        for path in val_files:
            f.write(f"{path}\n")
    
    print(f"Created {train_list_path} with {len(train_files)} images")
    print(f"Created {val_list_path} with {len(val_files)} images")
    
    return True

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_list_files.py <data_path>")
        print("Example: python create_list_files.py /home/njt4xc/SelfMedMAE")
        sys.exit(1)
    
    data_path = sys.argv[1]
    success = create_list_files(data_path)
    sys.exit(0 if success else 1)

