#!/usr/bin/env python3
"""
Simple script to create transunet.json for BTCV dataset.
For MAE pre-training, labels aren't needed, so we use all images with placeholder labels.
"""

import os
import json
import glob
from pathlib import Path

def create_transunet_json_simple(data_path, output_file='transunet.json'):
    """
    Create transunet.json file using ALL images.
    For MAE pre-training, labels aren't actually used, so we use placeholder labels.
    """
    data_path = Path(data_path)
    
    # Find image directory
    possible_image_dirs = [
        data_path / 'averaged-training-images',
        data_path / 'averaged-training-images ',
        data_path / 'BTCV' / 'training' / 'img',
        data_path / 'training' / 'images',
    ]
    
    image_dir = None
    for img_dir in possible_image_dirs:
        if img_dir.exists():
            image_dir = img_dir
            break
    
    # Auto-detect if not found
    if image_dir is None:
        for item in data_path.iterdir():
            if item.is_dir() and ('image' in item.name.lower() or 'img' in item.name.lower()):
                if list(item.glob('*.nii.gz')) or list(item.glob('*.nii')):
                    image_dir = item
                    break
    
    if image_dir is None:
        print("ERROR: Could not find image directory")
        return False
    
    print(f"Found image directory: {image_dir}")
    
    # Find all image files
    image_files = sorted(glob.glob(str(image_dir / '*.nii.gz')) + 
                        glob.glob(str(image_dir / '*.nii')))
    
    if not image_files:
        print(f"ERROR: No image files found")
        return False
    
    print(f"Found {len(image_files)} image files")
    
    # Create training list - use same image as label (dummy, since MAE doesn't use labels)
    training_list = []
    
    for img_file in image_files:
        img_rel = os.path.relpath(img_file, data_path)
        
        # For MAE pre-training, labels aren't used, so we can use the same image
        # or create a placeholder. Using same image is simplest.
        training_list.append({
            "image": img_rel,
            "label": img_rel  # Same file as label (won't be used by MAE)
        })
    
    # Create JSON structure
    json_data = {
        "training": training_list,
        "validation": []
    }
    
    # Save JSON file
    output_path = data_path / output_file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\nSuccessfully created {output_path}")
    print(f"Created {len(training_list)} entries (using all images)")
    print(f"\nNote: Labels are set to same as images since MAE pre-training doesn't use labels.")
    print(f"First few entries:")
    for i, entry in enumerate(training_list[:3]):
        print(f"  {i+1}. Image: {entry['image']}")
    
    return True

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_transunet_json_simple.py <data_path>")
        print("Example: python create_transunet_json_simple.py /home/njt4xc/SelfMedMAE")
        sys.exit(1)
    
    data_path = sys.argv[1]
    success = create_transunet_json_simple(data_path)
    sys.exit(0 if success else 1)

