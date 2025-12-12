#!/usr/bin/env python3
"""
Script to create transunet.json for BTCV dataset.
This script scans the BTCV directory and creates a JSON file in MONAI decathlon format.
"""

import os
import json
import glob
from pathlib import Path

def create_transunet_json(data_path, output_file='transunet.json'):
    """
    Create transunet.json file for BTCV dataset.
    
    Args:
        data_path: Path to the directory containing BTCV data
        output_file: Name of the output JSON file (will be saved in data_path)
    """
    data_path = Path(data_path)
    
    # Common BTCV directory structures
    possible_image_dirs = [
        data_path / 'averaged-training-images',
        data_path / 'averaged-training-images ',
        data_path / 'BTCV' / 'training' / 'img',
        data_path / 'BTCV' / 'training' / 'images',
        data_path / 'training' / 'img',
        data_path / 'training' / 'images',
        data_path / 'img',
        data_path / 'images',
    ]
    
    possible_label_dirs = [
        data_path / 'averaged-training-labels',
        data_path / 'averaged-training-labels ',
        data_path / 'BTCV' / 'training' / 'label',
        data_path / 'BTCV' / 'training' / 'labels',
        data_path / 'training' / 'label',
        data_path / 'training' / 'labels',
        data_path / 'label',
        data_path / 'labels',
    ]
    
    # Find image directory
    image_dir = None
    for img_dir in possible_image_dirs:
        if img_dir.exists():
            image_dir = img_dir
            break
    
    # If not found, search for any directory with "image" in name
    if image_dir is None:
        for item in data_path.iterdir():
            if item.is_dir() and ('image' in item.name.lower() or 'img' in item.name.lower()):
                # Check if it has .nii.gz files
                if list(item.glob('*.nii.gz')) or list(item.glob('*.nii')):
                    image_dir = item
                    print(f"Found image directory by search: {image_dir}")
                    break
    
    # Find label directory
    label_dir = None
    for lbl_dir in possible_label_dirs:
        if lbl_dir.exists():
            label_dir = lbl_dir
            break
    
    # If not found, search for any directory with "label" in name (including those with trailing spaces)
    if label_dir is None:
        for item in data_path.iterdir():
            item_name = item.name  # This preserves trailing spaces
            if item.is_dir() and ('label' in item_name.lower() or 'seg' in item_name.lower() or 'mask' in item_name.lower()):
                # Use the actual directory name as found (with space if present)
                test_path = data_path / item_name
                # Check if it has .nii.gz files
                nii_files = list(test_path.glob('*.nii.gz')) + list(test_path.glob('*.nii'))
                if nii_files:
                    label_dir = test_path
                    print(f"Found label directory by search: '{label_dir}' (name: '{item_name}')")
                    break
    
    if image_dir is None:
        print("ERROR: Could not find image directory. Tried:")
        for img_dir in possible_image_dirs:
            print(f"  - {img_dir}")
        print("\nPlease check your BTCV directory structure.")
        return False
    
    if label_dir is None:
        print("ERROR: Could not find label directory. Tried:")
        for lbl_dir in possible_label_dirs:
            print(f"  - {lbl_dir}")
        print("\nPlease check your BTCV directory structure.")
        return False
    
    print(f"Found image directory: {image_dir}")
    print(f"Found label directory: {label_dir}")
    
    # Find all image files
    image_files = sorted(glob.glob(str(image_dir / '*.nii.gz')) + 
                        glob.glob(str(image_dir / '*.nii')))
    
    if not image_files:
        print(f"ERROR: No image files found in {image_dir}")
        return False
    
    print(f"Found {len(image_files)} image files")
    
    # Create training list
    training_list = []
    
    # Get all label files for matching - handle trailing spaces in directory name
    # Use os.path.join to preserve trailing spaces in directory names
    import os
    label_pattern1 = os.path.join(str(label_dir), '*.nii.gz')
    label_pattern2 = os.path.join(str(label_dir), '*.nii')
    
    all_label_files = sorted(glob.glob(label_pattern1) + glob.glob(label_pattern2))
    
    print(f"Found {len(all_label_files)} label files")
    if all_label_files:
        sample = Path(all_label_files[0]).name
        if len(all_label_files) > 1:
            sample += f", {Path(all_label_files[1]).name}"
        print(f"Sample label files: {sample}")
    
    # Create a dictionary of label files by filename for fast lookup
    label_dict = {}
    for lbl_file in all_label_files:
        lbl_path = Path(lbl_file)
        label_dict[lbl_path.name] = lbl_path
    
    print(f"Label files in dictionary: {len(label_dict)}")
    if label_dict:
        sample_keys = list(label_dict.keys())[:5]
        print(f"Sample label filenames: {sample_keys}")
    
    # Show sample image filenames for comparison
    sample_images = [Path(f).name for f in image_files[:5]]
    print(f"Sample image filenames: {sample_images}")
    print(f"\nMatching images to labels...")
    
    # Create sets of base IDs (without _avg.nii.gz) for both images and labels
    image_ids = {Path(f).stem.replace('.nii', '').replace('.gz', '').replace('_avg', '') for f in image_files}
    label_ids = {Path(f).stem.replace('.nii', '').replace('.gz', '').replace('_avg', '') for f in all_label_files}
    
    # Find matching IDs
    matching_ids = image_ids.intersection(label_ids)
    print(f"Found {len(matching_ids)} matching IDs between {len(image_ids)} images and {len(label_ids)} labels")
    
    if len(matching_ids) == 0:
        print(f"\nNo matching IDs found!")
        print(f"Sample image IDs: {sorted(list(image_ids))[:10]}")
        print(f"Sample label IDs: {sorted(list(label_ids))[:10]}")
        print(f"\nThe images and labels appear to be from different sets.")
        print(f"You may need to check if you have the correct image/label pairs.")
        return False
    
    # Create a reverse lookup: ID -> full filename for labels
    label_by_id = {}
    for lbl_file in all_label_files:
        lbl_path = Path(lbl_file)
        lbl_id = lbl_path.stem.replace('.nii', '').replace('.gz', '').replace('_avg', '')
        label_by_id[lbl_id] = lbl_path
    
    # Create a reverse lookup: ID -> full filename for images
    image_by_id = {}
    for img_file in image_files:
        img_path = Path(img_file)
        img_id = img_path.stem.replace('.nii', '').replace('.gz', '').replace('_avg', '')
        image_by_id[img_id] = img_path
    
    # Create training list for matching IDs
    for match_id in sorted(matching_ids):
        img_file = image_by_id[match_id]
        label_file = label_by_id[match_id]
        
        # Get relative paths from data_path
        img_rel = os.path.relpath(str(img_file), data_path)
        label_rel = os.path.relpath(str(label_file), data_path)
        
        training_list.append({
            "image": img_rel,
            "label": label_rel
        })
    
    print(f"\nCreated {len(training_list)} matching image-label pairs")
    
    if not training_list:
        print("ERROR: No matching image-label pairs found!")
        print(f"\nThis means your image files and label files don't have matching names.")
        print(f"Images: {len(image_files)} files")
        print(f"Labels: {len(all_label_files)} files")
        print(f"\nFirst few image names: {[Path(f).name for f in image_files[:5]]}")
        print(f"First few label names: {[Path(f).name for f in all_label_files[:5]]}")
        print(f"\nYou may need to:")
        print(f"1. Check if image and label files have matching IDs")
        print(f"2. Rename files to match")
        print(f"3. Use only the subset that has matching pairs")
        return False
    
    # Create JSON structure (MONAI decathlon format)
    json_data = {
        "training": training_list,
        "validation": []  # Empty for now, can be filled later
    }
    
    # Save JSON file
    output_path = data_path / output_file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\nSuccessfully created {output_path}")
    print(f"Found {len(training_list)} training image-label pairs")
    print(f"\nFirst few entries:")
    for i, entry in enumerate(training_list[:3]):
        print(f"  {i+1}. Image: {entry['image']}, Label: {entry['label']}")
    
    return True

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_transunet_json.py <data_path> [output_file]")
        print("Example: python create_transunet_json.py /home/njt4xc/SelfMedMAE")
        sys.exit(1)
    
    data_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'transunet.json'
    
    success = create_transunet_json(data_path, output_file)
    sys.exit(0 if success else 1)

