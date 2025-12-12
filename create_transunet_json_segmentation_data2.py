#!/usr/bin/env python3
"""
Script to create transunet.json for Segmentation_data 2 dataset.
Handles .mhd files with mri_*/seg_* naming pattern.
"""

import os
import json
import glob
from pathlib import Path

def create_transunet_json_segmentation_data2(data_path, output_file='transunet.json'):
    """
    Create transunet.json file for Segmentation_data 2 dataset.
    
    Args:
        data_path: Path to the directory containing Segmentation_data 2
        output_file: Name of the output JSON file (will be saved in data_path)
    """
    data_path = Path(data_path)
    
   
    image_dir = data_path / 'Segmentation_data 2' / 'Training' / 'Brains'
    label_dir = data_path / 'Segmentation_data 2' / 'Training' / 'Labels'
    

    if not image_dir.exists():
        print(f"ERROR: Image directory not found: {image_dir}")
        return False
    
    if not label_dir.exists():
        print(f"ERROR: Label directory not found: {label_dir}")
        return False
    
    print(f"Found image directory: {image_dir}")
    print(f"Found label directory: {label_dir}")
    

    image_files = sorted(glob.glob(str(image_dir / '*.mhd')))
    
    if not image_files:
        print(f"ERROR: No .mhd image files found in {image_dir}")
        return False
    
    print(f"Found {len(image_files)} image files")
    

    training_list = []
    
    for img_file in image_files:
        img_path = Path(img_file)
  
        img_name = img_path.stem  
        
     
        if img_name.startswith('mri_'):
            number = img_name.replace('mri_', '')
     
            label_file = label_dir / f"seg_{number}.mhd"
            
            if not label_file.exists():
                print(f"WARNING: No label found for {img_path.name}, skipping...")
                continue
            
     
            img_rel = os.path.relpath(img_file, data_path)
            label_rel = os.path.relpath(str(label_file), data_path)
            
            training_list.append({
                "image": img_rel,
                "label": label_rel
            })
        else:
            print(f"WARNING: Unexpected image filename format: {img_path.name}, skipping...")
            continue
    
    if not training_list:
        print("ERROR: No matching image-label pairs found!")
        return False
    

    json_data = {
        "training": training_list,
        "validation": [] 
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
        print("Usage: python create_transunet_json_segmentation_data2.py <data_path> [output_file]")
        print("Example: python create_transunet_json_segmentation_data2.py /home/njt4xc/SelfMedMAE")
        sys.exit(1)
    
    data_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'transunet.json'
    
    success = create_transunet_json_segmentation_data2(data_path, output_file)
    sys.exit(0 if success else 1)

