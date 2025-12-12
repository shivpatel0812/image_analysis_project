#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import SimpleITK as sitk

sys.path.append('lib/')
import lib.models as models
import lib.networks as networks


def load_mhd_image(mhd_path):
    image = sitk.ReadImage(str(mhd_path))
    array = sitk.GetArrayFromImage(image)
    return array.astype(np.float32)


def save_prediction(pred_array, output_path, reference_mhd=None):
    pred_image = sitk.GetImageFromArray(pred_array.astype(np.uint8))
    if reference_mhd is not None:
        ref_image = sitk.ReadImage(str(reference_mhd))
        pred_image.SetSpacing(ref_image.GetSpacing())
        pred_image.SetOrigin(ref_image.GetOrigin())
        pred_image.SetDirection(ref_image.GetDirection())
    sitk.WriteImage(pred_image, str(output_path))


def compute_dice(pred, target, num_classes):
    dice_scores = []
    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        intersection = np.sum(pred_c * target_c)
        union = np.sum(pred_c) + np.sum(target_c)
        if union == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / union
        dice_scores.append(dice)
    return dice_scores


def build_model(checkpoint_path, device, num_classes=2, in_chans=1):
    class Args:
        pass
    
    args = Args()
    args.arch = 'vit_base'
    args.enc_arch = 'ViTBackbone'
    args.dec_arch = 'UNETR_decoder'
    args.patch_size = 16
    args.in_chans = in_chans
    args.feature_size = 16
    args.encoder_embed_dim = 768
    args.encoder_depth = 12
    args.encoder_num_heads = 12
    args.drop_path = 0.1
    args.spatial_dim = 3
    args.num_classes = num_classes
    args.roi_x = 96
    args.roi_y = 96
    args.roi_z = 16
    
    model = getattr(models, 'UNETR3D')(
        encoder=getattr(networks, args.enc_arch),
        decoder=getattr(networks, args.dec_arch),
        args=args
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, args


def preprocess_2d_to_3d(image_2d, target_depth=16, target_size=(96, 96)):
    """Preprocess 2D image to 3D volume with correct size for model."""
    h, w = image_2d.shape
    
    # Resize to target size (96x96) using center crop or resize
    if h != target_size[0] or w != target_size[1]:
        # Center crop to target size
        start_h = max(0, (h - target_size[0]) // 2)
        start_w = max(0, (w - target_size[1]) // 2)
        image_2d = image_2d[start_h:start_h+target_size[0], start_w:start_w+target_size[1]]
        
        # If image is smaller than target, pad it
        if image_2d.shape[0] < target_size[0] or image_2d.shape[1] < target_size[1]:
            padded = np.zeros(target_size, dtype=image_2d.dtype)
            ph = (target_size[0] - image_2d.shape[0]) // 2
            pw = (target_size[1] - image_2d.shape[1]) // 2
            padded[ph:ph+image_2d.shape[0], pw:pw+image_2d.shape[1]] = image_2d
            image_2d = padded
    
    # Stack to create 3D volume (D, H, W)
    volume = np.stack([image_2d] * target_depth, axis=0)
    
    # Normalize to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return volume


def run_inference(model, image_tensor, device):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model(image_tensor)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='Segmentation_data 2')
    parser.add_argument('--output_dir', type=str, default='segmentation_results')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_predictions', action='store_true')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.checkpoint}")
    model, model_args = build_model(args.checkpoint, device, args.num_classes)
    
    test_brain_dir = Path(args.data_path) / 'Testing' / 'Brains'
    test_label_dir = Path(args.data_path) / 'Testing' / 'Labels'
    
    test_images = sorted(glob.glob(str(test_brain_dir / '*.mhd')))
    print(f"Found {len(test_images)} test images")
    
    all_dice_scores = []
    
    for idx, img_path in enumerate(test_images):
        img_path = Path(img_path)
        img_name = img_path.stem
        img_num = img_name.replace('mri_', '')
        
        label_path = test_label_dir / f'seg_{img_num}.mhd'
        
        image_2d = load_mhd_image(img_path)
        if len(image_2d.shape) == 3:
            image_2d = image_2d[0]
        
        volume = preprocess_2d_to_3d(image_2d, target_depth=model_args.roi_z, 
                                      target_size=(model_args.roi_x, model_args.roi_y))
        
        # volume shape is (D, H, W), model expects (B, C, H, W, D)
        # Transpose from (D, H, W) to (H, W, D)
        volume = np.transpose(volume, (1, 2, 0))
        # Now add batch and channel dims: (H, W, D) -> (1, 1, H, W, D)
        image_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
        image_tensor = image_tensor.to(device)
        
        output = run_inference(model, image_tensor, device)
        
        if args.num_classes > 1:
            pred = torch.argmax(output, dim=1)
        else:
            pred = (torch.sigmoid(output) > 0.5).long()
        
        # Output shape is (B, C, H, W, D), take middle slice along D
        pred_np = pred.cpu().numpy()[0]  # Remove batch dim: (H, W, D)
        pred_2d = pred_np[:, :, pred_np.shape[2] // 2]  # Take middle slice: (H, W)
        
        if label_path.exists():
            label_2d = load_mhd_image(label_path)
            if len(label_2d.shape) == 3:
                label_2d = label_2d[0]
            
            # Center crop label to match prediction size
            h, w = label_2d.shape
            target_h, target_w = pred_2d.shape
            start_h = max(0, (h - target_h) // 2)
            start_w = max(0, (w - target_w) // 2)
            label_cropped = label_2d[start_h:start_h+target_h, start_w:start_w+target_w]
            
            dice_scores = compute_dice(pred_2d, label_cropped, args.num_classes)
            all_dice_scores.append(dice_scores)
            
            if (idx + 1) % 50 == 0:
                print(f"[{idx+1}/{len(test_images)}] {img_name} - Dice: {np.mean(dice_scores):.4f}")
        
        if args.save_predictions:
            pred_output_path = Path(args.output_dir) / f'pred_{img_num}.mhd'
            save_prediction(pred_2d, pred_output_path, img_path)
    
    if all_dice_scores:
        all_dice_scores = np.array(all_dice_scores)
        mean_dice = np.mean(all_dice_scores)
        std_dice = np.std(all_dice_scores)
        
        print("\n" + "="*50)
        print("SEGMENTATION RESULTS")
        print("="*50)
        print(f"Total test images: {len(test_images)}")
        print(f"Mean Dice Score: {mean_dice:.4f} (+/- {std_dice:.4f})")
        
        for c in range(all_dice_scores.shape[1]):
            class_mean = np.mean(all_dice_scores[:, c])
            class_std = np.std(all_dice_scores[:, c])
            print(f"  Class {c+1} Dice: {class_mean:.4f} (+/- {class_std:.4f})")
        
        results_file = Path(args.output_dir) / 'results.txt'
        with open(results_file, 'w') as f:
            f.write(f"Total test images: {len(test_images)}\n")
            f.write(f"Mean Dice Score: {mean_dice:.4f} (+/- {std_dice:.4f})\n")
            for c in range(all_dice_scores.shape[1]):
                class_mean = np.mean(all_dice_scores[:, c])
                class_std = np.std(all_dice_scores[:, c])
                f.write(f"Class {c+1} Dice: {class_mean:.4f} (+/- {class_std:.4f})\n")
        print(f"\nResults saved to {results_file}")
    
    print("\nSegmentation complete!")


if __name__ == '__main__':
    main()
