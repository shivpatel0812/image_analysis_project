"""
Test script to isolate the cuDNN issue and gather/index mismatch with PatchEmbed3D
Tests the exact-size kernel case (img_size == patch_size) and gather operation
"""
import torch
import sys
import os
import numpy as np


sys.path.append('lib')


try:
    from lib.networks.patch_embed_layers import PatchEmbed3D
    from timm.models.layers.helpers import to_3tuple
except ImportError as e:
    print(f"Error importing: {e}")
    print("\nThis might be due to:")
    print("1. Missing 'timm' package - install with: pip install timm==0.4.12")
    print("2. Wrong timm version - the code expects timm==0.4.12")
    print("\nYou can still test on Rivanna where dependencies are installed.")
    sys.exit(1)

def test_patch_embed(img_size, patch_size, in_chans=1, embed_dim=768, use_cuda=True):
    """Test PatchEmbed3D with given dimensions"""
    print(f"\n{'='*60}")
    print(f"Testing PatchEmbed3D:")
    print(f"  img_size: {img_size}")
    print(f"  patch_size: {patch_size}")
    print(f"  in_chans: {in_chans}")
    print(f"  embed_dim: {embed_dim}")
    print(f"  use_cuda: {use_cuda}")
    print(f"{'='*60}")
    
    # Create model
    try:
        model = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten=True,
            in_chan_last=False
        )
        print(f"✓ Model created successfully")
        print(f"  use_linear: {model.use_linear}")
        print(f"  proj type: {type(model.proj).__name__}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
   
    B = 2  
    L = 9  
    S = img_size[0] * img_size[1] * img_size[2] * in_chans
    
    print(f"\nInput shape: [B={B}, L={L}, S={S}]")
    print(f"  Expected S: {img_size[0]}*{img_size[1]}*{img_size[2]}*{in_chans} = {S}")
    
    x = torch.randn(B, L, S)
    
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        x = x.to(device)
        print(f"✓ Moved to CUDA: {device}")
    else:
        device = torch.device('cpu')
        print(f"✓ Using CPU")
    

    print(f"\nRunning forward pass...")
    try:
        with torch.no_grad():
            output = model(x)
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected shape: [{B*L}, {embed_dim}]")
        
        if output.shape == (B*L, embed_dim):
            print(f"  ✓ Output shape matches expected!")
        else:
            print(f"  ✗ Output shape mismatch!")
            return False
        
        return True
        
    except RuntimeError as e:
        error_msg = str(e)
        print(f"✗ Forward pass failed with RuntimeError:")
        print(f"  {error_msg}")
        
        if "CUDA" in error_msg or "device-side assert" in error_msg:
            print(f"\n  → This is the cuDNN error we're trying to fix!")
            print(f"  → The Conv3d with exact-size kernel is failing")
        elif "shape" in error_msg.lower() or "size" in error_msg.lower():
            print(f"\n  → This is a shape mismatch error")
        
        return False
    except Exception as e:
        print(f"✗ Forward pass failed with unexpected error:")
        print(f"  {type(e).__name__}: {e}")
        return False


def test_gather_operation(input_size, patch_size, in_chans=1, mask_ratio=0.75, use_cuda=True):
    """Test the gather operation that's failing in mae3d.py"""
    print(f"\n{'='*60}")
    print(f"Testing Gather Operation (index out of bounds issue):")
    print(f"  input_size: {input_size}")
    print(f"  patch_size: {patch_size}")
    print(f"  in_chans: {in_chans}")
    print(f"  mask_ratio: {mask_ratio}")
    print(f"{'='*60}")
    
    input_size = to_3tuple(input_size)
    patch_size = to_3tuple(patch_size)
    

    grid_size = []
    for in_sz, pa_sz in zip(input_size, patch_size):
        assert in_sz % pa_sz == 0, f"input size {in_sz} and patch size {pa_sz} are not divisible"
        gs = in_sz // pa_sz
        assert gs > 0, f"grid_size must be > 0, got {gs}"
        grid_size.append(gs)
    
    length = np.prod(grid_size)
    sel_length = int(length * (1 - mask_ratio))
    msk_length = length - sel_length
    
    print(f"\nCalculated dimensions:")
    print(f"  grid_size: {grid_size}")
    print(f"  total patches (length): {length}")
    print(f"  selected patches: {sel_length}")
    print(f"  masked patches: {msk_length}")
    

    batch_size = 2
    out_chans = in_chans * np.prod(patch_size)
    

    x = torch.randn(batch_size, length, out_chans)
    
    if use_cuda and torch.cuda.is_available():
        x = x.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"\nPatchified tensor shape: {x.shape}")
    print(f"  Expected: [B={batch_size}, length={length}, out_chans={out_chans}]")
    

    try:

        rand = torch.rand(batch_size, length).to(x.device)
        shuffle_indices = rand.argsort(dim=1)
        unshuffle_indices = shuffle_indices.argsort(dim=1)
        
        print(f"\nShuffle indices shape: {shuffle_indices.shape}")
        print(f"  Min index: {shuffle_indices.min().item()}, Max index: {shuffle_indices.max().item()}")
        print(f"  Expected range: [0, {length-1}]")
        

        if shuffle_indices.max() >= length or shuffle_indices.min() < 0:
            print(f"✗ Invalid shuffle indices!")
            return False
        

        print(f"\nTesting gather operation...")
        shuffled_x = x.gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, out_chans))
        sel_x = shuffled_x[:, :sel_length, :]
        
        print(f"✓ Gather operation successful!")
        print(f"  shuffled_x shape: {shuffled_x.shape}")
        print(f"  sel_x shape: {sel_x.shape}")
        
       
        print(f"\nTesting position embedding gather...")
       
        pos_embed = torch.randn(1, length, 768).to(x.device)
        print(f"  pos_embed shape: {pos_embed.shape}")
        print(f"  sel_indices shape: {shuffle_indices[:, :sel_length].shape}")
        
        sel_indices = shuffle_indices[:, :sel_length]
        sel_pos_embed = pos_embed.expand(batch_size, -1, -1).gather(
            dim=1, 
            index=sel_indices[:, :, None].expand(-1, -1, 768)
        )
        
        print(f"✓ Position embedding gather successful!")
        print(f"  sel_pos_embed shape: {sel_pos_embed.shape}")
        
        return True
        
    except RuntimeError as e:
        error_msg = str(e)
        print(f"✗ Gather operation failed:")
        print(f"  {error_msg}")
        
        if "index out of bounds" in error_msg.lower() or "Assertion" in error_msg:
            print(f"\n  → This is the index out of bounds error!")
            print(f"  → The position embedding size ({length}) doesn't match actual patches")
            print(f"  → Check that input_size in config matches actual cropped dimensions")
        
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        return False


def main():
    print("="*60)
    print("PatchEmbed3D cuDNN Issue & Gather Operation Test")
    print("="*60)
    

    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    

    print("\n" + "="*60)
    print("TEST 1: Exact-size kernel case (img_size == patch_size)")
    print("="*60)
    print("This should use Linear layer (if fix is applied)")
    result1 = test_patch_embed(
        img_size=(16, 16, 16),
        patch_size=(16, 16, 16),
        in_chans=1,
        embed_dim=768,
        use_cuda=cuda_available
    )
    

    print("\n" + "="*60)
    print("TEST 2: Normal case (img_size != patch_size)")
    print("="*60)
    print("This should use Conv3d (normal case)")
    result2 = test_patch_embed(
        img_size=(96, 96, 16),
        patch_size=(16, 16, 16),
        in_chans=1,
        embed_dim=768,
        use_cuda=cuda_available
    )
    

    print("\n" + "="*60)
    print("TEST 3: Gather operation (index out of bounds)")
    print("="*60)
    print("This tests the gather operation that fails in mae3d.py")
    result3 = test_gather_operation(
        input_size=[96, 96, 16],
        patch_size=16,
        in_chans=1,
        mask_ratio=0.75,
        use_cuda=cuda_available
    )
    

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test 1 (exact-size): {'✓ PASSED' if result1 else '✗ FAILED'}")
    print(f"Test 2 (normal):     {'✓ PASSED' if result2 else '✗ FAILED'}")
    print(f"Test 3 (gather):     {'✓ PASSED' if result3 else '✗ FAILED'}")
    
    if result1 and result2 and result3:
        print("\n✓ All tests passed! The fixes are working.")
    else:
        if not result3:
            print("\n✗ Test 3 failed - this is the actual error from training!")
            print("  The input_size config doesn't match actual image dimensions.")
            print("  Check that input_size: [96, 96, 16] matches cropped image size.")


if __name__ == '__main__':
    main()

