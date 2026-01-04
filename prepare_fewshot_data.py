#!/usr/bin/env python
"""
Few-shot data preparation script for MR-to-CT synthesis.

This script:
1. Randomly samples 2% of MR images from the source MR directory
2. Copies all CT images from the source CT directory
3. Places them in the DSE training directory structure

Usage:
    python prepare_fewshot_data.py --mr_dir data/mr_images --ct_dir data/ct_images --output_dir data/mr2ct/train --mr_ratio 0.02

The output directory structure will be:
    output_dir/
    ├── mr/     # 2% randomly sampled MR images
    └── ct/     # All CT images
"""

import os
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    """Check if a file is an image based on extension."""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_files(directory):
    """Get all image files from a directory recursively."""
    image_files = []
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                image_files.append(path)
    return sorted(image_files)

def copy_images(src_files, dst_dir, prefix=""):
    """Copy image files to destination directory."""
    os.makedirs(dst_dir, exist_ok=True)
    copied_count = 0
    for src_path in src_files:
        filename = os.path.basename(src_path)
        if prefix:
            filename = f"{prefix}_{filename}"
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy2(src_path, dst_path)
        copied_count += 1
    return copied_count

def main():
    parser = argparse.ArgumentParser(
        description='Prepare few-shot data for MR-to-CT DSE training'
    )
    parser.add_argument(
        '--mr_dir', 
        type=str, 
        required=True,
        help='Source directory containing MR images'
    )
    parser.add_argument(
        '--ct_dir', 
        type=str, 
        required=True,
        help='Source directory containing CT images'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data/mr2ct/train',
        help='Output directory for DSE training data (default: data/mr2ct/train)'
    )
    parser.add_argument(
        '--mr_ratio', 
        type=float, 
        default=0.02,
        help='Ratio of MR images to sample (default: 0.02 = 2%%)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--min_mr_samples',
        type=int,
        default=1,
        help='Minimum number of MR samples to ensure (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Get all image files
    print(f"Scanning MR directory: {args.mr_dir}")
    mr_files = get_image_files(args.mr_dir)
    print(f"Found {len(mr_files)} MR images")
    
    print(f"Scanning CT directory: {args.ct_dir}")
    ct_files = get_image_files(args.ct_dir)
    print(f"Found {len(ct_files)} CT images")
    
    if len(mr_files) == 0:
        print("Error: No MR images found!")
        return 1
    
    if len(ct_files) == 0:
        print("Error: No CT images found!")
        return 1
    
    # Calculate number of MR samples
    num_mr_samples = max(args.min_mr_samples, int(len(mr_files) * args.mr_ratio))
    num_mr_samples = min(num_mr_samples, len(mr_files))  # Don't exceed total
    
    print(f"\nSampling {num_mr_samples} MR images ({args.mr_ratio*100:.1f}% of {len(mr_files)})")
    
    # Randomly sample MR images
    sampled_mr_files = random.sample(mr_files, num_mr_samples)
    
    # Create output directories
    mr_output_dir = os.path.join(args.output_dir, 'mr')
    ct_output_dir = os.path.join(args.output_dir, 'ct')
    
    # Clear existing files in output directories if they exist
    for output_dir in [mr_output_dir, ct_output_dir]:
        if os.path.exists(output_dir):
            print(f"Clearing existing directory: {output_dir}")
            shutil.rmtree(output_dir)
    
    # Copy files
    print(f"\nCopying {len(sampled_mr_files)} MR images to {mr_output_dir}")
    mr_copied = copy_images(sampled_mr_files, mr_output_dir)
    
    print(f"Copying {len(ct_files)} CT images to {ct_output_dir}")
    ct_copied = copy_images(ct_files, ct_output_dir)
    
    print(f"\n=== Summary ===")
    print(f"MR images copied: {mr_copied} ({args.mr_ratio*100:.1f}% of original)")
    print(f"CT images copied: {ct_copied} (100% of original)")
    print(f"Output directory: {args.output_dir}")
    print(f"\nFew-shot data preparation complete!")
    print(f"You can now train DSE with: python run_train_dse.py")
    
    return 0

if __name__ == "__main__":
    exit(main())
