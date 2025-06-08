#!/usr/bin/env python3

"""
Dataset Processing Script for ImageNet
Processes datasets according to specified requirements:
- 200,000 random images (or max available)
- Split: 150,000 training, 50,000 validation
- Convert to float32 and save to HDF5
"""
import os
import sys
import h5py
import numpy as np
import requests
import gzip
import tarfile
import zipfile
from PIL import Image
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle

def download_file(url, filepath, description="Downloading"):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def process_images_from_directory(image_dir, n_samples=200000):
    """Process images from a directory (for ImageNet or similar datasets)"""
    print(f"Processing images from: {image_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(Path(root) / file)
    
    print(f"Found {len(image_files)} image files")
    
    if len(image_files) == 0:
        raise ValueError("No image files found in the specified directory")
    
    # Sample random images
    n_available = len(image_files)
    n_samples = min(n_samples, n_available)
    print(f"Sampling {n_samples} images")
    
    sampled_files = random.sample(image_files, n_samples)
    
    # Process images
    processed_images = []
    
    for img_path in tqdm(sampled_files, desc="Processing images"):
        try:
            # Load image
            with Image.open(img_path) as img:
                # Convert to grayscale
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Resize to consistent size (you may want to adjust this)
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img, dtype=np.float32) / 255.0
                processed_images.append(img_array)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    processed_images = np.array(processed_images)
    print(f"Successfully processed {len(processed_images)} images")
    
    return processed_images

def process_imagenet(imagenet_path, output_dir, n_samples=200000):
    """Process ImageNet dataset"""
    if not imagenet_path or not Path(imagenet_path).exists():
        print("ImageNet path not provided or doesn't exist.")
        print("Skipping ImageNet processing.")
        return None
    
    print("Processing ImageNet dataset...")
    
    # Process images from directory
    all_images = process_images_from_directory(imagenet_path, n_samples)
    
    if len(all_images) == 0:
        print("No images were successfully processed from ImageNet")
        return None
    
    # Split into train/valid
    n_total = len(all_images)
    n_train = 150000 if n_total >= 200000 else int(0.75 * n_total)
    n_valid = n_total - n_train
    
    # Shuffle the data
    indices = np.random.permutation(n_total)
    train_images = all_images[indices[:n_train]]
    valid_images = all_images[indices[n_train:n_train + n_valid]]
    
    print(f"Train images: {len(train_images)}, Valid images: {len(valid_images)}")
    
    # Save to HDF5
    output_path = Path(output_dir) / "imagenet_processed.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('train', data=train_images, compression='gzip')
        f.create_dataset('valid', data=valid_images, compression='gzip')
        
        # Add metadata
        f.attrs['dataset'] = 'ImageNet'
        f.attrs['n_train'] = n_train
        f.attrs['n_valid'] = n_valid
        f.attrs['image_shape'] = train_images.shape[1:]
    
    print(f"ImageNet processed and saved to: {output_path}")
    return output_path

def verify_hdf5_file(filepath):
    """Verify the created HDF5 file"""
    print(f"\nVerifying HDF5 file: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        print(f"Dataset: {f.attrs.get('dataset', 'Unknown')}")
        print(f"Keys in file: {list(f.keys())}")
        
        for key in ['train', 'valid']:
            if key in f:
                data = f[key]
                print(f"{key}: shape={data.shape}, dtype={data.dtype}")
                print(f"  Min value: {data[:].min():.4f}")
                print(f"  Max value: {data[:].max():.4f}")
                print(f"  Mean value: {data[:].mean():.4f}")
            else:
                print(f"{key}: Not found")

def main():
    parser = argparse.ArgumentParser(description='Process ImageNet datasets')
    parser.add_argument('--data-dir', default='./data', help='Directory to store raw data')
    parser.add_argument('--output-dir', default='./processed', help='Directory to store processed data')
    parser.add_argument('--imagenet-path', default='./imagenet_dataset', help='Path to ImageNet dataset directory')
    parser.add_argument('--n-samples', type=int, default=200000, help='Number of samples to process')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print("Dataset Processing Script")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target samples: {args.n_samples}")
    print(f"Random seed: {args.seed}")
    print("=" * 50)
    
    # Create directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    
    # Process ImageNet
    try:
        imagenet_file = process_imagenet(args.imagenet_path, args.output_dir, args.n_samples)
        if imagenet_file:
            processed_files.append(imagenet_file)
            verify_hdf5_file(imagenet_file)
    except Exception as e:
        print(f"Error processing ImageNet: {e}")
    
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print("Processed files:")
    for file in processed_files:
        print(f"  - {file}")
    
    if not processed_files:
        print("No files were successfully processed.")
    
    print("\nTo use the processed data:")
    print("```python")
    print("import h5py")
    print("with h5py.File('processed/imagenet_processed.h5', 'r') as f:")
    print("    train_data = f['train'][:]")
    print("    valid_data = f['valid'][:]")
    print("```")

if __name__ == "__main__":
    main()