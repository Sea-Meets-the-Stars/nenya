
import numpy as np
import h5py
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.utils import resample
import os



def load_and_preprocess_mnist(output_filename:str='mnist_resampled.h5'):
    """"
    Load MNIST dataset, perform bootstrap resampling, and save to HDF5 format.
    
    Parameters:
    - target_samples: Total number of samples to generate (default: 200,000)
    - train_split: Number of training samples (default: 150,000)
    - valid_split: Number of validation samples (default: 50,000)
    """
    
    print("Loading MNIST dataset...")
    # Define transform to convert PIL Image to tensor and then to numpy
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load the original MNIST dataset using PyTorch
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, 
        download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, 
        download=True, transform=transform
    )
    
    # Convert to numpy arrays
    x_train_orig = train_dataset.data.numpy().astype(np.float32) / 255.0
    y_train_orig = train_dataset.targets.numpy()
    x_test_orig = test_dataset.data.numpy().astype(np.float32) / 255.0
    y_test_orig = test_dataset.targets.numpy()


    with h5py.File(output_filename, 'w') as f:
        # Create datasets with proper shapes: N_images, N_pix, N_pix
        f.create_dataset('train', data=x_train_orig, dtype=np.float32, compression='gzip')
        f.create_dataset('valid', data=x_test_orig, dtype=np.float32, compression='gzip')
        
        # Also save labels (optional, but often useful)
        f.create_dataset('train_labels', data=y_train_orig, dtype=np.int32)
        f.create_dataset('valid_labels', data=y_test_orig, dtype=np.int32)
        
        # Add metadata
        f.attrs['description'] = 'MNIST dataset with bootstrap resampling'
        f.attrs['total_samples'] = x_train_orig.shape[0] + x_test_orig.shape[0]
        #f.attrs['train_samples'] = train_split
        #f.attrs['valid_samples'] = valid_split
        f.attrs['image_shape'] = x_train_orig.shape[1:]
        f.attrs['data_type'] = 'float32'
        f.attrs['normalized'] = 'True (0-1 range)'
    
    print(f"Successfully saved HDF5 file with shape:")
    print(f"  - train: {x_train_orig.shape}")
    print(f"  - valid: {x_valid_orig.shape}")
    
    return output_filename
    

def orig_load_and_preprocess_mnist(target_samples=200000, train_split=150000, valid_split=50000):
    """"
    Load MNIST dataset, perform bootstrap resampling, and save to HDF5 format.
    
    Parameters:
    - target_samples: Total number of samples to generate (default: 200,000)
    - train_split: Number of training samples (default: 150,000)
    - valid_split: Number of validation samples (default: 50,000)
    """
    
    print("Loading MNIST dataset...")
    # Define transform to convert PIL Image to tensor and then to numpy
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load the original MNIST dataset using PyTorch
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, 
        download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, 
        download=True, transform=transform
    )
    
    # Convert to numpy arrays
    x_train_orig = train_dataset.data.numpy()
    y_train_orig = train_dataset.targets.numpy()
    x_test_orig = test_dataset.data.numpy()
    y_test_orig = test_dataset.targets.numpy()
    
    # Combine training and test sets to have all 70,000 samples
    x_all = np.concatenate([x_train_orig, x_test_orig], axis=0)
    y_all = np.concatenate([y_train_orig, y_test_orig], axis=0)
    
    print(f"Original MNIST dataset size: {x_all.shape[0]} samples")
    print(f"Image shape: {x_all.shape[1:]} pixels")
    
    # Check if we have enough samples or need bootstrap resampling
    if x_all.shape[0] >= target_samples:
        print(f"Sufficient samples available. Randomly selecting {target_samples} samples...")
        indices = np.random.choice(x_all.shape[0], size=target_samples, replace=False)
        x_resampled = x_all[indices]
        y_resampled = y_all[indices]
    else:
        print(f"Bootstrap resampling {target_samples} samples from {x_all.shape[0]} original samples...")
        # Bootstrap resampling with replacement
        x_resampled, y_resampled = resample(
            x_all, y_all, 
            n_samples=target_samples, 
            replace=True, 
            random_state=42
        )
    print(f"Resampled dataset size: {x_resampled.shape[0]} samples")
    
    # Convert to float32 and normalize to [0, 1] range
    x_resampled = x_resampled.astype(np.float32) / 255.0
    
    # Randomly shuffle the resampled data
    indices = np.random.permutation(target_samples)
    x_resampled = x_resampled[indices]
    y_resampled = y_resampled[indices]
    
    # Split into training and validation sets
    x_train = x_resampled[:train_split]
    y_train = y_resampled[:train_split]
    x_valid = x_resampled[train_split:train_split + valid_split]
    y_valid = y_resampled[train_split:train_split + valid_split]
    
    print(f"Training set: {x_train.shape[0]} samples")
    print(f"Validation set: {x_valid.shape[0]} samples")
    
    # Save to HDF5 file
    output_filename = 'mnist_resampled.h5'
    print(f"Saving to HDF5 file: {output_filename}")
    
    with h5py.File(output_filename, 'w') as f:
        # Create datasets with proper shapes: N_images, N_pix, N_pix
        f.create_dataset('train', data=x_train, dtype=np.float32, compression='gzip')
        f.create_dataset('valid', data=x_valid, dtype=np.float32, compression='gzip')
        
        # Also save labels (optional, but often useful)
        f.create_dataset('train_labels', data=y_train, dtype=np.int32)
        f.create_dataset('valid_labels', data=y_valid, dtype=np.int32)
        
        # Add metadata
        f.attrs['description'] = 'MNIST dataset with bootstrap resampling'
        f.attrs['total_samples'] = target_samples
        f.attrs['train_samples'] = train_split
        f.attrs['valid_samples'] = valid_split
        f.attrs['image_shape'] = x_train.shape[1:]
        f.attrs['data_type'] = 'float32'
        f.attrs['normalized'] = 'True (0-1 range)'
    
    print(f"Successfully saved HDF5 file with shape:")
    print(f"  - train: {x_train.shape}")
    print(f"  - valid: {x_valid.shape}")
    
    return output_filename


# Command line
if __name__ == "__main__":
    import argparse
    
    load_and_preprocess_mnist()