#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create training and testing splits from wildlife datasets.
Ensures no overlap between training and testing data at the identity level.

Usage:
    python create_data.py --dataset LeopardID2022 --test_ratio 0.2 --output_dir ./data --min_images_per_id 2
"""

import os
import pandas as pd
import numpy as np
import shutil
import argparse
from sklearn.model_selection import train_test_split
from wildlife_datasets.datasets import MacaqueFaces, Cows2021v2, LeopardID2022, HyenaID2022


def parse_args():
    parser = argparse.ArgumentParser(description='Create train/test splits from wildlife datasets')
    parser.add_argument('--dataset', default='MacaqueFaces', type=str, 
                       choices=['LeopardID2022', 'MacaqueFaces', 'Cows2021v2', 'HyenaID2022'],
                       help='Dataset name')
    parser.add_argument('--test_ratio', default=0.2, type=float,
                       help='Percentage of identities to use for testing (0.0-1.0)')
    parser.add_argument('--output_dir', default='./data', type=str,
                       help='Output directory to save train.csv and test.csv')
    parser.add_argument('--min_images_per_id', default=2, type=int,
                       help='Minimum number of images per identity to include')
    parser.add_argument('--random_seed', default=42, type=int,
                       help='Random seed for reproducible splits')
    parser.add_argument('--dataset_path', default=None, type=str,
                       help='Custom path to dataset (if not using default)')
    
    return parser.parse_args()


def load_dataset(dataset_name, dataset_path=None):
    """Load the specified wildlife dataset"""
    print(f"Loading {dataset_name}...")
    
    if dataset_name == "LeopardID2022":
        path = dataset_path or '/home/wellvw12/leopard'
        dataset = LeopardID2022(path)
        df = dataset.df
    elif dataset_name == "MacaqueFaces":
        dataset = MacaqueFaces('/home/wellvw12/fedwild/MacaqueFaces')
        df = dataset.df
    elif dataset_name == "Cows2021v2":
        dataset = Cows2021v2()
        df = dataset.df
    elif dataset_name == "HyenaID2022":
        path = dataset_path or '/home/wellvw12/hyenaid2022'
        dataset = HyenaID2022(path)
        df = dataset.df
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    print(f"Loaded {len(df)} images")
    return df


def filter_by_min_images(df, min_images_per_id):
    """Filter out identities with fewer than min_images_per_id images"""
    print(f"Filtering identities with at least {min_images_per_id} images...")
    
    # Count images per identity
    id_counts = df['identity'].value_counts()
    valid_ids = id_counts[id_counts >= min_images_per_id].index
    
    # Filter dataframe
    filtered_df = df[df['identity'].isin(valid_ids)].copy()
    
    print(f"Original: {len(df['identity'].unique())} identities")
    print(f"Filtered: {len(filtered_df['identity'].unique())} identities")
    print(f"Remaining images: {len(filtered_df)}")
    
    return filtered_df


def split_by_identity(df, test_ratio, random_seed):
    """Split dataset by identity to ensure no overlap between train/test"""
    print(f"Splitting dataset with {test_ratio:.1%} of identities for testing...")
    
    # Get unique identities
    unique_ids = df['identity'].unique()
    print(f"Total unique identities: {len(unique_ids)}")
    
    # Split identities into train/test
    train_ids, test_ids = train_test_split(
        unique_ids, 
        test_size=test_ratio, 
        random_state=random_seed,
        shuffle=True
    )
    
    # Split dataframe based on identity splits
    train_df = df[df['identity'].isin(train_ids)].copy()
    test_df = df[df['identity'].isin(test_ids)].copy()
    
    print(f"Train identities: {len(train_ids)}")
    print(f"Test identities: {len(test_ids)}")
    print(f"Train images: {len(train_df)}")
    print(f"Test images: {len(test_df)}")
    
    # Verify no overlap
    train_identity_set = set(train_df['identity'].unique())
    test_identity_set = set(test_df['identity'].unique())
    overlap = train_identity_set.intersection(test_identity_set)
    
    if overlap:
        raise ValueError(f"Found overlap in identities: {overlap}")
    else:
        print("✓ No identity overlap between train and test sets")
    
    return train_df, test_df


def relabel_identities(train_df, test_df):
    """Relabel identities to have consecutive integer labels starting from 0"""
    print("Relabeling identities...")
    
    # Create label mappings
    train_ids = sorted(train_df['identity'].unique())
    test_ids = sorted(test_df['identity'].unique())
    
    train_id_to_label = {id_val: idx for idx, id_val in enumerate(train_ids)}
    test_id_to_label = {id_val: idx for idx, id_val in enumerate(test_ids)}
    
    # Apply mappings
    train_df_labeled = train_df.copy()
    test_df_labeled = test_df.copy()
    
    train_df_labeled['identity'] = train_df_labeled['identity'].map(train_id_to_label)
    test_df_labeled['identity'] = test_df_labeled['identity'].map(test_id_to_label)
    
    print(f"Train labels: 0 to {len(train_ids)-1}")
    print(f"Test labels: 0 to {len(test_ids)-1}")
    
    return train_df_labeled, test_df_labeled


def save_splits(train_df, test_df, output_dir):
    """Save train and test splits to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    # Save CSV files
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved train split to: {train_path}")
    print(f"Saved test split to: {test_path}")
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    print(f"Train: {len(train_df)} images, {len(train_df['identity'].unique())} identities")
    print(f"Test: {len(test_df)} images, {len(test_df['identity'].unique())} identities")
    print(f"Total: {len(train_df) + len(test_df)} images")
    
    # Print per-identity statistics
    train_counts = train_df['identity'].value_counts()
    test_counts = test_df['identity'].value_counts()
    
    print(f"\nTrain images per identity: min={train_counts.min()}, max={train_counts.max()}, mean={train_counts.mean():.1f}")
    print(f"Test images per identity: min={test_counts.min()}, max={test_counts.max()}, mean={test_counts.mean():.1f}")


def main():
    args = parse_args()
    
    print("=== Wildlife Dataset Splitter ===")
    print(f"Dataset: {args.dataset}")
    print(f"Test ratio: {args.test_ratio:.1%}")
    print(f"Output directory: {args.output_dir}")
    print(f"Minimum images per ID: {args.min_images_per_id}")
    print(f"Random seed: {args.random_seed}")
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Load dataset
    df = load_dataset(args.dataset, args.dataset_path)
    
    # Filter by minimum images per identity
    if args.min_images_per_id > 1:
        df = filter_by_min_images(df, args.min_images_per_id)
    
    # Split by identity
    train_df, test_df = split_by_identity(df, args.test_ratio, args.random_seed)
    
    # Relabel identities to consecutive integers
    train_df, test_df = relabel_identities(train_df, test_df)
    
    # Save splits
    save_splits(train_df, test_df, args.output_dir)
    
    print("\n✓ Dataset splitting completed successfully!")


if __name__ == '__main__':
    main()