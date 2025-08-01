#!/usr/bin/env python3
"""
Small Test Data Splitter for Federated Learning
Creates very small-scale clients with hardcoded sizes for testing purposes.
Based on federated_data_splitter.py but optimized for minimal client configurations.
"""

from wildlife_datasets import splits
from wildlife_datasets.datasets import LeopardID2022, MacaqueFaces
import os
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Small Test Federated Data Splitter')
    parser.add_argument('--dataset_path', default='/home/wellvw12/leopard', type=str, help='Path to dataset')
    parser.add_argument('--output_dir', default='/home/wellvw12/fedwild/small_test_leopards', type=str, help='Output directory')
    parser.add_argument('--dataset_type', default='leopard', choices=['leopard', 'macaque'], help='Dataset type')
    
    # Query/Gallery parameters
    parser.add_argument('--query_size', default=15, type=int, help='Number of query samples (IDs)')
    parser.add_argument('--samples_per_query_id', default=2, type=int, help='Number of samples per query ID')
    parser.add_argument('--samples_per_gallery_id', default=8, type=int, help='Number of samples per gallery ID')
    
    # General parameters
    parser.add_argument('--min_samples_per_id', default=4, type=int, help='Minimum samples per ID for inclusion (query+gallery+training)')
    parser.add_argument('--exclude_unknown', default=True, type=bool, help='Exclude samples with unknown identities')
    parser.add_argument('--random_seed', default=42, type=int, help='Random seed for reproducibility')
    
    return parser.parse_args()

def exclude_unknown_identities(metadata):
    """Exclude samples with 'unknown' identity labels"""
    print(f"\nExcluding 'unknown' identities...")
    original_count = len(metadata)
    original_ids = metadata['identity'].nunique()
    
    # Filter out unknown identities (case-insensitive)
    filtered_metadata = metadata[
        ~metadata['identity'].astype(str).str.lower().isin(['unknown', 'nan', 'none', ''])
    ].copy()
    
    # Also filter out NaN values
    filtered_metadata = filtered_metadata.dropna(subset=['identity'])
    
    excluded_count = original_count - len(filtered_metadata)
    remaining_ids = filtered_metadata['identity'].nunique()
    
    print(f"  Excluded {excluded_count} samples with unknown identities")
    print(f"  Original: {original_count} samples ({original_ids} IDs)")
    print(f"  Remaining: {len(filtered_metadata)} samples ({remaining_ids} IDs)")
    
    return filtered_metadata

def filter_ids_by_sample_count(metadata, min_samples):
    """Filter to keep only IDs that have at least min_samples"""
    id_counts = metadata['identity'].value_counts()
    valid_ids = id_counts[id_counts >= min_samples].index
    filtered_metadata = metadata[metadata['identity'].isin(valid_ids)].copy()
    print(f"Original IDs: {len(id_counts)}, IDs with ≥{min_samples} samples: {len(valid_ids)}")
    return filtered_metadata

def create_query_gallery_splits(metadata, query_size, samples_per_query_id, samples_per_gallery_id, random_seed=42):
    """
    Create query and gallery splits from metadata with no overlap to training data
    """
    np.random.seed(random_seed)
    
    # Get IDs with sufficient samples for query+gallery
    required_samples = samples_per_query_id + samples_per_gallery_id
    id_counts = metadata['identity'].value_counts()
    eligible_ids = id_counts[id_counts >= required_samples].index.tolist()
    
    print(f"IDs eligible for query/gallery (≥{required_samples} samples): {len(eligible_ids)}")
    
    if len(eligible_ids) < query_size:
        print(f"Warning: Only {len(eligible_ids)} IDs eligible, but {query_size} requested for query")
        query_size = len(eligible_ids)
    
    # Randomly select IDs for query/gallery
    selected_ids = np.random.choice(eligible_ids, size=query_size, replace=False)
    print(f"Selected {len(selected_ids)} IDs for query/gallery splits")
    
    query_data = []
    gallery_data = []
    
    for identity in selected_ids:
        id_samples = metadata[metadata['identity'] == identity].copy()
        
        # Shuffle samples for this ID
        id_samples = id_samples.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # Take samples for query and gallery
        query_samples = id_samples.head(samples_per_query_id)
        gallery_samples = id_samples.iloc[samples_per_query_id:samples_per_query_id + samples_per_gallery_id]
        
        query_data.append(query_samples)
        gallery_data.append(gallery_samples)
    
    # Combine query and gallery data
    query_metadata = pd.concat(query_data, ignore_index=True) if query_data else pd.DataFrame()
    gallery_metadata = pd.concat(gallery_data, ignore_index=True) if gallery_data else pd.DataFrame()
    
    # Get remaining data (excluding used samples)
    remaining_metadata = metadata.copy()
    for identity in selected_ids:
        id_samples = metadata[metadata['identity'] == identity]
        used_count = samples_per_query_id + samples_per_gallery_id
        # Remove the used samples and keep the rest
        remaining_metadata = remaining_metadata[
            ~((remaining_metadata['identity'] == identity) & 
              (remaining_metadata.index.isin(id_samples.head(used_count).index)))
        ]
    
    print(f"Query samples: {len(query_metadata)} ({query_metadata['identity'].nunique()} IDs)")
    print(f"Gallery samples: {len(gallery_metadata)} ({gallery_metadata['identity'].nunique()} IDs)")
    print(f"Remaining samples for training: {len(remaining_metadata)} ({remaining_metadata['identity'].nunique()} IDs)")
    
    return query_metadata, gallery_metadata, remaining_metadata

def create_small_clients(metadata, random_seed=42):
    """
    Create 4 small clients with hardcoded sizes: 50, 30, 10, 5 samples
    Each client gets roughly the specified samples per ID distribution
    
    Client configurations:
    - Client 0: 50 samples, ~5 samples per ID (10 IDs)
    - Client 1: 30 samples, ~3 samples per ID (10 IDs) 
    - Client 2: 10 samples, ~2 samples per ID (5 IDs)
    - Client 3: 5 samples, ~2-3 samples per ID (2-3 IDs)
    """
    np.random.seed(random_seed)
    
    # Client configurations: (target_samples, target_samples_per_id)
    client_configs = [
        (100, 10),  # Client 0: 50 samples, ~5 per ID
        (40, 8),  # Client 1: 30 samples, ~3 per ID
        (20, 5),  # Client 2: 10 samples, ~2 per ID
        # (10, 5),   # Client 3: 5 samples, ~2 per ID
    ]
    
    # Get available identities and shuffle
    available_ids = list(metadata['identity'].unique())
    np.random.shuffle(available_ids)
    
    client_metadata = {}
    used_ids = set()
    
    print(f"\nCreating small clients with hardcoded sizes:")
    print(f"Available training IDs: {len(available_ids)}")
    
    for client_id, (target_samples, target_samples_per_id) in enumerate(client_configs):
        print(f"\nClient {client_id} target: {target_samples} samples, ~{target_samples_per_id} samples per ID")
        
        client_data = []
        current_samples = 0
        client_ids_used = 0
        
        # Find IDs for this client
        for identity in available_ids:
            if identity in used_ids:
                continue
                
            # Get samples for this identity
            id_samples = metadata[metadata['identity'] == identity].copy()
            
            # Skip if not enough samples for target_samples_per_id
            if len(id_samples) < target_samples_per_id:
                continue
            
            # Take up to target_samples_per_id samples
            samples_to_take = min(target_samples_per_id, 
                                len(id_samples), 
                                target_samples - current_samples)
            
            if samples_to_take <= 0:
                break
                
            # Randomly sample from this identity
            selected_samples = id_samples.sample(n=samples_to_take, random_state=random_seed)
            client_data.append(selected_samples)
            
            current_samples += samples_to_take
            client_ids_used += 1
            used_ids.add(identity)
            
            # Stop if we've reached target samples
            if current_samples >= target_samples:
                break
        
        # Combine client data
        if client_data:
            client_metadata[client_id] = pd.concat(client_data, ignore_index=True)
        else:
            client_metadata[client_id] = pd.DataFrame()
        
        actual_samples = len(client_metadata[client_id])
        actual_ids = client_metadata[client_id]['identity'].nunique() if actual_samples > 0 else 0
        avg_samples_per_id = actual_samples / actual_ids if actual_ids > 0 else 0
        
        print(f"Client {client_id} actual: {actual_samples} samples, {actual_ids} IDs, "
              f"{avg_samples_per_id:.1f} samples per ID")
        
        # Show sample distribution per ID for this client
        if actual_samples > 0:
            id_counts = client_metadata[client_id]['identity'].value_counts()
            print(f"  Samples per ID: min={id_counts.min()}, max={id_counts.max()}, "
                  f"mean={id_counts.mean():.1f}")
    
    print(f"\nTotal IDs used across all clients: {len(used_ids)}")
    print(f"Unused IDs available: {len(available_ids) - len(used_ids)}")
    
    # Verify no overlap between clients
    all_assigned_ids = set()
    for client_id in range(len(client_configs)):
        if len(client_metadata[client_id]) > 0:
            client_ids = set(client_metadata[client_id]['identity'].unique())
            overlap = all_assigned_ids.intersection(client_ids)
            if overlap:
                print(f"WARNING: Found ID overlap for client {client_id}: {overlap}")
            all_assigned_ids.update(client_ids)
    
    print(f"Verification: Total unique IDs assigned = {len(all_assigned_ids)}")
    
    return client_metadata

def create_unified_metadata(query_metadata, gallery_metadata, client_metadata):
    """
    Create unified metadata with split and client columns
    """
    all_data = []
    
    # Add query data
    if len(query_metadata) > 0:
        query_df = query_metadata.copy()
        query_df['split'] = 'query'
        query_df['client'] = -1  # -1 indicates shared evaluation data
        all_data.append(query_df)
    
    # Add gallery data
    if len(gallery_metadata) > 0:
        gallery_df = gallery_metadata.copy()
        gallery_df['split'] = 'gallery'
        gallery_df['client'] = -1  # -1 indicates shared evaluation data
        all_data.append(gallery_df)
    
    # Add training data for each client
    for client_id in range(3):  # 4 hardcoded clients
        if len(client_metadata[client_id]) > 0:
            train_df = client_metadata[client_id].copy()
            train_df['split'] = 'train'
            train_df['client'] = client_id
            all_data.append(train_df)
    
    # Combine all data
    unified_metadata = pd.concat(all_data, ignore_index=True)
    
    print(f"\nUnified metadata summary:")
    print(f"Total samples: {len(unified_metadata)}")
    print("\nSplit distribution:")
    split_counts = unified_metadata['split'].value_counts()
    for split_name, count in split_counts.items():
        unique_ids = unified_metadata[unified_metadata['split'] == split_name]['identity'].nunique()
        print(f"  {split_name}: {count} samples ({unique_ids} IDs)")
    
    print(f"\nClient distribution:")
    client_counts = unified_metadata['client'].value_counts().sort_index()
    for client_id, count in client_counts.items():
        if client_id == -1:
            print(f"  Shared (query/gallery): {count} samples")
        else:
            unique_ids = unified_metadata[
                (unified_metadata['client'] == client_id) & 
                (unified_metadata['split'] == 'train')
            ]['identity'].nunique()
            print(f"  Client {client_id}: {count} samples ({unique_ids} IDs)")
    
    return unified_metadata

def save_detailed_stats(args, unified_metadata, client_metadata):
    """Save detailed statistics about the small test dataset"""
    stats_path = os.path.join(args.output_dir, 'small_test_stats.txt')
    
    with open(stats_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SMALL TEST FEDERATED DATA STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Dataset: {args.dataset_type}\n")
        f.write(f"Number of clients: 4 (hardcoded)\n")
        f.write(f"Client sizes: [50, 30, 10, 5] samples\n")
        f.write(f"Query size: {args.query_size} IDs\n")
        f.write(f"Samples per query ID: {args.samples_per_query_id}\n")
        f.write(f"Samples per gallery ID: {args.samples_per_gallery_id}\n")
        f.write(f"Minimum samples per ID: {args.min_samples_per_id}\n")
        f.write(f"Random seed: {args.random_seed}\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total samples: {len(unified_metadata)}\n")
        f.write(f"Total unique IDs: {unified_metadata['identity'].nunique()}\n\n")
        
        # Split distribution
        f.write("SPLIT DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        split_counts = unified_metadata['split'].value_counts()
        for split_name in ['train', 'query', 'gallery']:
            if split_name in split_counts:
                count = split_counts[split_name]
                unique_ids = unified_metadata[unified_metadata['split'] == split_name]['identity'].nunique()
                f.write(f"{split_name.capitalize()}: {count} samples, {unique_ids} unique IDs\n")
        f.write("\n")
        
        # Detailed client analysis
        f.write("CLIENT DISTRIBUTION DETAILS:\n")
        f.write("-" * 30 + "\n")
        
        client_sizes = []
        for client_id in range(3):
            if len(client_metadata[client_id]) > 0:
                num_samples = len(client_metadata[client_id])
                num_ids = client_metadata[client_id]['identity'].nunique()
                client_sizes.append(num_samples)
                
                f.write(f"Client {client_id}:\n")
                f.write(f"  - Samples: {num_samples}\n")
                f.write(f"  - Unique IDs: {num_ids}\n")
                f.write(f"  - Avg samples per ID: {num_samples/num_ids:.1f}\n")
                
                # Distribution of samples per ID
                id_counts = client_metadata[client_id]['identity'].value_counts()
                f.write(f"  - Samples per ID range: {id_counts.min()}-{id_counts.max()}\n")
                f.write(f"  - Identity list: {list(client_metadata[client_id]['identity'].unique())}\n")
                f.write("\n")
            else:
                client_sizes.append(0)
                f.write(f"Client {client_id}: No data\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 30 + "\n")
        total_train_samples = sum(client_sizes)
        f.write(f"Total training samples: {total_train_samples}\n")
        f.write(f"Client size distribution: {client_sizes}\n")
        f.write(f"Min client size: {min(client_sizes)}\n")
        f.write(f"Max client size: {max(client_sizes)}\n")
        f.write(f"Mean client size: {np.mean(client_sizes):.1f}\n")
        f.write(f"Std client size: {np.std(client_sizes):.1f}\n")
        
        # Data isolation verification
        f.write("\nDATA ISOLATION:\n")
        f.write("-" * 30 + "\n")
        all_train_ids = set()
        overlapping_ids = set()
        
        for client_id in range(3):
            if len(client_metadata[client_id]) > 0:
                client_ids = set(client_metadata[client_id]['identity'].unique())
                intersection = all_train_ids.intersection(client_ids)
                if intersection:
                    overlapping_ids.update(intersection)
                all_train_ids.update(client_ids)
        
        f.write(f"Total unique training IDs: {len(all_train_ids)}\n")
        f.write(f"Overlapping IDs: {len(overlapping_ids)}\n")
        f.write(f"Perfect isolation: {'Yes' if len(overlapping_ids) == 0 else 'No'}\n")
    
    print(f"\nDetailed statistics saved to: {stats_path}")

def main():
    args = parse_args()
    
    print("=== Small Test Federated Data Splitter ===")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Output directory: {args.output_dir}")
    print(f"Hardcoded client sizes: [50, 30, 10, 5] samples")
    print(f"Query size: {args.query_size} IDs")
    print(f"Samples per query ID: {args.samples_per_query_id}")
    print(f"Samples per gallery ID: {args.samples_per_gallery_id}")
    print(f"Minimum samples per ID: {args.min_samples_per_id}")
    print(f"Random seed: {args.random_seed}")
    
    # Load dataset
    print("\nLoading dataset...")
    if args.dataset_type == 'leopard':
        dataset = LeopardID2022(args.dataset_path).metadata
    elif args.dataset_type == 'macaque':
        dataset = MacaqueFaces(args.dataset_path).metadata
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    print(f"Total samples: {len(dataset)}")
    print(f"Total unique IDs: {dataset['identity'].nunique()}")
    
    # Exclude unknown identities first (if enabled)
    if args.exclude_unknown:
        dataset = exclude_unknown_identities(dataset)
    else:
        print("\nSkipping unknown identity exclusion")
    
    # Filter dataset to keep only IDs with sufficient samples
    print(f"\nFiltering IDs with at least {args.min_samples_per_id} samples...")
    filtered_metadata = filter_ids_by_sample_count(dataset, args.min_samples_per_id)
    print(f"Filtered dataset size: {len(filtered_metadata)}")
    
    # Reset index to ensure proper indexing
    filtered_metadata = filtered_metadata.reset_index(drop=True)
    
    # Create query and gallery splits
    print(f"\nCreating query and gallery splits...")
    query_metadata, gallery_metadata, remaining_metadata = create_query_gallery_splits(
        filtered_metadata, 
        args.query_size, 
        args.samples_per_query_id, 
        args.samples_per_gallery_id, 
        args.random_seed
    )
    
    # Create small clients with hardcoded sizes
    print(f"\nCreating 4 small clients with hardcoded sizes...")
    client_metadata = create_small_clients(remaining_metadata, args.random_seed)
    
    # Create unified metadata
    print(f"\nCreating unified metadata...")
    unified_metadata = create_unified_metadata(
        query_metadata, 
        gallery_metadata, 
        client_metadata
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save unified metadata
    metadata_path = os.path.join(args.output_dir, 'metadata.csv')
    unified_metadata.to_csv(metadata_path, index=False)
    print(f"\nSaved unified metadata to {metadata_path}")
    
    # Save detailed statistics
    save_detailed_stats(args, unified_metadata, client_metadata)
    
    # Print final summary
    print(f"\n=== Final Summary ===")
    print(f"Total samples processed: {len(unified_metadata)}")
    print(f"Query set: {len(query_metadata)} samples from {args.query_size} IDs")
    print(f"Gallery set: {len(gallery_metadata)} samples from {args.query_size} IDs")
    
    total_train_samples = sum(len(client_metadata[i]) for i in range(3))
    total_train_ids = len(remaining_metadata['identity'].unique()) if len(remaining_metadata) > 0 else 0
    print(f"Training set: {total_train_samples} samples from {total_train_ids} IDs")
    print(f"Distributed across 4 clients: [50, 30, 10, 5] target samples")
    
    # Show actual client sizes
    actual_sizes = [len(client_metadata[i]) for i in range(3)]
    print(f"Actual client sizes: {actual_sizes}")
    
    print(f"\nOutput saved to: {args.output_dir}")
    print("Ready for small-scale federated learning experiments!")

if __name__ == "__main__":
    main()