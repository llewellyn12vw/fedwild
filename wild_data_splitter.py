from wildlife_datasets import datasets, splits
from wildlife_datasets.datasets import Cows2021v2, LeopardID2022, HyenaID2022, SeaTurtleID2022, MacaqueFaces
import os
import pandas as pd
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Wildlife Data Splitter with Query/Gallery Creation')
    parser.add_argument('--dataset_path', default='/home/wellvw12/leopard', type=str, help='Path to dataset')
    parser.add_argument('--output_dir', default='/home/wellvw12/fedwild/data', type=str, help='Output directory')
    parser.add_argument('--client_id', default='0', type=str, help='Client ID for output')
    parser.add_argument('--test_split', default=0.96, type=float, help='Test split ratio (0.2 = 20%)')
    parser.add_argument('--min_samples_per_id', default=4, type=int, help='Minimum samples per ID (1 query + 3 gallery)')
    parser.add_argument('--max_query_ids', default=20, type=int, help='Maximum number of IDs to include in query set (caps query set size)')
    parser.add_argument('--max_gallery_ids', default=20, type=int, help='Maximum number of IDs to include in gallery set (caps gallery set size)')
    parser.add_argument('--random_seed', default=42, type=int, help='Random seed for reproducibility')
    return parser.parse_args()

def filter_ids_by_sample_count(metadata, min_samples):
    """Filter to keep only IDs that have at least min_samples"""
    id_counts = metadata['identity'].value_counts()
    valid_ids = id_counts[id_counts >= min_samples].index
    filtered_metadata = metadata[metadata['identity'].isin(valid_ids)].copy()
    print(f"Original IDs: {len(id_counts)}, IDs with ≥{min_samples} samples: {len(valid_ids)}")
    return filtered_metadata

def assign_query_gallery_splits(test_metadata, random_seed=42, max_query_ids=None, max_gallery_ids=None):
    """
    Assign query/gallery splits to test data:
    - 1 sample per ID for query
    - 3 samples per ID for gallery  
    - Remaining samples get empty split value
    - Caps query and gallery sets if max_query_ids and max_gallery_ids are specified
    Returns metadata with 'split' column updated
    """
    np.random.seed(random_seed)
    
    # Create a copy to modify
    result_metadata = test_metadata.copy()
    
    # Initialize all test samples with empty split
    result_metadata['split'] = ''
    
    query_count = 0
    gallery_count = 0
    unused_count = 0
    
    # Group by identity and get list of identities
    grouped = result_metadata.groupby('identity')
    all_identities = list(grouped.groups.keys())
    
    # Shuffle identities for random selection if capping
    np.random.shuffle(all_identities)
    
    # Determine which identities to use for query and gallery
    if max_query_ids is not None:
        query_identities = all_identities[:max_query_ids]
    else:
        query_identities = all_identities
    
    if max_gallery_ids is not None:
        gallery_identities = all_identities[:max_gallery_ids]
    else:
        gallery_identities = all_identities
    
    # Process each identity
    for identity in all_identities:
        group = grouped.get_group(identity)
        group_indices = group.sample(frac=1, random_state=random_seed).index
        
        if len(group_indices) < 4:  # Need at least 4 samples (1 query + 3 gallery)
            print(f"Warning: Identity {identity} has only {len(group_indices)} samples, leaving unused...")
            unused_count += len(group_indices)
            continue
        
        # Check if this identity should be included in query/gallery sets
        include_in_query = identity in query_identities
        include_in_gallery = identity in gallery_identities
        
        if include_in_query:
            # Assign query sample
            query_idx = group_indices[0]
            result_metadata.loc[query_idx, 'split'] = 'query'
            query_count += 1
            start_gallery_idx = 1
        else:
            start_gallery_idx = 0
        
        if include_in_gallery:
            # Assign gallery samples
            gallery_indices = group_indices[start_gallery_idx:start_gallery_idx+3]
            result_metadata.loc[gallery_indices, 'split'] = 'gallery'
            gallery_count += len(gallery_indices)
            unused_indices = group_indices[start_gallery_idx+3:]
        else:
            unused_indices = group_indices[start_gallery_idx:]
        
        unused_count += len(unused_indices)
    
    print(f"Query samples: {query_count}")
    print(f"Gallery samples: {gallery_count}")
    print(f"Unused test samples: {unused_count}")
    print(f"Unique IDs in query: {(result_metadata['split'] == 'query').sum()}")
    print(f"Unique IDs in gallery: {result_metadata[result_metadata['split'] == 'gallery']['identity'].nunique()}")
    
    if max_query_ids is not None:
        print(f"Query set capped at {max_query_ids} IDs")
    if max_gallery_ids is not None:
        print(f"Gallery set capped at {max_gallery_ids} IDs")
    
    return result_metadata

def main():
    args = parse_args()
    
    print("=== Wildlife Data Splitter with Query/Gallery Creation ===")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test split ratio: {args.test_split}")
    print(f"Minimum samples per ID: {args.min_samples_per_id}")
    print(f"Max query IDs: {args.max_query_ids if args.max_query_ids else 'unlimited'}")
    print(f"Max gallery IDs: {args.max_gallery_ids if args.max_gallery_ids else 'unlimited'}")
    print(f"Random seed: {args.random_seed}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = LeopardID2022(args.dataset_path).metadata
    print(f"Total samples: {len(dataset)}")
    print(f"Total unique IDs: {dataset['identity'].nunique()}")
    
    # Save full annotations
    dataset.to_csv("annotations.csv", index=False)
    print("Saved full annotations to annotations.csv")
    
    # Filter dataset to keep only IDs with sufficient samples
    print(f"\nFiltering IDs with at least {args.min_samples_per_id} samples...")
    filtered_metadata = filter_ids_by_sample_count(dataset, args.min_samples_per_id)
    print(f"Filtered dataset size: {len(filtered_metadata)}")
    
    # Reset index to ensure proper indexing for splitter
    filtered_metadata = filtered_metadata.reset_index(drop=True)
    
    # Split into train and test
    print(f"\nSplitting data (train: {1-args.test_split:.1%}, test: {args.test_split:.1%})...")
    splitter = splits.DisjointSetSplit(args.test_split)
    
    # Set numpy random seed for reproducibility
    np.random.seed(args.random_seed)
    idx_train, idx_test = splitter.split(filtered_metadata)[0]
    
    train_metadata = filtered_metadata.iloc[idx_train].copy()
    test_metadata = filtered_metadata.iloc[idx_test].copy()
    
    print(f"Train samples: {len(train_metadata)} ({train_metadata['identity'].nunique()} IDs)")
    print(f"Test samples: {len(test_metadata)} ({test_metadata['identity'].nunique()} IDs)")
    
    # Create query and gallery assignments from test split
    print(f"\nAssigning query/gallery splits from test data...")
    test_with_splits = assign_query_gallery_splits(
        test_metadata, 
        args.random_seed, 
        args.max_query_ids, 
        args.max_gallery_ids
    )
    
    # Create unified metadata with all split assignments
    final_metadata = filtered_metadata.copy()
    
    # Initialize split column
    final_metadata['split'] = ''
    
    # Assign train splits
    final_metadata.loc[idx_train, 'split'] = 'train'
    
    # Assign test splits (query, gallery, or empty for unused)
    final_metadata.loc[idx_test, 'split'] = test_with_splits['split'].values
    
    # Create output directory
    output_path = os.path.join(args.output_dir, args.client_id)
    os.makedirs(output_path, exist_ok=True)
    
    # Save the unified metadata file
    print(f"\nSaving unified metadata to {output_path}/...")
    final_metadata.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)
    print(f"Saved unified metadata.csv with split assignments")
    
    # Also save individual split files for compatibility
    train_data = final_metadata[final_metadata['split'] == 'train']
    query_data = final_metadata[final_metadata['split'] == 'query'] 
    gallery_data = final_metadata[final_metadata['split'] == 'gallery']
    
    
    print(f"Also saved individual split files for compatibility:")
    print(f"  train.csv: {len(train_data)} samples")
    print(f"  query.csv: {len(query_data)} samples") 
    print(f"  gallery.csv: {len(gallery_data)} samples")
    
    # Save backup to metadata directory
    os.makedirs('metadata/maq/', exist_ok=True)
    final_metadata.to_csv('metadata/maq/metadata.csv', index=False)
    print("Saved backup to metadata/maq/metadata.csv")
    
    # Print final statistics
    print(f"\n=== Final Statistics ===")
    split_counts = final_metadata['split'].value_counts()
    print(f"Split distribution:")
    for split_name, count in split_counts.items():
        if split_name == '':
            print(f"  unused: {count} samples")
        else:
            unique_ids = final_metadata[final_metadata['split'] == split_name]['identity'].nunique()
            print(f"  {split_name}: {count} samples ({unique_ids} IDs)")
    
    # Verify query/gallery structure
    query_data = final_metadata[final_metadata['split'] == 'query']
    gallery_data = final_metadata[final_metadata['split'] == 'gallery']
    
    if len(query_data) > 0:
        query_id_counts = query_data['identity'].value_counts()
        print(f"\nQuery samples per ID - Min: {query_id_counts.min()}, Max: {query_id_counts.max()}, Mean: {query_id_counts.mean():.1f}")
        
        if query_id_counts.min() == query_id_counts.max() == 1:
            print("✓ Query: Exactly 1 sample per ID")
        else:
            print("✗ Query: Not exactly 1 sample per ID")
    
    if len(gallery_data) > 0:
        gallery_id_counts = gallery_data['identity'].value_counts()
        print(f"Gallery samples per ID - Min: {gallery_id_counts.min()}, Max: {gallery_id_counts.max()}, Mean: {gallery_id_counts.mean():.1f}")
        
        if gallery_id_counts.min() == gallery_id_counts.max() == 3:
            print("✓ Gallery: Exactly 3 samples per ID")
        else:
            print("✗ Gallery: Not exactly 3 samples per ID")

if __name__ == "__main__":
    main()