from wildlife_datasets import splits
from wildlife_datasets.datasets import LeopardID2022, MacaqueFaces
import os
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Data Splitter with Client Allocation')
    parser.add_argument('--dataset_path', default='/home/wellvw12/fedwild/MacaqueFaces', type=str, help='Path to dataset')
    parser.add_argument('--output_dir', default='/home/wellvw12/fedwild/macaque_help', type=str, help='Output directory')
    parser.add_argument('--dataset_type', default='macaque', choices=['leopard', 'macaque'], help='Dataset type')
    
    # Query/Gallery parameters
    parser.add_argument('--query_size', default=7, type=int, help='Number of query samples (IDs)')
    parser.add_argument('--samples_per_query_id', default=2, type=int, help='Number of samples per query ID')
    parser.add_argument('--samples_per_gallery_id', default=5, type=int, help='Number of samples per gallery ID')
    
    # Client federation parameters
    parser.add_argument('--num_clients', default=5, type=int, help='Number of federated clients')
    parser.add_argument('--alpha', default=0.9, type=float, help='Dirichlet alpha parameter (lower = more heterogeneous)')
    
    # General parameters
    parser.add_argument('--min_samples_per_id', default=2, type=int, help='Minimum samples per ID for inclusion')
    parser.add_argument('--max_samples_per_id', default=20, type=int, help='Maximum samples per ID to include in training (None = no limit)')
    parser.add_argument('--min_train_samples_per_id', default=2, type=int, help='Minimum samples per ID in training data')
    parser.add_argument('--exclude_unknown', default=True, type=bool, help='Exclude samples with unknown identities')
    parser.add_argument('--random_seed', default=27, type=int, help='Random seed for reproducibility')
    
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

def limit_samples_per_id(metadata, max_samples_per_id, min_samples_per_id, random_seed=42):
    """
    Limit the number of samples per identity for training data
    
    Args:
        metadata: Input metadata
        max_samples_per_id: Maximum samples to keep per ID (None = no limit)
        min_samples_per_id: Minimum samples required per ID after limiting
        random_seed: Random seed for reproducible sampling
    
    Returns:
        Limited metadata
    """
    if max_samples_per_id is None:
        print("No maximum samples per ID limit applied")
        return metadata
    
    np.random.seed(random_seed)
    limited_data = []
    
    print(f"Limiting to maximum {max_samples_per_id} samples per ID")
    
    for identity, group in metadata.groupby('identity'):
        group_samples = group.copy()
        
        if len(group_samples) > max_samples_per_id:
            # Randomly sample max_samples_per_id from this identity
            group_samples = group_samples.sample(n=max_samples_per_id, random_state=random_seed)
        
        # Check if we still have minimum required samples
        if len(group_samples) >= min_samples_per_id:
            limited_data.append(group_samples)
    
    if limited_data:
        limited_metadata = pd.concat(limited_data, ignore_index=True)
    else:
        limited_metadata = pd.DataFrame()
    
    original_ids = metadata['identity'].nunique()
    limited_ids = limited_metadata['identity'].nunique() if len(limited_metadata) > 0 else 0
    
    print(f"After limiting: {len(limited_metadata)} samples from {limited_ids} IDs")
    print(f"IDs removed due to insufficient samples after limiting: {original_ids - limited_ids}")
    
    return limited_metadata

def create_query_gallery_splits(metadata, query_size, samples_per_query_id, samples_per_gallery_id, random_seed=42):
    """
    Create query and gallery splits from metadata
    
    Args:
        metadata: Full dataset metadata
        query_size: Number of IDs to include in query set
        samples_per_query_id: Number of samples per query ID
        samples_per_gallery_id: Number of samples per gallery ID
        random_seed: Random seed for reproducibility
    
    Returns:
        query_metadata, gallery_metadata, remaining_metadata
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
    used_indices = set()
    if len(query_metadata) > 0:
        used_indices.update(query_metadata.index)
    if len(gallery_metadata) > 0:
        used_indices.update(gallery_metadata.index)
    
    # Create remaining metadata by removing used samples
    remaining_metadata = metadata.copy()
    for identity in selected_ids:
        id_samples = metadata[metadata['identity'] == identity]
        used_count = samples_per_query_id + samples_per_gallery_id
        remaining_id_samples = id_samples.iloc[used_count:]
        # Remove the used samples and keep the rest
        remaining_metadata = remaining_metadata[
            ~((remaining_metadata['identity'] == identity) & 
              (remaining_metadata.index.isin(id_samples.head(used_count).index)))
        ]
    
    print(f"Query samples: {len(query_metadata)} ({query_metadata['identity'].nunique()} IDs)")
    print(f"Gallery samples: {len(gallery_metadata)} ({gallery_metadata['identity'].nunique()} IDs)")
    print(f"Remaining samples for training: {len(remaining_metadata)} ({remaining_metadata['identity'].nunique()} IDs)")
    
    return query_metadata, gallery_metadata, remaining_metadata

def dirichlet_split_by_identity(metadata, num_clients, alpha, random_seed=42):
    """
    Split identities among clients using Dirichlet distribution with no overlap
    Each identity is assigned to exactly one client
    
    Args:
        metadata: Training metadata
        num_clients: Number of clients
        alpha: Dirichlet alpha parameter
        random_seed: Random seed
    
    Returns:
        Dictionary mapping client_id to metadata
    """
    np.random.seed(random_seed)
    
    # Get unique identities
    unique_ids = metadata['identity'].unique()
    num_classes = len(unique_ids)
    
    print(f"Splitting {num_classes} identities among {num_clients} clients using Dirichlet(α={alpha})")
    print("No identity overlap between clients - each ID assigned to exactly one client")
    
    # Sample proportions from Dirichlet distribution for client sizes
    client_proportions = np.random.dirichlet([alpha] * num_clients)
    
    # Calculate how many identities each client should get
    ids_per_client = np.floor(client_proportions * num_classes).astype(int)
    
    # Handle remainder identities
    remainder = num_classes - ids_per_client.sum()
    if remainder > 0:
        # Add remainder identities to clients with highest proportions
        top_clients = np.argsort(client_proportions)[-remainder:]
        ids_per_client[top_clients] += 1
    
    print(f"Identity distribution per client: {ids_per_client}")
    
    # Shuffle identities for random assignment
    shuffled_ids = np.random.permutation(unique_ids)
    
    # Assign identities to clients
    client_data = {i: [] for i in range(num_clients)}
    start_idx = 0
    
    for client_id in range(num_clients):
        end_idx = start_idx + ids_per_client[client_id]
        client_identities = shuffled_ids[start_idx:end_idx]
        
        # Get all samples for these identities
        for identity in client_identities:
            identity_data = metadata[metadata['identity'] == identity].copy()
            client_data[client_id].append(identity_data)
        
        start_idx = end_idx
    
    # Combine data for each client
    client_metadata = {}
    for client_id in range(num_clients):
        if client_data[client_id]:
            client_metadata[client_id] = pd.concat(client_data[client_id], ignore_index=True)
        else:
            client_metadata[client_id] = pd.DataFrame()
        
        print(f"Client {client_id}: {len(client_metadata[client_id])} samples, "
              f"{client_metadata[client_id]['identity'].nunique() if len(client_metadata[client_id]) > 0 else 0} IDs")
    
    # Verify no overlap
    all_assigned_ids = set()
    for client_id in range(num_clients):
        if len(client_metadata[client_id]) > 0:
            client_ids = set(client_metadata[client_id]['identity'].unique())
            overlap = all_assigned_ids.intersection(client_ids)
            if overlap:
                print(f"WARNING: Found ID overlap for client {client_id}: {overlap}")
            all_assigned_ids.update(client_ids)
    
    print(f"Total unique IDs assigned: {len(all_assigned_ids)} out of {num_classes}")
    
    return client_metadata

def create_unified_metadata(query_metadata, gallery_metadata, client_metadata, num_clients):
    """
    Create unified metadata with split and client columns
    
    Args:
        query_metadata: Query split data
        gallery_metadata: Gallery split data  
        client_metadata: Dictionary of client training data
        num_clients: Number of clients
    
    Returns:
        Unified metadata DataFrame
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
    for client_id in range(num_clients):
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

def main():
    args = parse_args()
    
    print("=== Federated Data Splitter with Client Allocation ===")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Output directory: {args.output_dir}")
    print(f"Query size: {args.query_size} IDs")
    print(f"Samples per query ID: {args.samples_per_query_id}")
    print(f"Samples per gallery ID: {args.samples_per_gallery_id}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Dirichlet alpha: {args.alpha}")
    print(f"Minimum samples per ID: {args.min_samples_per_id}")
    print(f"Maximum samples per ID: {args.max_samples_per_id if args.max_samples_per_id else 'unlimited'}")
    print(f"Minimum training samples per ID: {args.min_train_samples_per_id}")
    print(f"Exclude unknown identities: {args.exclude_unknown}")
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
    
    # Save full annotations
    dataset.to_csv("annotations.csv", index=False)
    print("Saved full annotations to annotations.csv")
    
    # Exclude unknown identities first (if enabled)
    if args.exclude_unknown:
        dataset = exclude_unknown_identities(dataset)
    else:
        print("\nSkipping unknown identity exclusion (disabled by --exclude_unknown=False)")
    
    # Filter dataset to keep only IDs with sufficient samples
    print(f"\nFiltering IDs with at least {args.min_samples_per_id} samples...")
    filtered_metadata = filter_ids_by_sample_count(dataset, args.min_samples_per_id)
    print(f"Filtered dataset size: {len(filtered_metadata)}")
    
    # Limit samples per ID for training
    print(f"\nApplying sample limits per ID...")
    limited_metadata = limit_samples_per_id(
        filtered_metadata, 
        args.max_samples_per_id, 
        args.min_train_samples_per_id, 
        args.random_seed
    )
    
    # Reset index to ensure proper indexing
    limited_metadata = limited_metadata.reset_index(drop=True)
    
    # Create query and gallery splits
    print(f"\nCreating query and gallery splits...")
    query_metadata, gallery_metadata, remaining_metadata = create_query_gallery_splits(
        limited_metadata, 
        args.query_size, 
        args.samples_per_query_id, 
        args.samples_per_gallery_id, 
        args.random_seed
    )
    
    # Split remaining data among clients using Dirichlet distribution
    print(f"\nSplitting training data among {args.num_clients} clients...")
    client_metadata = dirichlet_split_by_identity(
        remaining_metadata, 
        args.num_clients, 
        args.alpha, 
        args.random_seed
    )
    
    # Create unified metadata
    print(f"\nCreating unified metadata...")
    unified_metadata = create_unified_metadata(
        query_metadata, 
        gallery_metadata, 
        client_metadata, 
        args.num_clients
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save unified metadata
    metadata_path = os.path.join(args.output_dir, 'metadata.csv')
    unified_metadata.to_csv(metadata_path, index=False)
    print(f"\nSaved unified metadata to {metadata_path}")
    
    # Save backup to metadata directory
    os.makedirs('metadata/federated/', exist_ok=True)
    unified_metadata.to_csv('metadata/federated/metadata.csv', index=False)
    print("Saved backup to metadata/federated/metadata.csv")
    
    # Print final statistics
    print(f"\n=== Final Statistics ===")
    print(f"Total samples processed: {len(unified_metadata)}")
    print(f"Query set: {len(query_metadata)} samples from {args.query_size} IDs")
    print(f"Gallery set: {len(gallery_metadata)} samples from {args.query_size} IDs")
    
    total_train_samples = sum(len(client_metadata[i]) for i in range(args.num_clients))
    total_train_ids = len(remaining_metadata['identity'].unique()) if len(remaining_metadata) > 0 else 0
    print(f"Training set: {total_train_samples} samples from {total_train_ids} IDs")
    print(f"Distributed across {args.num_clients} clients using Dirichlet(α={args.alpha})")
    
    # Calculate and save heterogeneity statistics
    if total_train_samples > 0:
        client_sizes = [len(client_metadata[i]) for i in range(args.num_clients)]
        client_ids_counts = [client_metadata[i]['identity'].nunique() if len(client_metadata[i]) > 0 else 0 for i in range(args.num_clients)]
        
        print(f"Client size distribution - Min: {min(client_sizes)}, Max: {max(client_sizes)}, "
              f"Mean: {np.mean(client_sizes):.1f}, Std: {np.std(client_sizes):.1f}")
        
        # Save detailed statistics to file
        save_client_stats(args, unified_metadata, client_metadata, args.num_clients)

def save_client_stats(args, unified_metadata, client_metadata, num_clients):
    """Save detailed client distribution statistics to a text file"""
    
    stats_path = os.path.join(args.output_dir, 'client_distribution_stats.txt')
    
    with open(stats_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FEDERATED DATA DISTRIBUTION STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic configuration
        f.write("CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Dataset: {args.dataset_type}\n")
        f.write(f"Number of clients: {num_clients}\n")
        f.write(f"Dirichlet alpha: {args.alpha}\n")
        f.write(f"Query size: {args.query_size} IDs\n")
        f.write(f"Samples per query ID: {args.samples_per_query_id}\n")
        f.write(f"Samples per gallery ID: {args.samples_per_gallery_id}\n")
        f.write(f"Minimum samples per ID: {args.min_samples_per_id}\n")
        f.write(f"Maximum samples per ID: {args.max_samples_per_id if args.max_samples_per_id else 'unlimited'}\n")
        f.write(f"Minimum training samples per ID: {args.min_train_samples_per_id}\n")
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
        
        if '' in split_counts:
            f.write(f"Unused: {split_counts['']} samples\n")
        f.write("\n")
        
        # Client distribution details
        f.write("CLIENT DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        
        client_sizes = []
        client_ids_counts = []
        
        for client_id in range(num_clients):
            if len(client_metadata[client_id]) > 0:
                num_samples = len(client_metadata[client_id])
                num_ids = client_metadata[client_id]['identity'].nunique()
                client_sizes.append(num_samples)
                client_ids_counts.append(num_ids)
                
                f.write(f"Client {client_id}: {num_samples} samples, {num_ids} unique IDs\n")
                
                # Top identities for this client
                id_counts = client_metadata[client_id]['identity'].value_counts()
                f.write(f"  - Samples per ID: min={id_counts.min()}, max={id_counts.max()}, mean={id_counts.mean():.1f}\n")
                f.write(f"  - Top 5 IDs: {list(id_counts.head().index)}\n")
            else:
                client_sizes.append(0)
                client_ids_counts.append(0)
                f.write(f"Client {client_id}: 0 samples, 0 unique IDs\n")
        
        f.write("\n")
        
        # Statistical analysis
        f.write("HETEROGENEITY ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        if len(client_sizes) > 0:
            f.write(f"Sample distribution across clients:\n")
            f.write(f"  - Min: {min(client_sizes)} samples\n")
            f.write(f"  - Max: {max(client_sizes)} samples\n")
            f.write(f"  - Mean: {np.mean(client_sizes):.1f} samples\n")
            f.write(f"  - Std: {np.std(client_sizes):.1f} samples\n")
            f.write(f"  - Coefficient of Variation: {np.std(client_sizes)/np.mean(client_sizes):.3f}\n\n")
            
            f.write(f"ID distribution across clients:\n")
            f.write(f"  - Min: {min(client_ids_counts)} IDs\n")
            f.write(f"  - Max: {max(client_ids_counts)} IDs\n")
            f.write(f"  - Mean: {np.mean(client_ids_counts):.1f} IDs\n")
            f.write(f"  - Std: {np.std(client_ids_counts):.1f} IDs\n")
            f.write(f"  - Coefficient of Variation: {np.std(client_ids_counts)/np.mean(client_ids_counts):.3f}\n\n")
        
        # Data overlap analysis
        f.write("DATA OVERLAP ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        all_train_ids = set()
        overlapping_ids = set()
        
        for client_id in range(num_clients):
            if len(client_metadata[client_id]) > 0:
                client_ids = set(client_metadata[client_id]['identity'].unique())
                intersection = all_train_ids.intersection(client_ids)
                if intersection:
                    overlapping_ids.update(intersection)
                all_train_ids.update(client_ids)
        
        f.write(f"Total unique training IDs across all clients: {len(all_train_ids)}\n")
        f.write(f"Overlapping IDs between clients: {len(overlapping_ids)}\n")
        f.write(f"Data isolation: {'Perfect' if len(overlapping_ids) == 0 else 'Partial'}\n\n")
        
        # Alpha parameter interpretation
        f.write("DIRICHLET PARAMETER INTERPRETATION:\n")
        f.write("-" * 30 + "\n")
        if args.alpha < 0.1:
            f.write("α < 0.1: Highly heterogeneous (very non-IID)\n")
        elif args.alpha < 0.5:
            f.write("α < 0.5: Moderately heterogeneous (non-IID)\n")
        elif args.alpha < 1.0:
            f.write("α < 1.0: Slightly heterogeneous (mildly non-IID)\n")
        elif args.alpha == 1.0:
            f.write("α = 1.0: Uniform distribution (most balanced)\n")
        else:
            f.write("α > 1.0: Concentrated distribution\n")
        
        cv_samples = np.std(client_sizes) / np.mean(client_sizes) if np.mean(client_sizes) > 0 else 0
        if cv_samples < 0.1:
            f.write(f"Current α = {args.alpha}: Very balanced distribution\n")
        elif cv_samples < 0.3:
            f.write(f"Current α = {args.alpha}: Moderately balanced distribution\n")
        elif cv_samples < 0.5:
            f.write(f"Current α = {args.alpha}: Somewhat imbalanced distribution\n")
        else:
            f.write(f"Current α = {args.alpha}: Highly imbalanced distribution\n")
    
    print(f"\nDetailed client distribution statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()