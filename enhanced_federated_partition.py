#!/usr/bin/env python3
"""
Enhanced Federated Partitioning Tool
Splits dataset into test/training sets, then distributes training data across clients using Dirichlet distribution.
Ensures no ID overlap between test and training sets.
"""

import os
import pandas as pd
import numpy as np
import shutil
import argparse
from wildlife_datasets.datasets import MacaqueFaces, Cows2021v2, LeopardID2022


def enhanced_federated_partition(dataset_name, num_clients, test_ratio=0.2, alpha=0.1,
                                output_dir="federated_clients_enhanced", random_seed=42,
                                min_samples_per_identity=5):
    """
    Create enhanced federated partition with test/train split and Dirichlet distribution.
    
    Args:
        dataset_name: Name of the dataset ('LeopardID2022', 'MacaqueFaces', 'Cows2021v2')
        num_clients: Number of clients for federated learning
        test_ratio: Proportion of data to use for testing (0.0-1.0)
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        output_dir: Output directory name
        random_seed: Random seed for reproducibility
        min_samples_per_identity: Minimum samples required per identity to include
    
    Returns:
        dict: Statistics about the partition
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load dataset
    print(f"Loading {dataset_name}...")
    if dataset_name == "LeopardID2022":
        df = LeopardID2022('/home/wellvw12/leopard').df
    elif dataset_name == "MacaqueFaces":
        df = MacaqueFaces().df
    elif dataset_name == "Cows2021v2":
        df = Cows2021v2().df
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    print(f"Original dataset size: {len(df)} samples")
    
    # Filter out identities with insufficient samples
    identity_counts = df['identity'].value_counts()
    valid_identities = identity_counts[identity_counts >= min_samples_per_identity].index.tolist()
    
    # Remove 'unknown' identity if present
    if 'unknown' in valid_identities:
        valid_identities.remove('unknown')
    
    df_filtered = df[df['identity'].isin(valid_identities)].copy()
    print(f"After filtering (min {min_samples_per_identity} samples per identity): {len(df_filtered)} samples, {len(valid_identities)} identities")
    
    if len(valid_identities) < num_clients:
        raise ValueError(f"Not enough identities ({len(valid_identities)}) for {num_clients} clients")
    
    # Step 1: Split identities into test and training sets (no overlap)
    np.random.shuffle(valid_identities)
    
    # Calculate number of identities for test set
    num_test_identities = max(1, int(len(valid_identities) * test_ratio))
    
    test_identities = valid_identities[:num_test_identities]
    train_identities = valid_identities[num_test_identities:]
    
    print(f"\nIdentity split:")
    print(f"  Test identities: {len(test_identities)}")
    print(f"  Training identities: {len(train_identities)}")
    
    # Create test and train dataframes
    test_df = df_filtered[df_filtered['identity'].isin(test_identities)].copy()
    train_df = df_filtered[df_filtered['identity'].isin(train_identities)].copy()
    
    print(f"\nData split:")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Training samples: {len(train_df)}")
    
    # Verify no overlap
    test_ids_set = set(test_identities)
    train_ids_set = set(train_identities)
    overlap = test_ids_set.intersection(train_ids_set)
    if overlap:
        raise ValueError(f"Found overlapping identities between test and train: {overlap}")
    
    print("✅ Verified no identity overlap between test and training sets")
    
    # Step 2: Apply Dirichlet distribution to training identities (following federated_partition.py approach)
    print(f"\nApplying Dirichlet distribution (alpha={alpha}) to training data...")
    
    # Apply Dirichlet partitioning to training identities
    # client_proportions = np.random.dirichlet([alpha] * num_clients)
    client_proportions = np.array([0.5,0.35,0.15])
    print(f"Client proportions: {[f'{p:.3f}' for p in client_proportions]}")
    
    # Calculate target number of identities per client
    total_train_identities = len(train_identities)
    target_ids_per_client = (client_proportions * total_train_identities).astype(int)
    
    # Adjust for rounding errors
    remaining_ids = total_train_identities - np.sum(target_ids_per_client)
    for i in range(remaining_ids):
        target_ids_per_client[i % num_clients] += 1
    
    print(f"Target identities per client: {target_ids_per_client}")
    
    # Allocate identities to clients
    np.random.shuffle(train_identities)
    client_identities = [[] for _ in range(num_clients)]
    
    current_idx = 0
    for client_id in range(num_clients):
        num_ids_for_client = target_ids_per_client[client_id]
        client_identities[client_id] = train_identities[current_idx:current_idx + num_ids_for_client]
        current_idx += num_ids_for_client
        print(f"  Client {client_id}: {num_ids_for_client} identities")
    
    # Ensure each client has at least one identity
    for client_id in range(num_clients):
        if len(client_identities[client_id]) == 0:
            max_client = max(range(num_clients), key=lambda x: len(client_identities[x]))
            if len(client_identities[max_client]) > 1:
                identity_to_move = client_identities[max_client].pop()
                client_identities[client_id].append(identity_to_move)
                print(f"  Moved identity to client {client_id}")
    
    # Convert to dictionary format for compatibility with rest of code
    identity_client_assignments = {}
    for client_id in range(num_clients):
        for identity in client_identities[client_id]:
            identity_client_assignments[identity] = client_id
    
    print("✅ All clients have at least one identity")
    
    # Step 3: Create output directory structure
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Step 4: Save test data (single file outside client folders)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Saved test data: {len(test_df)} samples to {output_dir}/test.csv")
    
    # Step 5: Create client training data in numbered folders
    client_stats = []
    
    for client_id in range(num_clients):
        client_dir = os.path.join(output_dir, str(client_id))
        os.makedirs(client_dir)
        
        # Get identities assigned to this client
        client_identities = [identity for identity, assigned_client in identity_client_assignments.items() 
                           if assigned_client == client_id]
        
        # Create training data for this client
        if client_identities:
            client_train_data = train_df[train_df['identity'].isin(client_identities)].copy()
            client_train_data.to_csv(os.path.join(client_dir, 'train.csv'), index=False)
        else:
            # Create empty train file if no data (shouldn't happen after redistribution)
            pd.DataFrame(columns=df.columns).to_csv(os.path.join(client_dir, 'train.csv'), index=False)
            client_train_data = pd.DataFrame()
        
        # Store statistics
        client_stats.append({
            'client_id': client_id,
            'train_samples': len(client_train_data),
            'identities': len(client_identities),
            'identity_list': client_identities
        })
    
    # Step 6: Display comprehensive statistics
    print(f"\n" + "="*60)
    print(f"FEDERATED PARTITION SUMMARY")
    print(f"="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Total identities: {len(valid_identities)}")
    print(f"Test identities: {len(test_identities)} ({test_ratio*100:.1f}%)")
    print(f"Training identities: {len(train_identities)} ({(1-test_ratio)*100:.1f}%)")
    print(f"Test samples: {len(test_df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Number of clients: {num_clients}")
    print(f"Dirichlet alpha: {alpha}")
    print(f"Random seed: {random_seed}")
    
    print(f"\n" + "-"*60)
    print(f"CLIENT DISTRIBUTION:")
    print(f"-"*60)
    print(f"{'Client':<8} {'Samples':<10} {'IDs':<8} {'Percentage':<12}")
    print(f"-"*60)
    
    total_train_samples = sum(stats['train_samples'] for stats in client_stats)
    total_train_identities = sum(stats['identities'] for stats in client_stats)
    
    for stats in client_stats:
        percentage = (stats['train_samples'] / total_train_samples * 100) if total_train_samples > 0 else 0
        print(f"{stats['client_id']:<8} {stats['train_samples']:<10} {stats['identities']:<8} {percentage:<12.1f}%")
    
    print(f"-"*60)
    print(f"{'TOTAL:':<8} {total_train_samples:<10} {total_train_identities:<8} {'100.0%':<12}")
    print(f"="*60)
    
    # Calculate and display heterogeneity metrics
    sample_distribution = [stats['train_samples'] for stats in client_stats]
    id_distribution = [stats['identities'] for stats in client_stats]
    
    sample_std = np.std(sample_distribution)
    sample_cv = sample_std / np.mean(sample_distribution) if np.mean(sample_distribution) > 0 else 0
    
    id_std = np.std(id_distribution)
    id_cv = id_std / np.mean(id_distribution) if np.mean(id_distribution) > 0 else 0
    
    print(f"\nHETEROGENEITY METRICS:")
    print(f"Sample distribution CV: {sample_cv:.3f}")
    print(f"Identity distribution CV: {id_cv:.3f}")
    print(f"(Lower CV = more heterogeneous)")
    
    # Return comprehensive statistics
    return {
        'dataset_name': dataset_name,
        'total_identities': len(valid_identities),
        'test_identities': len(test_identities),
        'train_identities': len(train_identities),
        'test_samples': len(test_df),
        'train_samples': len(train_df),
        'num_clients': num_clients,
        'alpha': alpha,
        'test_ratio': test_ratio,
        'client_stats': client_stats,
        'sample_cv': sample_cv,
        'identity_cv': id_cv,
        'output_dir': output_dir,
        'random_seed': random_seed
    }


def validate_partition(output_dir):
    """
    Validate the created partition for correctness.
    
    Args:
        output_dir: Directory containing the partition
    
    Returns:
        bool: True if partition is valid
    """
    print(f"\n🔍 Validating partition in {output_dir}...")
    
    # Check if test.csv exists
    test_file = os.path.join(output_dir, 'test.csv')
    if not os.path.exists(test_file):
        print("❌ test.csv not found")
        return False
    
    # Load test data
    test_df = pd.read_csv(test_file)
    test_identities = set(test_df['identity'].unique())
    
    # Check client folders
    client_identities_combined = set()
    client_dirs = [d for d in os.listdir(output_dir) if d.isdigit()]
    
    for client_dir in sorted(client_dirs, key=int):
        train_file = os.path.join(output_dir, client_dir, 'train.csv')
        if os.path.exists(train_file):
            client_df = pd.read_csv(train_file)
            if len(client_df) > 0:
                client_ids = set(client_df['identity'].unique())
                client_identities_combined.update(client_ids)
    
    # Check for overlap
    overlap = test_identities.intersection(client_identities_combined)
    if overlap:
        print(f"❌ Found {len(overlap)} overlapping identities between test and training")
        return False
    
    print(f"✅ Validation passed:")
    print(f"   - Test identities: {len(test_identities)}")
    print(f"   - Training identities: {len(client_identities_combined)}")
    print(f"   - No overlap detected")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Enhanced Federated Data Partitioning')
    parser.add_argument('--dataset', default='LeopardID2022', 
                       choices=['LeopardID2022', 'MacaqueFaces', 'Cows2021v2'],
                       help='Dataset name')
    parser.add_argument('--num_clients', type=int, default=3, 
                       help='Number of federated clients')
    parser.add_argument('--test_ratio', type=float, default=0.3, 
                       help='Proportion of data for testing (0.0-1.0)')
    parser.add_argument('--alpha', type=float, default=0.9, 
                       help='Dirichlet concentration parameter (lower = more heterogeneous)')
    parser.add_argument('--output_dir', default='federated_clients_enhanced', 
                       help='Output directory')
    parser.add_argument('--random_seed', type=int, default=56, 
                       help='Random seed for reproducibility')
    parser.add_argument('--min_samples_per_identity', type=int, default=5, 
                       help='Minimum samples required per identity')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate partition after creation')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 <= args.test_ratio <= 1.0:
        raise ValueError("test_ratio must be between 0.0 and 1.0")
    
    if args.alpha <= 0:
        raise ValueError("alpha must be positive")
    
    if args.num_clients <= 0:
        raise ValueError("num_clients must be positive")
    
    # Create partition
    try:
        stats = enhanced_federated_partition(
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            test_ratio=args.test_ratio,
            alpha=args.alpha,
            output_dir=args.output_dir,
            random_seed=args.random_seed,
            min_samples_per_identity=args.min_samples_per_identity
        )
        
        print(f"\n✅ Partition created successfully in '{args.output_dir}'")
        
        # Validate if requested
        if args.validate:
            is_valid = validate_partition(args.output_dir)
            if not is_valid:
                print("❌ Validation failed!")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error creating partition: {e}")
        return 1


if __name__ == "__main__":
    exit(main())