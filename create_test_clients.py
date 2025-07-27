#!/usr/bin/env python3
"""
Test Client Creator for Enhanced Federated Partition
Creates query and gallery sets for specified number of clients from test.csv
"""

import os
import pandas as pd
import numpy as np
import argparse


def create_test_clients(enhanced_partition_dir, num_clients, query_set_size, 
                       query_samples_per_id, gallery_samples_per_id, 
                       output_dir="test_clients", random_seed=42):
    """
    Create query and gallery sets for test clients from enhanced federated partition.
    
    Args:
        enhanced_partition_dir: Directory containing the enhanced federated partition
        num_clients: Number of test clients to create
        query_set_size: Total size of query set per client
        query_samples_per_id: Number of samples per identity in query set
        gallery_samples_per_id: Number of samples per identity in gallery set
        output_dir: Output directory for test clients
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: Statistics about the created test clients
    """
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Load test data from enhanced partition
    test_file = os.path.join(enhanced_partition_dir, 'test.csv')
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    print(f"Loading test data from {test_file}...")
    test_df = pd.read_csv(test_file)
    print(f"Loaded {len(test_df)} test samples")
    
    # Group by identity
    identity_groups = test_df.groupby('identity')
    available_identities = list(identity_groups.groups.keys())
    
    print(f"Available identities: {len(available_identities)}")
    
    # Calculate minimum samples needed per identity
    min_samples_per_id = query_samples_per_id + gallery_samples_per_id
    
    # Filter identities with sufficient samples
    valid_identities = []
    for identity in available_identities:
        identity_samples = len(identity_groups.get_group(identity))
        if identity_samples >= min_samples_per_id:
            valid_identities.append(identity)
    
    print(f"Valid identities (>= {min_samples_per_id} samples): {len(valid_identities)}")
    
    if len(valid_identities) == 0:
        raise ValueError("No identities have enough samples for query and gallery sets")
    
    # Calculate how many identities we need for query set
    max_query_identities = query_set_size // query_samples_per_id
    actual_query_identities = min(max_query_identities, len(valid_identities))
    
    print(f"Query identities per client: {actual_query_identities}")
    print(f"Query samples per identity: {query_samples_per_id}")
    print(f"Gallery samples per identity: {gallery_samples_per_id}")
    
    # Create output directory
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    client_stats = []
    
    # Create test clients
    for client_id in range(num_clients):
        print(f"\nCreating client {client_id}...")
        
        client_dir = os.path.join(output_dir, str(client_id))
        os.makedirs(client_dir)
        
        # Randomly select identities for this client
        np.random.shuffle(valid_identities)
        client_identities = valid_identities[:actual_query_identities]
        
        query_data = []
        gallery_data = []
        
        total_query_samples = 0
        total_gallery_samples = 0
        
        for identity in client_identities:
            identity_df = identity_groups.get_group(identity)
            identity_samples = identity_df.sample(frac=1, random_state=random_seed + client_id).reset_index(drop=True)
            
            # Split into query and gallery
            query_samples = identity_samples.iloc[:query_samples_per_id]
            gallery_samples = identity_samples.iloc[query_samples_per_id:query_samples_per_id + gallery_samples_per_id]
            
            query_data.append(query_samples)
            gallery_data.append(gallery_samples)
            
            total_query_samples += len(query_samples)
            total_gallery_samples += len(gallery_samples)
        
        # Combine data
        if query_data:
            query_df = pd.concat(query_data, ignore_index=True)
            gallery_df = pd.concat(gallery_data, ignore_index=True)
        else:
            query_df = pd.DataFrame(columns=test_df.columns)
            gallery_df = pd.DataFrame(columns=test_df.columns)
        
        # Verify all query identities are in gallery
        if len(query_df) > 0 and len(gallery_df) > 0:
            query_identities = set(query_df['identity'].unique())
            gallery_identities = set(gallery_df['identity'].unique())
            
            if not query_identities.issubset(gallery_identities):
                missing_ids = query_identities - gallery_identities
                print(f"⚠️  Warning: Query identities missing from gallery: {missing_ids}")
            else:
                print(f"✅ All {len(query_identities)} query identities present in gallery")
        
        # Shuffle the sets
        query_df = query_df.sample(frac=1, random_state=random_seed + client_id).reset_index(drop=True)
        gallery_df = gallery_df.sample(frac=1, random_state=random_seed + client_id + 1000).reset_index(drop=True)
        
        # Save to CSV files
        query_file = os.path.join(client_dir, 'query.csv')
        gallery_file = os.path.join(client_dir, 'gallery.csv')
        
        query_df.to_csv(query_file, index=False)
        gallery_df.to_csv(gallery_file, index=False)
        
        print(f"  Query: {len(query_df)} samples, {len(client_identities)} identities")
        print(f"  Gallery: {len(gallery_df)} samples, {len(client_identities)} identities")
        
        # Store statistics
        client_stats.append({
            'client_id': client_id,
            'query_samples': len(query_df),
            'gallery_samples': len(gallery_df),
            'identities': len(client_identities),
            'identity_list': client_identities
        })
    
    # Display summary
    print(f"\n" + "="*60)
    print(f"TEST CLIENT CREATION SUMMARY")
    print(f"="*60)
    print(f"Source: {enhanced_partition_dir}/test.csv")
    print(f"Total test samples: {len(test_df)}")
    print(f"Available identities: {len(available_identities)}")
    print(f"Valid identities: {len(valid_identities)}")
    print(f"Number of clients: {num_clients}")
    print(f"Query samples per identity: {query_samples_per_id}")
    print(f"Gallery samples per identity: {gallery_samples_per_id}")
    
    print(f"\n" + "-"*60)
    print(f"CLIENT STATISTICS:")
    print(f"-"*60)
    print(f"{'Client':<8} {'Query':<10} {'Gallery':<10} {'IDs':<8}")
    print(f"-"*60)
    
    for stats in client_stats:
        print(f"{stats['client_id']:<8} {stats['query_samples']:<10} {stats['gallery_samples']:<10} {stats['identities']:<8}")
    
    print(f"="*60)
    
    # Verify no overlap between query and gallery within each client
    print(f"\n🔍 Verifying query/gallery separation...")
    all_valid = True
    
    for client_id in range(num_clients):
        query_file = os.path.join(output_dir, str(client_id), 'query.csv')
        gallery_file = os.path.join(output_dir, str(client_id), 'gallery.csv')
        
        query_df = pd.read_csv(query_file)
        gallery_df = pd.read_csv(gallery_file)
        
        # Check for sample overlap (by file path if available, or by index)
        if 'path' in query_df.columns:
            query_paths = set(query_df['path'])
            gallery_paths = set(gallery_df['path'])
            overlap = query_paths.intersection(gallery_paths)
            
            if overlap:
                print(f"❌ Client {client_id}: Found {len(overlap)} overlapping samples")
                all_valid = False
            else:
                print(f"✅ Client {client_id}: No sample overlap")
        else:
            print(f"⚠️  Client {client_id}: Cannot verify sample overlap (no path column)")
    
    if all_valid:
        print("✅ All clients have properly separated query and gallery sets")
    
    return {
        'enhanced_partition_dir': enhanced_partition_dir,
        'output_dir': output_dir,
        'num_clients': num_clients,
        'total_test_samples': len(test_df),
        'available_identities': len(available_identities),
        'valid_identities': len(valid_identities),
        'query_samples_per_id': query_samples_per_id,
        'gallery_samples_per_id': gallery_samples_per_id,
        'client_stats': client_stats,
        'random_seed': random_seed
    }


def main():
    parser = argparse.ArgumentParser(description='Create Test Clients from Enhanced Federated Partition')
    parser.add_argument('--enhanced_partition_dir', default='data',
                       help='Directory containing enhanced federated partition')
    parser.add_argument('--num_clients', type=int, default=1,
                       help='Number of test clients to create')
    parser.add_argument('--query_set_size', type=int, default=35,
                       help='Total size of query set per client')
    parser.add_argument('--query_samples_per_id', type=int, default=5,
                       help='Number of samples per identity in query set')
    parser.add_argument('--gallery_samples_per_id', type=int, default=8,
                       help='Number of samples per identity in gallery set')
    parser.add_argument('--output_dir', default='test_clients',
                       help='Output directory for test clients')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_clients <= 0:
        raise ValueError("num_clients must be positive")
    
    if args.query_set_size <= 0:
        raise ValueError("query_set_size must be positive")
    
    if args.query_samples_per_id <= 0:
        raise ValueError("query_samples_per_id must be positive")
    
    if args.gallery_samples_per_id <= 0:
        raise ValueError("gallery_samples_per_id must be positive")
    
    # Create test clients
    try:
        create_test_clients(
            enhanced_partition_dir=args.enhanced_partition_dir,
            num_clients=args.num_clients,
            query_set_size=args.query_set_size,
            query_samples_per_id=args.query_samples_per_id,
            gallery_samples_per_id=args.gallery_samples_per_id,
            output_dir=args.output_dir,
            random_seed=args.random_seed
        )
        
        print(f"\n✅ Test clients created successfully in '{args.output_dir}'")
        return 0
        
    except Exception as e:
        print(f"❌ Error creating test clients: {e}")
        return 1


if __name__ == "__main__":
    exit(main())