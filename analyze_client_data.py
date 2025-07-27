#!/usr/bin/env python3
"""
Analyze Client Data Structure
Analyzes numbered folders containing train.csv, query.csv, and gallery.csv
Provides statistics on identity overlap and data distribution
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def load_client_data(data_dir):
    """Load all client data from numbered folders"""
    print(f"Loading client data from {data_dir}")
    
    clients = {}
    client_dirs = []
    
    # Find all numbered client directories
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            client_dirs.append((int(item), item_path))
    
    client_dirs.sort()  # Sort by client ID
    
    if not client_dirs:
        raise ValueError(f"No numbered client directories found in {data_dir}")
    
    print(f"Found {len(client_dirs)} client directories: {[str(cid) for cid, _ in client_dirs]}")
    
    # Load data for each client
    for client_id, client_path in client_dirs:
        client_data = {}
        
        # Check for required CSV files
        required_files = ['train.csv', 'query.csv', 'gallery.csv']
        missing_files = []
        
        for file_name in required_files:
            file_path = os.path.join(client_path, file_name)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    client_data[file_name.replace('.csv', '')] = df
                    print(f"  Client {client_id} {file_name}: {len(df)} samples")
                except Exception as e:
                    print(f"  Warning: Failed to load {file_path}: {e}")
                    missing_files.append(file_name)
            else:
                missing_files.append(file_name)
        
        if missing_files:
            print(f"  Warning: Client {client_id} missing files: {missing_files}")
        
        if client_data:  # Only add clients with some data
            clients[client_id] = client_data
    
    print(f"Successfully loaded data for {len(clients)} clients\n")
    return clients

def analyze_identity_overlap(clients):
    """Analyze identity overlap within and across clients"""
    print("IDENTITY OVERLAP ANALYSIS")
    print("=" * 50)
    
    # Track all identities across clients and splits
    all_train_ids = set()
    all_query_ids = set()
    all_gallery_ids = set()
    
    client_train_ids = {}
    client_query_ids = {}
    client_gallery_ids = {}
    
    # Per-client analysis
    client_stats = []
    
    for client_id, data in clients.items():
        print(f"\nClient {client_id}:")
        
        # Extract identities for each split
        train_ids = set(data.get('train', pd.DataFrame()).get('identity', []))
        query_ids = set(data.get('query', pd.DataFrame()).get('identity', []))
        gallery_ids = set(data.get('gallery', pd.DataFrame()).get('identity', []))
        
        # Remove any NaN/None values
        train_ids = {id for id in train_ids if pd.notna(id)}
        query_ids = {id for id in query_ids if pd.notna(id)}
        gallery_ids = {id for id in gallery_ids if pd.notna(id)}
        
        client_train_ids[client_id] = train_ids
        client_query_ids[client_id] = query_ids
        client_gallery_ids[client_id] = gallery_ids
        
        # Update global sets
        all_train_ids.update(train_ids)
        all_query_ids.update(query_ids)
        all_gallery_ids.update(gallery_ids)
        
        # Within-client overlap analysis
        train_query_overlap = train_ids & query_ids
        train_gallery_overlap = train_ids & gallery_ids
        query_gallery_overlap = query_ids & gallery_ids
        
        print(f"  Train identities: {len(train_ids)}")
        print(f"  Query identities: {len(query_ids)}")
        print(f"  Gallery identities: {len(gallery_ids)}")
        print(f"  Train ∩ Query: {len(train_query_overlap)} {'✓' if len(train_query_overlap) == 0 else '✗'}")
        print(f"  Train ∩ Gallery: {len(train_gallery_overlap)} {'✓' if len(train_gallery_overlap) == 0 else '✗'}")
        print(f"  Query ∩ Gallery: {len(query_gallery_overlap)} {'✓' if len(query_gallery_overlap) > 0 else '✗'}")
        
        # Store stats for summary
        client_stats.append({
            'client_id': client_id,
            'train_ids': len(train_ids),
            'query_ids': len(query_ids),
            'gallery_ids': len(gallery_ids),
            'train_query_overlap': len(train_query_overlap),
            'train_gallery_overlap': len(train_gallery_overlap),
            'query_gallery_overlap': len(query_gallery_overlap),
            'valid_split': len(train_query_overlap) == 0 and len(train_gallery_overlap) == 0 and len(query_gallery_overlap) > 0
        })
        
        if train_query_overlap:
            print(f"    ⚠️  Train-Query overlap: {list(train_query_overlap)[:5]}{'...' if len(train_query_overlap) > 5 else ''}")
        if train_gallery_overlap:
            print(f"    ⚠️  Train-Gallery overlap: {list(train_gallery_overlap)[:5]}{'...' if len(train_gallery_overlap) > 5 else ''}")
    
    # Cross-client analysis
    print(f"\nCROSS-CLIENT ANALYSIS:")
    print(f"  Total unique train identities: {len(all_train_ids)}")
    print(f"  Total unique query identities: {len(all_query_ids)}")
    print(f"  Total unique gallery identities: {len(all_gallery_ids)}")
    print(f"  Global train ∩ query: {len(all_train_ids & all_query_ids)}")
    print(f"  Global train ∩ gallery: {len(all_train_ids & all_gallery_ids)}")
    print(f"  Total unique identities: {len(all_train_ids | all_query_ids | all_gallery_ids)}")
    
    # Check for identity overlap between clients
    print(f"\nCLIENT IDENTITY OVERLAP:")
    client_ids = list(clients.keys())
    
    for i, client_a in enumerate(client_ids):
        for j, client_b in enumerate(client_ids[i+1:], i+1):
            train_overlap = client_train_ids[client_a] & client_train_ids[client_b]
            query_overlap = client_query_ids[client_a] & client_query_ids[client_b]
            gallery_overlap = client_gallery_ids[client_a] & client_gallery_ids[client_b]
            
            total_overlap = train_overlap | query_overlap | gallery_overlap
            
            if total_overlap:
                print(f"  Client {client_a} ∩ Client {client_b}: {len(total_overlap)} identities")
                print(f"    Train: {len(train_overlap)}, Query: {len(query_overlap)}, Gallery: {len(gallery_overlap)}")
    
    return client_stats, {
        'all_train_ids': all_train_ids,
        'all_query_ids': all_query_ids,
        'all_gallery_ids': all_gallery_ids,
        'client_train_ids': client_train_ids,
        'client_query_ids': client_query_ids,
        'client_gallery_ids': client_gallery_ids
    }

def analyze_data_distribution(clients):
    """Analyze data distribution across clients"""
    print(f"\nDATA DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Sample counts per client
    client_samples = []
    
    for client_id, data in clients.items():
        train_samples = len(data.get('train', pd.DataFrame()))
        query_samples = len(data.get('query', pd.DataFrame()))
        gallery_samples = len(data.get('gallery', pd.DataFrame()))
        total_samples = train_samples + query_samples + gallery_samples
        
        client_samples.append({
            'client_id': client_id,
            'train': train_samples,
            'query': query_samples,
            'gallery': gallery_samples,
            'total': total_samples
        })
    
    # Create summary table
    df_samples = pd.DataFrame(client_samples)
    
    print("Sample Distribution:")
    print(f"{'Client':<8} {'Train':<8} {'Query':<8} {'Gallery':<8} {'Total':<8}")
    print("-" * 45)
    
    for _, row in df_samples.iterrows():
        print(f"{row['client_id']:<8} {row['train']:<8} {row['query']:<8} {row['gallery']:<8} {row['total']:<8}")
    
    print("-" * 45)
    print(f"{'TOTAL:':<8} {df_samples['train'].sum():<8} {df_samples['query'].sum():<8} {df_samples['gallery'].sum():<8} {df_samples['total'].sum():<8}")
    
    # Statistics
    print(f"\nDistribution Statistics:")
    print(f"  Mean samples per client: {df_samples['total'].mean():.1f}")
    print(f"  Std samples per client: {df_samples['total'].std():.1f}")
    print(f"  Min samples per client: {df_samples['total'].min()}")
    print(f"  Max samples per client: {df_samples['total'].max()}")
    
    return df_samples

def analyze_identity_distribution(clients):
    """Analyze identity distribution per client"""
    print(f"\nIDENTITY DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    identity_stats = []
    
    for client_id, data in clients.items():
        # Count unique identities per split
        train_df = data.get('train', pd.DataFrame())
        query_df = data.get('query', pd.DataFrame())
        gallery_df = data.get('gallery', pd.DataFrame())
        
        train_ids = train_df['identity'].nunique() if 'identity' in train_df.columns else 0
        query_ids = query_df['identity'].nunique() if 'identity' in query_df.columns else 0
        gallery_ids = gallery_df['identity'].nunique() if 'identity' in gallery_df.columns else 0
        
        # Calculate samples per identity
        train_samples_per_id = len(train_df) / train_ids if train_ids > 0 else 0
        query_samples_per_id = len(query_df) / query_ids if query_ids > 0 else 0
        gallery_samples_per_id = len(gallery_df) / gallery_ids if gallery_ids > 0 else 0
        
        identity_stats.append({
            'client_id': client_id,
            'train_ids': train_ids,
            'query_ids': query_ids,
            'gallery_ids': gallery_ids,
            'train_samples_per_id': train_samples_per_id,
            'query_samples_per_id': query_samples_per_id,
            'gallery_samples_per_id': gallery_samples_per_id
        })
    
    df_identities = pd.DataFrame(identity_stats)
    
    print("Identity Distribution:")
    print(f"{'Client':<8} {'Train IDs':<10} {'Query IDs':<10} {'Gallery IDs':<12} {'Avg Train/ID':<12} {'Avg Query/ID':<12} {'Avg Gallery/ID':<12}")
    print("-" * 85)
    
    for _, row in df_identities.iterrows():
        print(f"{row['client_id']:<8} {row['train_ids']:<10} {row['query_ids']:<10} {row['gallery_ids']:<12} "
              f"{row['train_samples_per_id']:<12.1f} {row['query_samples_per_id']:<12.1f} {row['gallery_samples_per_id']:<12.1f}")
    
    return df_identities

def generate_summary_report(client_stats, identity_overlap_data, sample_distribution, identity_distribution):
    """Generate comprehensive summary report"""
    print(f"\nSUMMARY REPORT")
    print("=" * 50)
    
    # Overall health check
    valid_clients = sum(1 for stat in client_stats if stat['valid_split'])
    total_clients = len(client_stats)
    
    print(f"Dataset Health:")
    print(f"  Total clients analyzed: {total_clients}")
    print(f"  Clients with valid splits: {valid_clients}/{total_clients} ({100*valid_clients/total_clients:.1f}%)")
    
    # Identity separation check
    global_train_test_overlap = len(identity_overlap_data['all_train_ids'] & 
                                   (identity_overlap_data['all_query_ids'] | identity_overlap_data['all_gallery_ids']))
    
    print(f"  Global train-test identity separation: {'✓' if global_train_test_overlap == 0 else '✗'}")
    
    if global_train_test_overlap > 0:
        print(f"    ⚠️  {global_train_test_overlap} identities appear in both train and test sets")
    
    # Data distribution summary
    total_samples = sample_distribution['total'].sum()
    total_train = sample_distribution['train'].sum()
    total_query = sample_distribution['query'].sum()
    total_gallery = sample_distribution['gallery'].sum()
    
    print(f"\nData Distribution:")
    print(f"  Total samples: {total_samples}")
    print(f"  Train: {total_train} ({100*total_train/total_samples:.1f}%)")
    print(f"  Query: {total_query} ({100*total_query/total_samples:.1f}%)")
    print(f"  Gallery: {total_gallery} ({100*total_gallery/total_samples:.1f}%)")
    
    # Identity statistics
    total_unique_ids = len(identity_overlap_data['all_train_ids'] | 
                          identity_overlap_data['all_query_ids'] | 
                          identity_overlap_data['all_gallery_ids'])
    
    print(f"\nIdentity Statistics:")
    print(f"  Total unique identities: {total_unique_ids}")
    print(f"  Train identities: {len(identity_overlap_data['all_train_ids'])}")
    print(f"  Test identities: {len(identity_overlap_data['all_query_ids'] | identity_overlap_data['all_gallery_ids'])}")
    
    # Recommendations
    print(f"\nRecommendations:")
    if valid_clients < total_clients:
        print(f"  ⚠️  Fix identity overlap in {total_clients - valid_clients} clients")
    
    if global_train_test_overlap > 0:
        print(f"  ⚠️  Remove {global_train_test_overlap} overlapping identities from train/test sets")
    
    # Check for empty query-gallery overlaps
    clients_without_query_gallery_overlap = sum(1 for stat in client_stats if stat['query_gallery_overlap'] == 0)
    if clients_without_query_gallery_overlap > 0:
        print(f"  ⚠️  {clients_without_query_gallery_overlap} clients have no query-gallery identity overlap (required for ReID)")
    
    if valid_clients == total_clients and global_train_test_overlap == 0 and clients_without_query_gallery_overlap == 0:
        print(f"  ✅ Dataset appears to be properly structured for federated re-identification!")

def save_analysis_results(output_dir, client_stats, sample_distribution, identity_distribution):
    """Save analysis results to CSV files"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save client statistics
        pd.DataFrame(client_stats).to_csv(os.path.join(output_dir, 'client_overlap_stats.csv'), index=False)
        sample_distribution.to_csv(os.path.join(output_dir, 'sample_distribution.csv'), index=False)  
        identity_distribution.to_csv(os.path.join(output_dir, 'identity_distribution.csv'), index=False)
        
        print(f"\nAnalysis results saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Analyze client data structure and identity overlap')
    parser.add_argument('--data_dir', default='/home/wellvw12/baselines/baseline3.3.3',help='Directory containing numbered client folders')
    parser.add_argument('--output_dir', help='Output directory for analysis results (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist")
        sys.exit(1)
    
    try:
        # Load client data
        clients = load_client_data(args.data_dir)
        
        if not clients:
            print("No client data found!")
            sys.exit(1)
        
        # Perform analyses
        client_stats, identity_overlap_data = analyze_identity_overlap(clients)
        sample_distribution = analyze_data_distribution(clients)
        identity_distribution = analyze_identity_distribution(clients)
        
        # Generate summary
        generate_summary_report(client_stats, identity_overlap_data, sample_distribution, identity_distribution)
        
        # Save results if output directory specified
        if args.output_dir:
            save_analysis_results(args.output_dir, client_stats, sample_distribution, identity_distribution)
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()