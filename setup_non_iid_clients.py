#!/usr/bin/env python3
"""
Non-IID Client Distribution Setup for Federated ReID
Creates 7 clients with varying data distributions to demonstrate
how richer clients can dominate the global model.
"""

import pandas as pd
import numpy as np
import os
from collections import Counter
from wildlife_datasets.datasets import LeopardID2022

# Global configuration
TEST_PERCENTAGE = 0.2  # Percentage of data for testing (query + gallery)

def create_non_iid_distribution():
    """
    Create non-IID client distribution demonstrating rich vs poor client imbalance.
    
    Client Distribution Strategy:
    - Client 0-1: Rich clients (high identity diversity + many samples)
    - Client 2-3: Medium clients (moderate identity diversity)  
    - Client 4-6: Poor clients (low identity diversity + few samples)
    """
    
    # Load the dataset
    print("Loading LeopardID2022 dataset...")
    df = LeopardID2022('/home/wellvw12/leopard').df
    
    # Get identity counts and sort by frequency
    identity_counts = df['identity'].value_counts()
    print(f"Total identities: {len(identity_counts)}")
    print(f"Total samples: {len(df)}")
    
    # Remove 'unknown' identity if present
    if 'unknown' in identity_counts:
        df = df[df['identity'] != 'unknown']
        identity_counts = df['identity'].value_counts()
        print(f"After removing unknown: {len(identity_counts)} identities, {len(df)} samples")
    
    # Filter out identities with less than 8 samples
    valid_identities = identity_counts[identity_counts >= 8].index.tolist()
    df = df[df['identity'].isin(valid_identities)]
    identity_counts = df['identity'].value_counts()
    print(f"After filtering identities with <8 samples: {len(identity_counts)} identities, {len(df)} samples")
    
    # Sort identities by sample count (descending)
    sorted_identities = identity_counts.index.tolist()
    
    # Create client distributions
    clients = {}
    
    # Rich clients (0-1): Get top identities with most samples (further reduced)
    rich_identities_per_client = len(sorted_identities) // 24  # Top 4% of identities each
    
    clients[0] = sorted_identities[:rich_identities_per_client]
    clients[1] = sorted_identities[rich_identities_per_client:2*rich_identities_per_client]
    
    # Medium clients (2-3): Get middle-range identities (increased allocation)
    medium_start = 2 * rich_identities_per_client
    medium_identities_per_client = len(sorted_identities) // 10  # Increased from 15 to 10
    
    clients[2] = sorted_identities[medium_start:medium_start + medium_identities_per_client]
    clients[3] = sorted_identities[medium_start + medium_identities_per_client:
                                  medium_start + 2*medium_identities_per_client]
    
    # Poor clients (4-6): Get remaining identities with fewer samples
    poor_start = medium_start + 2*medium_identities_per_client
    remaining_identities = sorted_identities[poor_start:]
    
    # Distribute remaining identities among poor clients
    poor_chunk_size = len(remaining_identities) // 3
    clients[4] = remaining_identities[:poor_chunk_size]
    clients[5] = remaining_identities[poor_chunk_size:2*poor_chunk_size]
    clients[6] = remaining_identities[2*poor_chunk_size:]
    
    return df, clients, identity_counts

def create_train_query_gallery_split(client_df, client_type="Rich"):
    """
    Split client data into train, query, and gallery sets using global TEST_PERCENTAGE.
    
    Query/Gallery logic:
    - At least 1 query image with 2 gallery images per ID
    - If more test samples: 2 query images with more gallery images
    - Scale up proportionally
    - If only 1 test image per ID: place in gallery
    
    Args:
        client_df: DataFrame for a specific client
        client_type: Type of client (Rich, Medium, Poor)
    """
    train_data = []
    query_data = []
    gallery_data = []
    
    # Split by identity to ensure all sets have diverse identities
    for identity in client_df['identity'].unique():
        identity_data = client_df[client_df['identity'] == identity].copy()
        
        # Shuffle the data
        identity_data = identity_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n_samples = len(identity_data)
        
        # Calculate train/test split using global TEST_PERCENTAGE
        n_test = max(1, int(n_samples * TEST_PERCENTAGE))
        n_train = n_samples - n_test
        
        # Split into train and test
        train_data.append(identity_data[:n_train])
        test_data = identity_data[n_train:]
        
        # Handle query/gallery split based on test samples available
        if len(test_data) == 1:
            # Only 1 test image: place in gallery
            gallery_data.append(test_data)
        elif len(test_data) == 2:
            # 2 test images: 1 query, 1 gallery
            query_data.append(test_data[:1])
            gallery_data.append(test_data[1:])
        elif len(test_data) == 3:
            # 3 test images: 1 query, 2 gallery
            query_data.append(test_data[:1])
            gallery_data.append(test_data[1:])
        elif len(test_data) <= 5:
            # 4-5 test images: 2 query, rest gallery
            query_data.append(test_data[:2])
            gallery_data.append(test_data[2:])
        else:
            # More than 5 test images: scale up proportionally
            # Aim for 1:2 ratio (query:gallery) but ensure minimums
            n_query = max(2, len(test_data) // 3)
            n_gallery = len(test_data) - n_query
            
            query_data.append(test_data[:n_query])
            gallery_data.append(test_data[n_query:])
    
    # Combine all data
    train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
    query_df = pd.concat(query_data, ignore_index=True) if query_data else pd.DataFrame()
    gallery_df = pd.concat(gallery_data, ignore_index=True) if gallery_data else pd.DataFrame()
    
    return train_df, query_df, gallery_df

def main():
    """Main function to create non-IID client distributions."""
    
    # Create output directory
    output_dir = "client_data_non_iid"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create non-IID distribution
    df, clients, identity_counts = create_non_iid_distribution()
    
    # Generate client datasets
    client_stats = []
    
    for client_id in range(7):
        print(f"\nProcessing Client {client_id}...")
        
        # Get client identities
        client_identities = clients[client_id]
        client_df = df[df['identity'].isin(client_identities)].copy()
        
        # Calculate statistics
        n_identities = len(client_identities)
        n_samples = len(client_df)
        avg_samples_per_identity = n_samples / n_identities if n_identities > 0 else 0
        
        # Determine client type
        if client_id <= 1:
            client_type = "Rich"
        elif client_id <= 3:
            client_type = "Medium"
        else:
            client_type = "Poor"
        
        # Create train/query/gallery splits
        train_df, query_df, gallery_df = create_train_query_gallery_split(client_df, client_type)
        
        # Validation: Check if all query images are in gallery
        query_identities = set(query_df['identity'].unique()) if not query_df.empty else set()
        gallery_identities = set(gallery_df['identity'].unique()) if not gallery_df.empty else set()
        missing_in_gallery = query_identities - gallery_identities
        
        if missing_in_gallery:
            print(f"  WARNING: Query identities NOT in gallery: {missing_in_gallery}")
        else:
            print(f"  ✓ All query identities present in gallery")
        
        client_stats.append({
            'client_id': client_id,
            'client_type': client_type,
            'n_identities': n_identities,
            'n_samples': n_samples,
            'n_train': len(train_df),
            'n_query': len(query_df),
            'n_gallery': len(gallery_df),
            'avg_samples_per_identity': avg_samples_per_identity
        })
        
        # Save client data
        client_dir = os.path.join(output_dir, f"{client_id}")
        os.makedirs(client_dir, exist_ok=True)
        
        # Validate that all sets have data
        if len(query_df) == 0 or len(gallery_df) == 0:
            print(f"  WARNING: Client {client_id} has empty query or gallery set!")
            print(f"  Query: {len(query_df)}, Gallery: {len(gallery_df)}")
        
        train_df.to_csv(os.path.join(client_dir, "train.csv"), index=False)
        query_df.to_csv(os.path.join(client_dir, "query.csv"), index=False)
        gallery_df.to_csv(os.path.join(client_dir, "gallery.csv"), index=False)
        
        print(f"  Type: {client_type}")
        print(f"  Identities: {n_identities}")
        print(f"  Total samples: {n_samples}")
        print(f"  Train: {len(train_df)}, Query: {len(query_df)}, Gallery: {len(gallery_df)}")
        print(f"  Avg samples/identity: {avg_samples_per_identity:.1f}")
    
    # Create summary statistics
    stats_df = pd.DataFrame(client_stats)
    stats_df.to_csv(os.path.join(output_dir, "client_statistics.csv"), index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("CLIENT DISTRIBUTION SUMMARY")
    print("="*80)
    print(stats_df.to_string(index=False))
    
    # Calculate dominance metrics
    total_samples = stats_df['n_samples'].sum()
    rich_samples = stats_df[stats_df['client_type'] == 'Rich']['n_samples'].sum()
    poor_samples = stats_df[stats_df['client_type'] == 'Poor']['n_samples'].sum()
    
    print(f"\nDATA DISTRIBUTION ANALYSIS:")
    print(f"Rich clients (0-1): {rich_samples:,} samples ({rich_samples/total_samples*100:.1f}%)")
    print(f"Poor clients (4-6): {poor_samples:,} samples ({poor_samples/total_samples*100:.1f}%)")
    print(f"Rich/Poor ratio: {rich_samples/poor_samples:.2f}x")
    
    # Show detailed distribution
    print(f"\nDETAILED DISTRIBUTION:")
    for _, row in stats_df.iterrows():
        print(f"Client {row['client_id']} ({row['client_type']}): "
              f"{row['n_identities']} identities, "
              f"Train: {row['n_train']}, Query: {row['n_query']}, Gallery: {row['n_gallery']}, "
              f"Avg: {row['avg_samples_per_identity']:.1f} samples/identity")
    
    print(f"\nData saved to: {output_dir}/")
    print("This distribution will cause rich clients to dominate the global model!")

if __name__ == "__main__":
    main()