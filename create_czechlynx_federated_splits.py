#!/usr/bin/env python3
"""
Script to create federated learning splits for Czech Lynx dataset
optimized for FedGDK knowledge distillation experiments.

Creates three clients:
- Client 0 (Beskydy): Geographic test region with diverse age groups
- Client 1 (NPS): Southern region with different environmental conditions  
- Client 2 (Sumava): Western region completing the geographic diversity

Each client gets its own metadata.csv file for WildlifeDataset integration.
"""

import pandas as pd
import os
from pathlib import Path
import argparse


def load_czechlynx_metadata(metadata_path: str) -> pd.DataFrame:
    """Load the original Czech Lynx metadata."""
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} total samples from {metadata_path}")
    return df


def create_federated_splits(df: pd.DataFrame) -> dict:
    """
    Create three federated clients based on geographic regions.
    
    Returns:
        dict: Dictionary with client_id -> dataframe mapping
    """
    clients = {}
    
    # Client 0: Beskydy region (using geo-aware test split)
    client_0 = df[df['split-geo_aware'] == 'test'].copy()
    clients['client_0_beskydy'] = client_0
    print(f"Client 0 (Beskydy): {len(client_0)} samples")
    
    # Client 1: NPS region 
    client_1 = df[df['source'] == 'nps'].copy()
    clients['client_1_nps'] = client_1
    print(f"Client 1 (NPS): {len(client_1)} samples")
    
    # Client 2: Sumava region
    client_2 = df[df['source'] == 'sumava'].copy()
    clients['client_2_sumava'] = client_2
    print(f"Client 2 (Sumava): {len(client_2)} samples")
    
    # Print distribution statistics
    print("\n=== FEDERATED SPLIT STATISTICS ===")
    total_samples = sum(len(client_df) for client_df in clients.values())
    print(f"Total samples distributed: {total_samples}")
    print(f"Original total: {len(df)}")
    
    for client_name, client_df in clients.items():
        unique_individuals = client_df['unique_name'].nunique()
        age_range = f"{client_df['relative_age'].min()}-{client_df['relative_age'].max()}"
        print(f"{client_name}: {len(client_df)} samples, {unique_individuals} individuals, ages {age_range}")
    
    return clients


def create_wildlife_metadata(client_df: pd.DataFrame, client_name: str) -> pd.DataFrame:
    """
    Convert client dataframe to WildlifeDataset format.
    
    Required columns: image_id, identity, path
    Optional: various metadata columns
    """
    wildlife_df = pd.DataFrame({
        'image_id': range(len(client_df)),
        'identity': client_df['unique_name'].values,
        'path': client_df['path'].values,
        'source_region': client_df['source'].values,
        'date': client_df['date'].values,
        'relative_age': client_df['relative_age'].values,
        'location': client_df['location'].values,
        'latitude': client_df['latitude'].values,
        'longitude': client_df['longitude'].values,
        'encounter': client_df['encounter'].values,
        'client_id': client_name
    })
    
    return wildlife_df


def save_client_metadata(clients: dict, output_dir: str):
    """Save metadata CSV files for each client."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== SAVING CLIENT METADATA TO {output_dir} ===")
    
    for client_name, client_df in clients.items():
        # Create WildlifeDataset compatible metadata
        wildlife_df = create_wildlife_metadata(client_df, client_name)
        
        # Save to CSV
        output_file = output_path / f"{client_name}_metadata.csv"
        wildlife_df.to_csv(output_file, index=False)
        print(f"Saved {len(wildlife_df)} samples to {output_file}")
        
        # Print sample of the data
        print(f"\nSample from {client_name}:")
        print(wildlife_df[['image_id', 'identity', 'path', 'source_region', 'relative_age']].head(3))


def analyze_knowledge_distribution(clients: dict):
    """Analyze the knowledge distribution across clients for distillation potential."""
    print("\n=== KNOWLEDGE DISTRIBUTION ANALYSIS ===")
    
    all_individuals = set()
    for client_df in clients.values():
        all_individuals.update(client_df['unique_name'].unique())
    
    print(f"Total unique individuals across all clients: {len(all_individuals)}")
    
    # Check for individual overlap (should be minimal for good non-IID)
    for i, (client1_name, client1_df) in enumerate(clients.items()):
        for j, (client2_name, client2_df) in enumerate(clients.items()):
            if i < j:  # Avoid duplicate comparisons
                overlap = set(client1_df['unique_name']) & set(client2_df['unique_name'])
                print(f"Individual overlap {client1_name} <-> {client2_name}: {len(overlap)} individuals")
    
    # Age distribution analysis
    print("\nAge distribution per client:")
    for client_name, client_df in clients.items():
        age_dist = client_df['relative_age'].value_counts().sort_index()
        print(f"{client_name}: {dict(age_dist)}")
    
    # Geographic diversity
    print("\nGeographic diversity:")
    for client_name, client_df in clients.items():
        locations = client_df['location'].nunique()
        coord_range_lat = client_df['latitude'].max() - client_df['latitude'].min() if not client_df['latitude'].isna().all() else 0
        coord_range_lon = client_df['longitude'].max() - client_df['longitude'].min() if not client_df['longitude'].isna().all() else 0
        print(f"{client_name}: {locations} locations, lat_range={coord_range_lat:.3f}, lon_range={coord_range_lon:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Create federated splits for Czech Lynx dataset')
    parser.add_argument('--metadata_path', 
                       default='/home/wellvw12/.cache/kagglehub/datasets/picekl/czechlynx/versions/6/metadata.csv',
                       help='Path to original Czech Lynx metadata.csv')
    parser.add_argument('--output_dir', 
                       default='/home/wellvw12/fedwild/czechlynx_federated',
                       help='Output directory for client metadata files')
    
    args = parser.parse_args()
    
    # Load original metadata
    df = load_czechlynx_metadata(args.metadata_path)
    
    # Create federated splits
    clients = create_federated_splits(df)
    
    # Analyze knowledge distribution
    analyze_knowledge_distribution(clients)
    
    # Save client metadata files
    save_client_metadata(clients, args.output_dir)
    
    print(f"\n=== FEDERATED SPLIT CREATION COMPLETE ===")
    print(f"Client metadata files saved to: {args.output_dir}")
    print(f"Ready for FedGDK knowledge distillation experiments!")


if __name__ == "__main__":
    main()