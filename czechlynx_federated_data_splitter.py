#!/usr/bin/env python3
"""
Czech Lynx Federated Data Splitter

Creates federated splits for Czech Lynx dataset with query/gallery/training
splits for each geographic client (Beskydy, NPS, Šumava).

Based on federated_data_splitter.py pattern but adapted for geographic-based
non-IID distribution optimal for FedGDK knowledge distillation.
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Czech Lynx Federated Data Splitter')
    parser.add_argument('--czechlynx_metadata', 
                       default='/home/wellvw12/fedwild/lynx_metadata.csv',
                       type=str, help='Path to Czech Lynx metadata CSV')
    parser.add_argument('--output_dir', 
                       default='/home/wellvw12/fedwild/czechlynx_federated', 
                       type=str, help='Output directory for client metadata')
    
    # Query/Gallery parameters
    parser.add_argument('--query_size', default=20, type=int, help='Number of query IDs per client')
    parser.add_argument('--samples_per_query_id', default=2, type=int, help='Number of samples per query ID')
    parser.add_argument('--samples_per_gallery_id', default=4, type=int, help='Number of samples per gallery ID')
    
    # Data filtering parameters
    def nullable_int(value):
        if value.lower() == 'none':
            return None
        return int(value)
    
    parser.add_argument('--min_samples_per_id', default=None, type=nullable_int, help='Minimum samples per ID for inclusion (None for no limit)')
    parser.add_argument('--max_samples_per_id', default=100, type=nullable_int, help='Maximum samples per ID in training (None for no limit)')
    parser.add_argument('--min_train_samples_per_id', default=2, type=nullable_int, help='Minimum samples per ID in training')
    parser.add_argument('--random_seed', default=42, type=int, help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_czechlynx_metadata(czechlynx_path):
    """Load and preprocess Czech Lynx metadata."""
    print(f"Loading Czech Lynx metadata from: {czechlynx_path}")
    
    if not os.path.exists(czechlynx_path):
        raise FileNotFoundError(f"Czech Lynx metadata not found: {czechlynx_path}")
    
    df = pd.read_csv(czechlynx_path)
    
    # Rename Czech Lynx specific columns to standard format
    if 'unique_name' in df.columns and 'identity' not in df.columns:
        df['identity'] = df['unique_name']
    
    print(f"Loaded {len(df)} samples with {df['identity'].nunique()} unique individuals")
    
    # Convert date format from DD-MM-YYYY to YYYY-MM-DD
    if 'date' in df.columns:
        def convert_date(date_str):
            if pd.isna(date_str) or date_str == '':
                return None
            try:
                day, month, year = date_str.split('-')
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            except:
                return None
        
        df['date'] = df['date'].apply(convert_date)
    
    # Add image_id column (required by WildlifeDataset)
    if 'image_id' not in df.columns:
        # Extract filename from path as image_id
        df['image_id'] = df['path'].apply(lambda x: os.path.basename(x) if pd.notna(x) else None)
    
    # Ensure path column exists (already present in Czech Lynx)
    
    return df


def split_by_geographic_clients(metadata):
    """Split data into geographic clients based on source regions."""
    print("\nSplitting data by geographic regions...")
    
    # Define client mappings based on original analysis
    client_splits = {
        'beskydy': metadata[metadata['split-geo_aware'] == 'test'].copy(),  # Northern/Eastern
        'nps': metadata[metadata['source'] == 'nps'].copy(),              # Southern  
        'sumava': metadata[metadata['source'] == 'sumava'].copy()          # Western
    }
    
    print(f"Geographic client distribution:")
    for client_name, client_data in client_splits.items():
        unique_individuals = client_data['identity'].nunique()
        print(f"  {client_name}: {len(client_data)} samples, {unique_individuals} individuals")
        
        # Check source regions within each client
        if 'source' in client_data.columns:
            sources = client_data['source'].value_counts()
            print(f"    Sources: {dict(sources)}")
    
    return client_splits


def filter_ids_by_sample_count(metadata, min_samples):
    """Filter to keep only IDs that have at least min_samples."""
    if min_samples is None:
        print(f"  No minimum sample filtering applied")
        return metadata
    
    id_counts = metadata['identity'].value_counts()
    valid_ids = id_counts[id_counts >= min_samples].index
    filtered_metadata = metadata[metadata['identity'].isin(valid_ids)].copy()
    print(f"  Filtered from {len(id_counts)} to {len(valid_ids)} IDs (≥{min_samples} samples)")
    return filtered_metadata


def limit_samples_per_id(metadata, max_samples_per_id, min_samples_per_id, random_seed=42):
    """Limit the number of samples per identity for training data."""
    if max_samples_per_id is None:
        print(f"  No maximum sample limit applied")
        return metadata
    
    np.random.seed(random_seed)
    limited_data = []
    
    for _, group in metadata.groupby('identity'):
        if len(group) > max_samples_per_id:
            group = group.sample(n=max_samples_per_id, random_state=random_seed)
        
        if min_samples_per_id is None or len(group) >= min_samples_per_id:
            limited_data.append(group)
    
    if limited_data:
        limited_metadata = pd.concat(limited_data, ignore_index=True)
    else:
        limited_metadata = pd.DataFrame()
    
    print(f"  Limited to max {max_samples_per_id} samples per ID: {len(limited_metadata)} samples")
    return limited_metadata


def create_query_gallery_splits(metadata, query_size, samples_per_query_id, samples_per_gallery_id, random_seed=42):
    """Create query and gallery splits from client metadata."""
    np.random.seed(random_seed)
    
    # Get IDs with sufficient samples for query+gallery
    required_samples = samples_per_query_id + samples_per_gallery_id
    id_counts = metadata['identity'].value_counts()
    eligible_ids = id_counts[id_counts >= required_samples].index.tolist()
    
    if len(eligible_ids) < query_size:
        print(f"    Warning: Only {len(eligible_ids)} IDs eligible, reducing query size from {query_size}")
        query_size = len(eligible_ids)
    
    # Randomly select IDs for query/gallery
    selected_ids = np.random.choice(eligible_ids, size=query_size, replace=False)
    
    query_data = []
    gallery_data = []
    
    for identity in selected_ids:
        id_samples = metadata[metadata['identity'] == identity].copy()
        id_samples = id_samples.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # Take samples for query and gallery
        query_samples = id_samples.head(samples_per_query_id)
        gallery_samples = id_samples.iloc[samples_per_query_id:samples_per_query_id + samples_per_gallery_id]
        
        query_data.append(query_samples)
        gallery_data.append(gallery_samples)
    
    # Combine query and gallery data
    query_metadata = pd.concat(query_data, ignore_index=True) if query_data else pd.DataFrame()
    gallery_metadata = pd.concat(gallery_data, ignore_index=True) if gallery_data else pd.DataFrame()
    
    # Get remaining data for training (completely excluding query/gallery IDs)
    # For proper train/test separation, remove ALL samples from query/gallery IDs
    remaining_metadata = metadata[~metadata['identity'].isin(selected_ids)].copy()
    
    print(f"    Query: {len(query_metadata)} samples ({query_metadata['identity'].nunique()} IDs)")
    print(f"    Gallery: {len(gallery_metadata)} samples ({gallery_metadata['identity'].nunique()} IDs)")
    print(f"    Training: {len(remaining_metadata)} samples ({remaining_metadata['identity'].nunique()} IDs)")
    
    return query_metadata, gallery_metadata, remaining_metadata


def create_client_metadata(client_name, query_df, gallery_df, train_df):
    """Create unified metadata for a client with split and client columns."""
    all_data = []
    
    # Add query data
    if len(query_df) > 0:
        query_split = query_df.copy()
        query_split['split'] = 'query'
        query_split['client'] = str(client_name)
        all_data.append(query_split)
    
    # Add gallery data
    if len(gallery_df) > 0:
        gallery_split = gallery_df.copy()
        gallery_split['split'] = 'gallery'
        gallery_split['client'] = str(client_name)
        all_data.append(gallery_split)
    
    # Add training data
    if len(train_df) > 0:
        train_split = train_df.copy()
        train_split['split'] = 'train'
        train_split['client'] = str(client_name)
        all_data.append(train_split)
    
    # Combine all data
    if all_data:
        unified_metadata = pd.concat(all_data, ignore_index=True)
    else:
        unified_metadata = pd.DataFrame()
    
    return unified_metadata


def save_combined_metadata(client_metadata_dict, output_dir):
    """Save combined metadata from all clients to a single file."""
    # Combine all client metadata
    all_client_data = []
    for client_name, client_metadata in client_metadata_dict.items():
        if len(client_metadata) > 0:
            all_client_data.append(client_metadata)
    
    if all_client_data:
        combined_metadata = pd.concat(all_client_data, ignore_index=True)
    else:
        combined_metadata = pd.DataFrame()
    
    # Save to single metadata file
    metadata_path = Path(output_dir) / "metadata.csv"
    combined_metadata.to_csv(metadata_path, index=False)
    
    print(f"\nSaved combined metadata: {metadata_path}")
    print(f"  Total samples: {len(combined_metadata)}")
    print(f"  Total clients: {len(client_metadata_dict)}")
    
    # Print distribution by client and split
    if len(combined_metadata) > 0:
        print(f"\nClient and split distribution:")
        for client_name in combined_metadata['client'].unique():
            client_data = combined_metadata[combined_metadata['client'] == client_name]
            print(f"  {client_name}: {len(client_data)} samples")
            
            if 'split' in client_data.columns:
                split_counts = client_data['split'].value_counts()
                for split_name, count in split_counts.items():
                    unique_ids = client_data[client_data['split'] == split_name]['identity'].nunique()
                    print(f"    {split_name}: {count} samples ({unique_ids} IDs)")
    
    return metadata_path


def create_federated_manager_example(output_dir):
    """Create example usage script for CzechLynxFederatedManager."""
    example_path = Path(output_dir) / "example_usage.py"
    
    example_code = '''#!/usr/bin/env python3
"""
Example usage of Czech Lynx Federated Dataset
"""

import pandas as pd

# Load the combined metadata
metadata = pd.read_csv('metadata.csv')

print(f"Total samples: {len(metadata)}")
print(f"Clients: {metadata['client'].unique()}")
print(f"Splits: {metadata['split'].unique()}")

# Example: Get training data for each client
for client_name in metadata['client'].unique():
    client_data = metadata[metadata['client'] == client_name]
    train_data = client_data[client_data['split'] == 'train']
    query_data = client_data[client_data['split'] == 'query'] 
    gallery_data = client_data[client_data['split'] == 'gallery']
    
    print(f"\\n{client_name.upper()} Client:")
    print(f"  Training: {len(train_data)} samples ({train_data['identity'].nunique()} IDs)")
    print(f"  Query: {len(query_data)} samples ({query_data['identity'].nunique()} IDs)")
    print(f"  Gallery: {len(gallery_data)} samples ({gallery_data['identity'].nunique()} IDs)")

# Example: Filter by specific client and split
beskydy_train = metadata[(metadata['client'] == 'beskydy') & (metadata['split'] == 'train')]
print(f"\\nBeskydy training data: {len(beskydy_train)} samples")
'''
    
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    print(f"Created example usage script: {example_path}")


def print_federation_summary(client_metadata_dict):
    """Print summary of the federated setup."""
    print(f"\n{'='*60}")
    print("CZECH LYNX FEDERATED DATASET SUMMARY")
    print(f"{'='*60}")
    
    total_samples = 0
    total_individuals = set()
    
    for client_name, client_df in client_metadata_dict.items():
        if len(client_df) > 0:
            individuals = set(client_df['identity'].unique())
            total_samples += len(client_df)
            total_individuals.update(individuals)
            
            print(f"\n{client_name.upper()} CLIENT:")
            print(f"  Total samples: {len(client_df)}")
            print(f"  Unique individuals: {len(individuals)}")
            
            if 'split' in client_df.columns:
                split_counts = client_df['split'].value_counts()
                for split_name in ['train', 'query', 'gallery']:
                    if split_name in split_counts:
                        count = split_counts[split_name]
                        split_df = client_df[client_df['split'] == split_name]
                        unique_ids = split_df['identity'].nunique()
                        print(f"    {split_name}: {count} samples ({unique_ids} IDs)")
    
    print(f"\nFEDERATION TOTALS:")
    print(f"  Total samples: {total_samples}")
    print(f"  Total unique individuals: {len(total_individuals)}")
    
    # Check for individual overlap between clients
    client_names = list(client_metadata_dict.keys())
    individual_overlaps = []
    
    for i in range(len(client_names)):
        for j in range(i+1, len(client_names)):
            client1_individuals = set(client_metadata_dict[client_names[i]]['identity'].unique())
            client2_individuals = set(client_metadata_dict[client_names[j]]['identity'].unique())
            overlap = len(client1_individuals & client2_individuals)
            individual_overlaps.append(f"{client_names[i]}-{client_names[j]}: {overlap}")
    
    print(f"  Individual overlaps: {', '.join(individual_overlaps)}")
    print(f"{'='*60}")


def main():
    args = parse_args()
    
    print("=== CZECH LYNX FEDERATED DATA SPLITTER ===")
    print(f"Czech Lynx metadata: {args.czechlynx_metadata}")
    print(f"Output directory: {args.output_dir}")
    print(f"Query size per client: {args.query_size} IDs")
    print(f"Samples per query ID: {args.samples_per_query_id}")
    print(f"Samples per gallery ID: {args.samples_per_gallery_id}")
    print(f"Random seed: {args.random_seed}")
    
    # Load Czech Lynx metadata
    czechlynx_metadata = load_czechlynx_metadata(args.czechlynx_metadata)
    
    # Split into geographic clients
    client_splits = split_by_geographic_clients(czechlynx_metadata)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each client
    client_metadata_dict = {}
    
    for client_name, client_data in client_splits.items():
        print(f"\nProcessing {client_name.upper()} client...")
        
        # Filter IDs by sample count
        if args.min_samples_per_id is not None:
            print(f"  Filtering IDs with ≥{args.min_samples_per_id} samples...")
        else:
            print(f"  No minimum sample filtering (all IDs included)...")
        filtered_data = filter_ids_by_sample_count(client_data, args.min_samples_per_id)
        
        if len(filtered_data) == 0:
            print(f"  Warning: No data remaining for {client_name} after filtering")
            continue
        
        # Limit samples per ID
        print(f"  Limiting samples per ID...")
        limited_data = limit_samples_per_id(
            filtered_data, 
            args.max_samples_per_id, 
            args.min_train_samples_per_id, 
            args.random_seed
        )
        
        if len(limited_data) == 0:
            print(f"  Warning: No data remaining for {client_name} after limiting")
            continue
        
        # Reset index for proper split creation
        limited_data = limited_data.reset_index(drop=True)
        
        # Create query/gallery/training splits
        print(f"  Creating query/gallery/training splits...")
        query_df, gallery_df, train_df = create_query_gallery_splits(
            limited_data,
            args.query_size,
            args.samples_per_query_id, 
            args.samples_per_gallery_id,
            args.random_seed
        )
        
        # Create unified client metadata
        client_metadata = create_client_metadata(client_name, query_df, gallery_df, train_df)
        client_metadata_dict[client_name] = client_metadata
    
    # Save combined metadata to single file
    metadata_path = save_combined_metadata(client_metadata_dict, args.output_dir)
    
    # Print federation summary
    print_federation_summary(client_metadata_dict)
    
    # Create example usage script
    create_federated_manager_example(args.output_dir)
    
    print(f"\n✅ Czech Lynx federated dataset splitting completed!")
    print(f"📁 Combined metadata saved to: {metadata_path}")
    print(f"📝 Example usage: {args.output_dir}/example_usage.py")
    
    # Save federation summary to file
    summary_path = Path(args.output_dir) / "federation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("CZECH LYNX FEDERATED DATASET CONFIGURATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated with parameters:\n")
        f.write(f"- Query size per client: {args.query_size} IDs\n")
        f.write(f"- Samples per query ID: {args.samples_per_query_id}\n")
        f.write(f"- Samples per gallery ID: {args.samples_per_gallery_id}\n")
        f.write(f"- Min samples per ID: {args.min_samples_per_id}\n")
        f.write(f"- Max samples per ID: {args.max_samples_per_id}\n")
        f.write(f"- Random seed: {args.random_seed}\n\n")
        
        f.write("DATA STRUCTURE:\n")
        f.write("- metadata.csv: Combined metadata for all clients\n")
        f.write("  - 'client' column: beskydy, nps, sumava (geographic regions)\n")
        f.write("  - 'split' column: train, query, gallery\n")
        f.write("  - 'identity' column: individual animal IDs\n\n")
        
        f.write("USAGE:\n")
        f.write("Load metadata.csv and filter by 'client' and 'split' columns\n")
        f.write("to access data for specific federated learning scenarios.\n")
    
    print(f"📄 Configuration summary: {summary_path}")


if __name__ == "__main__":
    main()