#!/usr/bin/env python3
"""
Exact Distribution Client Generator
Creates 4 clients with specific sample distributions:
Client 1: Rich A (~1400:50ids), Poor B (~10:2ids)
Client 2: Moderate A (~500:25ids), Moderate B (~100:10ids)  
Client 3: Medium A (~300:40ids), No B training, both A&B in test
Client 4: Poor A (~10:2ids), Rich B (~1200:50ids)
"""

import pandas as pd
import numpy as np
import os
import argparse
from collections import defaultdict

def analyze_dataframe(df, species_column='species', identity_column='identity'):
    """Analyze the input dataframe"""
    print(f"Total samples: {len(df)}")
    print(f"Species distribution:")
    species_counts = df[species_column].value_counts()
    print(species_counts)
    
    print(f"\nIdentity distribution by species:")
    for species in df[species_column].unique():
        species_df = df[df[species_column] == species]
        identity_counts = species_df[identity_column].value_counts()
        print(f"{species}: {len(identity_counts)} identities")
        print(f"  Avg samples per identity: {identity_counts.mean():.1f}")
        print(f"  Min/Max samples: {identity_counts.min()}/{identity_counts.max()}")
    
    return species_counts

def filter_identities_by_samples(df, species_column='species', identity_column='identity', min_samples=5):
    """Filter identities that have minimum samples"""
    print(f"\nFiltering identities with minimum {min_samples} samples...")
    
    filtered_records = []
    for species in df[species_column].unique():
        species_df = df[df[species_column] == species]
        identity_counts = species_df[identity_column].value_counts()
        
        # Keep only identities with minimum samples
        valid_identities = identity_counts[identity_counts >= min_samples].index
        species_filtered = species_df[species_df[identity_column].isin(valid_identities)]
        filtered_records.append(species_filtered)
        
        print(f"  {species}: {len(valid_identities)}/{len(identity_counts)} identities kept")
    
    filtered_df = pd.concat(filtered_records, ignore_index=True)
    print(f"After filtering: {len(filtered_df)} samples")
    return filtered_df

def select_identities_for_target(species_df, target_ids, target_samples, identity_column='identity', random_seed=42):
    """Select identities to approximately reach target number of IDs and samples"""
    np.random.seed(random_seed)
    
    # Get identity counts
    identity_counts = species_df[identity_column].value_counts()
    available_identities = identity_counts.index.tolist()
    np.random.shuffle(available_identities)
    
    selected_identities = []
    total_samples = 0
    
    # Select identities until we reach target
    for identity_id in available_identities:
        if len(selected_identities) >= target_ids:
            break
        if total_samples >= target_samples:
            break
            
        identity_sample_count = identity_counts[identity_id]
        selected_identities.append(identity_id)
        total_samples += identity_sample_count
    
    # If we haven't reached target samples but have IDs, continue adding
    remaining_identities = [id for id in available_identities if id not in selected_identities]
    for identity_id in remaining_identities:
        if total_samples >= target_samples:
            break
        identity_sample_count = identity_counts[identity_id]
        selected_identities.append(identity_id)
        total_samples += identity_sample_count
    
    selected_df = species_df[species_df[identity_column].isin(selected_identities)]
    actual_samples = len(selected_df)
    actual_ids = len(selected_identities)
    
    print(f"    Selected {actual_ids} IDs with {actual_samples} samples (target: {target_ids} IDs, {target_samples} samples)")
    
    return selected_df, selected_identities

def create_exact_distribution(df, species_column='species', identity_column='identity',
                             species_a_value='leopard', species_b_value='hyena',
                             output_dir='exact_distribution_clients', random_seed=42):
    """
    Create 4 clients with exact distributions:
    Client 1: ~1400 A (50 IDs), ~10 B (2 IDs)
    Client 2: ~500 A (25 IDs), ~100 B (10 IDs)
    Client 3: ~300 A (40 IDs), 0 B training, B in test
    Client 4: ~10 A (2 IDs), ~1200 B (50 IDs)
    """
    
    np.random.seed(random_seed)
    
    # Separate species
    species_a_df = df[df[species_column] == species_a_value].copy()
    species_b_df = df[df[species_column] == species_b_value].copy()
    
    print(f"\nSpecies separation:")
    print(f"Species A ({species_a_value}): {len(species_a_df)} samples, {species_a_df[identity_column].nunique()} IDs")
    print(f"Species B ({species_b_value}): {len(species_b_df)} samples, {species_b_df[identity_column].nunique()} IDs")
    
    # Target distributions
    targets = {
        # Client 1: Data-rich in A, minimal B (common real-world scenario)
        'client_1': {
            'a_samples': 1400, 'a_ids': 50,  # Dominant class
            'b_samples': 10,   'b_ids': 2,    # Very few B samples (hard to learn locally)
            'test_a': 20, 'test_b': 20,       # Balanced test (unlocks evaluation fairness)
            'gallery_a': 90, 'gallery_b': 90
        },
        # Client 2: Moderate A, moderate B (acts as a "bridge" for knowledge sharing)
        'client_2': {
            'a_samples': 500, 'a_ids': 25,
            'b_samples': 100, 'b_ids': 10,    # Enough B to contribute meaningfully
            'test_a': 30, 'test_b': 30,
            'gallery_a': 90, 'gallery_b': 90
        },
        # Client 3: Missing B entirely (tests catastrophic forgetting)
        'client_3': {
            'a_samples': 300, 'a_ids': 40,
            'b_samples': 0,   'b_ids': 0,     # No B samples (worst-case non-IID)
            'test_a': 30, 'test_b': 30,       # Note: Still test B to measure transfer
            'gallery_a': 90, 'gallery_b': 90
        },
        # Client 4: Data-rich in B, minimal A (inverts the skew, tests generalization)
        'client_4': {
            'a_samples': 10,  'a_ids': 2,
            'b_samples': 1200, 'b_ids': 50,   # Dominant in B
            'test_a': 20, 'test_b': 20,
            'gallery_a': 90, 'gallery_b': 90
        }
    }
    
    # Track used identities to avoid overlap
    used_a_identities = set()
    used_b_identities = set()
    
    # Create client assignments
    client_data = {}
    
    for client_name, target in targets.items():
        print(f"\n=== Creating {client_name} ===")
        client_data[client_name] = {}
        
        # Select Species A identities for training
        if target['a_samples'] > 0:
            available_a = species_a_df[~species_a_df[identity_column].isin(used_a_identities)]
            selected_a, selected_a_ids = select_identities_for_target(
                available_a, target['a_ids'], target['a_samples'], identity_column, random_seed
            )
            client_data[client_name]['train_a'] = selected_a
            used_a_identities.update(selected_a_ids)
        else:
            client_data[client_name]['train_a'] = pd.DataFrame()
        
        # Select Species B identities for training
        if target['b_samples'] > 0:
            available_b = species_b_df[~species_b_df[identity_column].isin(used_b_identities)]
            selected_b, selected_b_ids = select_identities_for_target(
                available_b, target['b_ids'], target['b_samples'], identity_column, random_seed
            )
            client_data[client_name]['train_b'] = selected_b
            used_b_identities.update(selected_b_ids)
        else:
            client_data[client_name]['train_b'] = pd.DataFrame()
        
        # Select test identities (separate from training)
        # Test A
        available_a_test = species_a_df[~species_a_df[identity_column].isin(used_a_identities)]
        test_a_needed = target['test_a'] + target['gallery_a']  # Total test samples needed
        test_a_ids_needed = min(20, len(available_a_test[identity_column].unique()))  # Reasonable number of test IDs
        
        if len(available_a_test) > 0:
            selected_test_a, selected_test_a_ids = select_identities_for_target(
                available_a_test, test_a_ids_needed, test_a_needed, identity_column, random_seed
            )
            client_data[client_name]['test_a'] = selected_test_a
            used_a_identities.update(selected_test_a_ids)
        else:
            client_data[client_name]['test_a'] = pd.DataFrame()
        
        # Test B  
        available_b_test = species_b_df[~species_b_df[identity_column].isin(used_b_identities)]
        test_b_needed = target['test_b'] + target['gallery_b']  # Total test samples needed
        test_b_ids_needed = min(20, len(available_b_test[identity_column].unique()))  # Reasonable number of test IDs
        
        if len(available_b_test) > 0:
            selected_test_b, selected_test_b_ids = select_identities_for_target(
                available_b_test, test_b_ids_needed, test_b_needed, identity_column, random_seed
            )
            client_data[client_name]['test_b'] = selected_test_b
            used_b_identities.update(selected_test_b_ids)
        else:
            client_data[client_name]['test_b'] = pd.DataFrame()
    
    # Create client folders and CSV files
    os.makedirs(output_dir, exist_ok=True)
    client_stats = []
    
    for idx, (client_name, data) in enumerate(client_data.items()):
        client_folder = os.path.join(output_dir, str(idx))
        os.makedirs(client_folder, exist_ok=True)
        
        print(f"\n=== Saving {client_name} (folder {idx}) ===")
        
        # Combine training data
        train_records = []
        if not data['train_a'].empty:
            train_records.append(data['train_a'])
        if not data['train_b'].empty:
            train_records.append(data['train_b'])
        
        if train_records:
            train_df = pd.concat(train_records, ignore_index=True)
        else:
            train_df = pd.DataFrame()
        
        # Create test sets from test identities
        query_records = []
        gallery_records = []
        
        # Process test A
        if not data['test_a'].empty:
            test_a_identities = data['test_a'][identity_column].unique()
            target_query_a = targets[client_name]['test_a']
            target_gallery_a = targets[client_name]['gallery_a']
            
            # Collect all available samples for each identity
            query_a_samples = []
            gallery_a_samples = []
            
            # First, ensure each identity has at least one sample in gallery
            for identity_id in test_a_identities:
                identity_samples = data['test_a'][data['test_a'][identity_column] == identity_id]
                if len(identity_samples) >= 2:  # Need at least 1 query + 1 gallery
                    # Take samples for queries (can be multiple per identity)
                    query_portion = max(1, len(identity_samples) // 3)  # Take roughly 1/3 for queries
                    query_a_samples.append(identity_samples.iloc[:query_portion])
                    gallery_a_samples.append(identity_samples.iloc[query_portion:])
                elif len(identity_samples) == 1:
                    # If only one sample, put it in gallery only
                    gallery_a_samples.append(identity_samples)
            
            # Create query dataset up to target
            if query_a_samples:
                query_a_df = pd.concat(query_a_samples, ignore_index=True)
                # Shuffle and take up to target
                query_a_df = query_a_df.sample(frac=1, random_state=42).reset_index(drop=True)
                query_records.append(query_a_df.head(target_query_a))
            
            # Create gallery dataset up to target
            if gallery_a_samples:
                gallery_a_df = pd.concat(gallery_a_samples, ignore_index=True)
                # Shuffle and take up to target
                gallery_a_df = gallery_a_df.sample(frac=1, random_state=42).reset_index(drop=True)
                gallery_records.append(gallery_a_df.head(target_gallery_a))
        
        # Process test B (similar to test A)
        if not data['test_b'].empty:
            test_b_identities = data['test_b'][identity_column].unique()
            target_query_b = targets[client_name]['test_b']
            target_gallery_b = targets[client_name]['gallery_b']
            
            # Collect all available samples for each identity
            query_b_samples = []
            gallery_b_samples = []
            
            # First, ensure each identity has at least one sample in gallery
            for identity_id in test_b_identities:
                identity_samples = data['test_b'][data['test_b'][identity_column] == identity_id]
                if len(identity_samples) >= 2:  # Need at least 1 query + 1 gallery
                    # Take samples for queries (can be multiple per identity)
                    query_portion = max(1, len(identity_samples) // 3)  # Take roughly 1/3 for queries
                    query_b_samples.append(identity_samples.iloc[:query_portion])
                    gallery_b_samples.append(identity_samples.iloc[query_portion:])
                elif len(identity_samples) == 1:
                    # If only one sample, put it in gallery only
                    gallery_b_samples.append(identity_samples)
            
            # Create query dataset up to target
            if query_b_samples:
                query_b_df = pd.concat(query_b_samples, ignore_index=True)
                # Shuffle and take up to target
                query_b_df = query_b_df.sample(frac=1, random_state=42).reset_index(drop=True)
                query_records.append(query_b_df.head(target_query_b))
            
            # Create gallery dataset up to target
            if gallery_b_samples:
                gallery_b_df = pd.concat(gallery_b_samples, ignore_index=True)
                # Shuffle and take up to target
                gallery_b_df = gallery_b_df.sample(frac=1, random_state=42).reset_index(drop=True)
                gallery_records.append(gallery_b_df.head(target_gallery_b))
        
        # Create final DataFrames
        if query_records:
            query_df = pd.concat(query_records, ignore_index=True)
        else:
            query_df = pd.DataFrame()
        
        if gallery_records:
            gallery_df = pd.concat(gallery_records, ignore_index=True)
        else:
            gallery_df = pd.DataFrame()
        
        # Ensure all queries are in gallery (add query samples to gallery if not present)
        if not query_df.empty and not gallery_df.empty:
            # Add all query samples to gallery to ensure they're present
            combined_gallery = pd.concat([gallery_df, query_df], ignore_index=True).drop_duplicates()
            gallery_df = combined_gallery
        
        # Save CSV files
        train_df.to_csv(os.path.join(client_folder, 'train.csv'), index=False)
        query_df.to_csv(os.path.join(client_folder, 'query.csv'), index=False)
        gallery_df.to_csv(os.path.join(client_folder, 'gallery.csv'), index=False)
        
        # Calculate statistics
        train_a = (train_df[species_column] == species_a_value).sum() if not train_df.empty else 0
        train_b = (train_df[species_column] == species_b_value).sum() if not train_df.empty else 0
        query_a = (query_df[species_column] == species_a_value).sum() if not query_df.empty else 0
        query_b = (query_df[species_column] == species_b_value).sum() if not query_df.empty else 0
        gallery_a = (gallery_df[species_column] == species_a_value).sum() if not gallery_df.empty else 0
        gallery_b = (gallery_df[species_column] == species_b_value).sum() if not gallery_df.empty else 0
        
        train_ids_a = train_df[train_df[species_column] == species_a_value][identity_column].nunique() if not train_df.empty else 0
        train_ids_b = train_df[train_df[species_column] == species_b_value][identity_column].nunique() if not train_df.empty else 0
        
        print(f"  Train: {len(train_df)} samples (A:{train_a}/{train_ids_a}ids, B:{train_b}/{train_ids_b}ids)")
        print(f"  Query: {len(query_df)} samples (A:{query_a}, B:{query_b})")
        print(f"  Gallery: {len(gallery_df)} samples (A:{gallery_a}, B:{gallery_b})")
        
        client_stats.append({
            'client_id': idx,
            'client_name': client_name,
            'train_samples_a': train_a,
            'train_samples_b': train_b,
            'train_ids_a': train_ids_a,
            'train_ids_b': train_ids_b,
            'query_samples_a': query_a,
            'query_samples_b': query_b,
            'gallery_samples_a': gallery_a,
            'gallery_samples_b': gallery_b,
            'total_samples': len(train_df) + len(query_df) + len(gallery_df)
        })
    
    # Save summary
    if client_stats:
        stats_df = pd.DataFrame(client_stats)
        stats_df.to_csv(os.path.join(output_dir, 'exact_distribution_summary.csv'), index=False)
        
        print(f"\n{'='*90}")
        print("EXACT DISTRIBUTION SUMMARY")
        print(f"{'='*90}")
        print(f"Client | Train A/B (IDs)    | Query A/B | Gallery A/B | Total | Description")
        print(f"-------|---------------------|-----------|-------------|-------|-------------")
        descriptions = ["Rich A, Poor B", "Moderate A+B", "Medium A, Test-Only B", "Poor A, Rich B"]
        
        for i, row in stats_df.iterrows():
            train_str = f"{row['train_samples_a']:4}/{row['train_samples_b']:2} ({row['train_ids_a']:2}/{row['train_ids_b']:1})"
            print(f"{row['client_id']:6} | {train_str:19} | {row['query_samples_a']:4}/{row['query_samples_b']:4} | {row['gallery_samples_a']:6}/{row['gallery_samples_b']:6} | {row['total_samples']:5} | {descriptions[i]}")
        
        print(f"\nClient data saved to: {output_dir}/")
        print(f"Summary saved to: {os.path.join(output_dir, 'exact_distribution_summary.csv')}")
    
    return client_stats

def main():
    parser = argparse.ArgumentParser(description='Create Exact Distribution Clients')
    parser.add_argument('--input_csv', required=True, help='Input CSV file with two species')
    parser.add_argument('--output_dir', default='lep_hyn_exact2', help='Output directory')
    parser.add_argument('--species_column', default='species', help='Species column name')
    parser.add_argument('--identity_column', default='identity', help='Identity column name')
    parser.add_argument('--species_a_value', default='leopard', help='Species A value')
    parser.add_argument('--species_b_value', default='hyena', help='Species B value')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--min_samples', type=int, default=5, help='Minimum samples per identity')
    
    args = parser.parse_args()
    
    print("Exact Distribution Client Generator")
    print("=" * 50)
    
    # Load dataframe
    print(f"Loading data from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Analyze dataframe
    print("\nAnalyzing input dataframe...")
    analyze_dataframe(df, args.species_column, args.identity_column)
    
    # Filter identities with minimum samples
    df_filtered = filter_identities_by_samples(df, args.species_column, args.identity_column, args.min_samples)
    
    if len(df_filtered) == 0:
        print("Error: No identities have sufficient samples after filtering")
        return
    
    # Verify we have both species
    species_in_filtered = df_filtered[args.species_column].unique()
    if args.species_a_value not in species_in_filtered:
        print(f"Error: Species A ({args.species_a_value}) not found in filtered data")
        return
    if args.species_b_value not in species_in_filtered:
        print(f"Error: Species B ({args.species_b_value}) not found in filtered data")
        return
    
    # Create exact distribution clients
    client_stats = create_exact_distribution(
        df_filtered,
        args.species_column,
        args.identity_column,
        args.species_a_value,
        args.species_b_value,
        args.output_dir,
        args.random_seed
    )
    
    print("\nExact distribution generation completed successfully!")

if __name__ == '__main__':
    main()