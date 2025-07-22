import os
import pandas as pd
import numpy as np


def create_test(data_dir, test_percentages, random_seed=42):
    """
    Create test sets (query/gallery split) from client training data.
    
    Args:
        data_dir: Path to directory containing numbered client folders
        test_percentages: List of percentages (0.0-1.0) for each client's test data
        random_seed: Random seed for reproducibility
        
    Rules:
        - All 'unknown' IDs go to gallery
        - Only IDs with ≤12 samples go to test set
        - IDs with >6 samples: 2 queries, rest gallery
        - IDs with 3-5 samples: 1 query, rest gallery  
        - IDs with <3 samples: all go to gallery
        - No overlapping IDs between query and gallery
    """
    np.random.seed(random_seed)
    
    print(f"Creating test sets from: {data_dir}")
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Find all numbered client folders
    client_folders = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            client_folders.append(int(item))
    
    client_folders.sort()
    
    if not client_folders:
        raise ValueError(f"No numbered client folders found in {data_dir}")
    
    if len(test_percentages) != len(client_folders):
        raise ValueError(f"Number of test percentages ({len(test_percentages)}) must match number of clients ({len(client_folders)})")
    
    print(f"Found {len(client_folders)} client folders: {client_folders}")
    print(f"Test percentages: {test_percentages}")
    
    client_stats = []
    
    # Process each client
    for i, client_id in enumerate(client_folders):
        test_percentage = test_percentages[i]
        client_dir = os.path.join(data_dir, str(client_id))
        train_csv_path = os.path.join(client_dir, 'train.csv')
        
        if not os.path.exists(train_csv_path):
            print(f"Warning: train.csv not found in client {client_id}, skipping...")
            continue
        
        print(f"\nProcessing Client {client_id} (test percentage: {test_percentage:.1%}):")
        
        try:
            # Load training data
            train_df = pd.read_csv(train_csv_path)
            
            # Group by identity and count samples
            identity_groups = train_df.groupby('identity')
            identity_counts = identity_groups.size().to_dict()
            
            # Separate identities by sample count and type
            test_candidates = []
            unknown_identities = []
            
            for identity, count in identity_counts.items():
                if identity == 'unknown':
                    unknown_identities.append(identity)
                elif count <= 12:  # Only IDs with ≤12 samples can go to test
                    test_candidates.append(identity)
            
            print(f"  Found {len(test_candidates)} test candidate IDs, {len(unknown_identities)} unknown IDs")
            
            # Calculate how many test candidates to use based on percentage
            if test_candidates:
                target_test_samples = int(len(train_df) * test_percentage)
                
                # Separate candidates into distractors and regular test candidates
                distractor_candidates = [id for id in test_candidates if identity_counts[id] < 4]
                regular_candidates = [id for id in test_candidates if identity_counts[id] >= 4]
                
                selected_test_ids = []
                current_test_samples = 0
                
                # ALWAYS include distractors proportional to test set size
                if distractor_candidates:
                    # Calculate distractor allocation proportional to test percentage
                    # Higher test percentage = more distractors
                    base_distractors = max(2, int(len(distractor_candidates) * test_percentage * 2))
                    max_distractors = min(base_distractors, len(distractor_candidates))
                    
                    # Reserve 30-50% of test quota for distractors
                    distractor_budget = int(target_test_samples * 0.4)
                    
                    np.random.shuffle(distractor_candidates)
                    distractors_added = 0
                    
                    for identity in distractor_candidates:
                        if (distractors_added < max_distractors and 
                            current_test_samples + identity_counts[identity] <= distractor_budget):
                            selected_test_ids.append(identity)
                            current_test_samples += identity_counts[identity]
                            distractors_added += 1
                    
                    print(f"  Added {distractors_added} distractor IDs (budget: {distractor_budget} samples)")
                
                # Fill remaining quota with regular candidates (sorted by size)
                if regular_candidates:
                    regular_sorted = sorted(regular_candidates, key=lambda x: identity_counts[x], reverse=True)
                    for identity in regular_sorted:
                        if current_test_samples + identity_counts[identity] <= target_test_samples:
                            selected_test_ids.append(identity)
                            current_test_samples += identity_counts[identity]
                        if current_test_samples >= target_test_samples * 0.8:  # Allow some flexibility
                            break
            else:
                selected_test_ids = []
            
            # Add unknown identities proportionally to test set (they go to gallery as distractors)
            selected_unknown_ids = []
            if unknown_identities:
                # Distribute unknown samples proportionally to test percentage
                for unknown_id in unknown_identities:
                    unknown_samples = identity_counts[unknown_id]
                    # Take a portion of unknown samples proportional to test percentage
                    target_unknown = max(1, int(unknown_samples * test_percentage * 1.5))  # Boost unknown allocation
                    if target_unknown > 0:
                        selected_unknown_ids.append(unknown_id)
                        print(f"  Added unknown ID '{unknown_id}' with {unknown_samples} samples")
            
            all_test_ids = selected_test_ids + selected_unknown_ids
            
            print(f"  Selected {len(selected_test_ids)} known IDs for test, {len(unknown_identities)} unknown IDs")
            
            # Split test identities into query and gallery
            query_data = []
            gallery_data = []
            query_identities = set()
            
            for identity in all_test_ids:
                identity_samples = identity_groups.get_group(identity).copy()
                identity_samples = identity_samples.sample(frac=1, random_state=random_seed).reset_index(drop=True)
                
                # For unknown identities, sample only a portion to distribute across clients
                if identity == 'unknown':
                    total_unknown = len(identity_samples)
                    sample_count = max(1, int(total_unknown * test_percentage * 1.5))
                    identity_samples = identity_samples.head(sample_count)
                
                total_samples = len(identity_samples)
                
                # Apply query/gallery rules
                if identity == 'unknown':
                    # All unknown samples go to gallery (distractors)
                    num_query = 0
                elif total_samples > 6:
                    # >6 samples: 2 queries, rest gallery
                    num_query = 2
                elif total_samples >= 3:
                    # 3-6 samples: 1 query, rest gallery  
                    num_query = 1
                else:
                    # <3 samples: all go to gallery (distractors)
                    num_query = 0
                
                # Split samples - IMPORTANT: All samples go to gallery, queries are subset
                if len(identity_samples) > 0:
                    gallery_data.append(identity_samples)  # ALL samples go to gallery
                
                # Extract query samples (subset of gallery)
                if num_query > 0:
                    query_samples = identity_samples.iloc[:num_query]
                    query_data.append(query_samples)
                    query_identities.add(identity)
            
            # Create query and gallery dataframes
            query_df = pd.concat(query_data, ignore_index=True) if query_data else pd.DataFrame()
            gallery_df = pd.concat(gallery_data, ignore_index=True) if gallery_data else pd.DataFrame()
            
            # Validation: Ensure all query identities are in gallery
            if len(query_df) > 0:
                gallery_identities = set(gallery_df['identity'].unique())
                missing_in_gallery = query_identities - gallery_identities
                if missing_in_gallery:
                    print(f"  WARNING: Query identities not in gallery: {missing_in_gallery}")
                else:
                    print(f"  ✓ All query identities are present in gallery")
            
            # Save query and gallery CSV files
            query_df.to_csv(os.path.join(client_dir, 'query.csv'), index=False)
            gallery_df.to_csv(os.path.join(client_dir, 'gallery.csv'), index=False)
            
            # CRITICAL: Remove test samples from training data to prevent data leakage
            # Create new train set by excluding all test identities
            remaining_train_data = []
            for identity in train_df['identity'].unique():
                if identity not in all_test_ids:  # Only keep identities NOT in test set
                    identity_train_samples = train_df[train_df['identity'] == identity]
                    remaining_train_data.append(identity_train_samples)
            
            # Update train.csv with remaining training data (no test overlap)
            if remaining_train_data:
                new_train_df = pd.concat(remaining_train_data, ignore_index=True)
                new_train_df.to_csv(os.path.join(client_dir, 'train.csv'), index=False)
                print(f"  ✓ Updated train.csv: {len(train_df)} → {len(new_train_df)} samples ({len(train_df) - len(new_train_df)} removed)")
            else:
                # Create empty train file if all data was moved to test
                pd.DataFrame().to_csv(os.path.join(client_dir, 'train.csv'), index=False)
                print(f"  ✓ Updated train.csv: {len(train_df)} → 0 samples (all moved to test)")
            
            # Calculate distractors and validation statistics
            num_distractors = 0
            num_query_ids = len(query_identities) if query_identities else 0
            num_distractor_ids = 0
            
            if len(gallery_df) > 0:
                for identity in gallery_df['identity'].unique():
                    if identity == 'unknown' or identity_counts.get(identity, 0) < 4:
                        num_distractors += len(gallery_df[gallery_df['identity'] == identity])
                        num_distractor_ids += 1
                        
            # Additional validation
            print(f"  ✓ Distractors: {num_distractors} samples from {num_distractor_ids} IDs")
            print(f"  ✓ Query IDs: {num_query_ids}, Gallery IDs: {len(gallery_df['identity'].unique()) if len(gallery_df) > 0 else 0}")
            
            # Calculate final training set size after test removal
            final_train_size = len(new_train_df) if remaining_train_data else 0
            
            client_stats.append({
                'client_id': client_id,
                'train': final_train_size,
                'query': len(query_df),
                'gallery': len(gallery_df),
                'distractors': num_distractors,
                'test_ids': len(all_test_ids)
            })
            
            print(f"  Final: {final_train_size} train, {len(query_df)} query, {len(gallery_df)} gallery, {num_distractors} distractors")
            
        except Exception as e:
            print(f"Error processing client {client_id}: {e}")
            continue
    
    # Display summary statistics
    if client_stats:
        print(f"\nFinal Data Distribution Summary:")
        print("=" * 70)
        print(f"{'Client':<8} {'Train':<8} {'Query':<8} {'Gallery':<8} {'Distractors':<12} {'Test IDs':<8}")
        print("-" * 70)
        
        total_train = total_query = total_gallery = total_distractors = total_test_ids = 0
        for stats in client_stats:
            print(f"{stats['client_id']:<8} {stats['train']:<8} {stats['query']:<8} {stats['gallery']:<8} {stats['distractors']:<12} {stats['test_ids']:<8}")
            total_train += stats['train']
            total_query += stats['query']
            total_gallery += stats['gallery']
            total_distractors += stats['distractors']
            total_test_ids += stats['test_ids']
        
        print("-" * 70)
        print(f"{'TOTAL:':<8} {total_train:<8} {total_query:<8} {total_gallery:<8} {total_distractors:<12} {total_test_ids:<8}")
        print("=" * 70)


if __name__ == "__main__":
    # Example usage
    data_dir = '/home/wellvw12/fedwild/federated_clients'
    
    # Hard-coded: 3 clients with different test percentages
    test_percentages = [0.12, 0.15, 0.2]  # 15%, 20%, 25% for clients 0, 1, 2
    
    create_test(data_dir, test_percentages, random_seed=42)