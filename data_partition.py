from wildlife_datasets.datasets import MacaqueFaces, Cows2021v2, LeopardID2022
import os
import pandas as pd
import numpy as np
import shutil

def create_client_partition_hardcoded(dataset_name, client_configs, target_samples_per_id=None, num_queries=30, num_gallery=200, output_dir="clients", random_seed=42, query_ids_per_client=15, queries_per_id=2, gallery_samples_per_query_id=12):
    """
    Partition dataset into client folders with train.csv, query.csv, and gallery.csv
    
    Args:
        dataset_name: Name of the dataset to use
        client_configs: List of integers specifying samples per client [client0_samples, client1_samples, ...]
        target_samples_per_id: Target samples per identity (optional, will be estimated if None)
        num_queries: Number of query samples (default 30) - DEPRECATED, calculated from query_ids_per_client * queries_per_id
        num_gallery: Number of gallery samples (default 200) - DEPRECATED, calculated from query config
        output_dir: Output directory name
        random_seed: Random seed for reproducible partitioning (default 42)
        query_ids_per_client: Number of query identities per client (default 15)
        queries_per_id: Number of query samples per identity (default 2)  
        gallery_samples_per_query_id: Number of gallery samples per query identity (default 12)
    """
    
    # Load dataset
    if dataset_name == "LeopardID2022":
        df = LeopardID2022('/home/wellvw12/leopard').df
    elif dataset_name == "MacaqueFaces":
        df = MacaqueFaces().df
    elif dataset_name == "Cows2021v2":
        df = Cows2021v2().df
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Group by identity and get identity sample counts
    identity_groups = df.groupby('identity')
    identity_sample_counts = {}
    for identity, group in identity_groups:
        identity_sample_counts[identity] = len(group)
    
    # Estimate optimal samples_per_id if not provided
    if target_samples_per_id is None:
        total_samples_needed = sum(client_configs)
        total_identities_available = len(identity_sample_counts)
        avg_samples_per_identity = df.groupby('identity').size().mean()
        
        # Estimate samples_per_id to balance utilization and feasibility
        estimated_samples_per_id = max(2, min(10, int(avg_samples_per_identity * 0.7)))
        print(f"Auto-estimated samples_per_id: {estimated_samples_per_id} (avg available: {avg_samples_per_identity:.1f})")
    else:
        estimated_samples_per_id = target_samples_per_id
        print(f"Using target samples_per_id: {estimated_samples_per_id}")
    
    # Filter identities that have at least estimated_samples_per_id samples
    valid_identities = []
    for identity, count in identity_sample_counts.items():
        if count >= estimated_samples_per_id:
            valid_identities.append(identity)
    
    print(f"Found {len(valid_identities)} identities with at least {estimated_samples_per_id} samples")
    
    # Create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    num_clients = len(client_configs)
    
    # Calculate how many identities each client needs and adjust samples_per_id for each client
    client_identity_counts = []
    client_samples_per_id = []
    total_identities_needed = 0
    
    print("Client allocation:")
    for i, samples_per_client in enumerate(client_configs):
        # Start with estimated samples_per_id but adjust to fit target
        best_samples_per_id = estimated_samples_per_id
        best_identities_needed = samples_per_client // best_samples_per_id
        best_actual_samples = best_identities_needed * best_samples_per_id
        best_diff = abs(samples_per_client - best_actual_samples)
        
        # Try different samples_per_id values to get closer to target
        for test_samples_per_id in range(max(2, estimated_samples_per_id - 3), min(20, estimated_samples_per_id + 4)):
            test_identities_needed = samples_per_client // test_samples_per_id
            test_actual_samples = test_identities_needed * test_samples_per_id
            test_diff = abs(samples_per_client - test_actual_samples)
            
            # Prefer solution that gets closer to target samples
            if test_diff < best_diff:
                best_samples_per_id = test_samples_per_id
                best_identities_needed = test_identities_needed
                best_actual_samples = test_actual_samples
                best_diff = test_diff
        
        client_identity_counts.append(best_identities_needed)
        client_samples_per_id.append(best_samples_per_id)
        total_identities_needed += best_identities_needed
        
        print(f"  Client {i}: {best_actual_samples}/{samples_per_client} samples, {best_identities_needed} identities, {best_samples_per_id} samples/id")
    
    # Check if we have enough identities
    if total_identities_needed > len(valid_identities):
        raise ValueError(f"Not enough identities. Need {total_identities_needed}, have {len(valid_identities)}")
    
    print(f"Total identities needed: {total_identities_needed}/{len(valid_identities)} available")
    
    # Randomly shuffle identities with fixed seed for reproducibility
    np.random.seed(random_seed)
    np.random.shuffle(valid_identities)
    
    # Partition identities to clients (ensuring no overlap)
    client_identities = []
    used_identities = set()
    current_idx = 0
    
    for i in range(num_clients):
        identities_for_client = client_identity_counts[i]
        client_ids = valid_identities[current_idx:current_idx + identities_for_client]
        client_identities.append(client_ids)
        used_identities.update(client_ids)
        current_idx += identities_for_client
    
    # Get unused identities for test set (from ALL identities, not just valid_identities)
    all_identities = list(identity_sample_counts.keys())
    unused_identities = [id for id in all_identities if id not in used_identities]
    print(f"Unused identities for test set: {len(unused_identities)}")
    
    # Calculate query/gallery configuration 
    calculated_num_queries = query_ids_per_client * queries_per_id
    query_gallery_samples = query_ids_per_client * gallery_samples_per_query_id
    remaining_gallery_slots = max(0, num_gallery - query_gallery_samples)
    
    print(f"Query/Gallery Config: {query_ids_per_client} IDs × {queries_per_id} queries = {calculated_num_queries} queries")
    print(f"Gallery for query IDs: {query_ids_per_client} × {gallery_samples_per_query_id} = {query_gallery_samples} samples")  
    print(f"Remaining gallery slots for distractors: {remaining_gallery_slots}")
    
    # Filter test identities that have enough samples (queries + gallery per ID)
    min_samples_per_test_id = queries_per_id + gallery_samples_per_query_id
    test_identities = []
    for identity in unused_identities:
        identity_samples = df[df['identity'] == identity]
        if len(identity_samples) >= min_samples_per_test_id:
            test_identities.append(identity)
    
    print(f"Available test identities (≥{min_samples_per_test_id} samples): {len(test_identities)}")
    
    # Create global query and gallery sets using improved configuration
    query_data = []
    gallery_data = []
    
    # Select query identities - need enough identities for the improved config
    if len(test_identities) < query_ids_per_client:
        raise ValueError(f"Not enough test identities. Need {query_ids_per_client}, have {len(test_identities)}")
    
    np.random.seed(random_seed + 1)
    np.random.shuffle(test_identities)
    query_identities_used = test_identities[:query_ids_per_client]
    
    # Create query set: queries_per_id samples per identity
    for i, identity in enumerate(query_identities_used):
        identity_samples = df[df['identity'] == identity]
        available_samples = len(identity_samples)
        
        if available_samples < queries_per_id + gallery_samples_per_query_id:
            print(f"Warning: Identity {identity} has only {available_samples} samples, need {queries_per_id + gallery_samples_per_query_id}")
            continue
            
        # Sample queries_per_id for queries
        query_samples = identity_samples.sample(n=queries_per_id, random_state=random_seed + 100 + i)
        query_data.append(query_samples)
    
    # Create gallery set: gallery_samples_per_query_id per query identity + distractors
    for i, identity in enumerate(query_identities_used):
        identity_samples = df[df['identity'] == identity]
        
        # Exclude samples already used for queries
        query_samples_for_this_id = df[(df['identity'] == identity) & 
                                      (df.index.isin(query_data[i].index))]
        remaining_samples = identity_samples[~identity_samples.index.isin(query_samples_for_this_id.index)]
        
        # Sample gallery_samples_per_query_id for gallery
        samples_to_take = min(gallery_samples_per_query_id, len(remaining_samples))
        if samples_to_take > 0:
            gallery_samples = remaining_samples.sample(n=samples_to_take, random_state=random_seed + 200 + i)
            gallery_data.append(gallery_samples)
    
    # Add distractor samples to fill remaining gallery slots
    current_gallery_count = sum(len(gd) for gd in gallery_data)
    distractor_slots = num_gallery - current_gallery_count
    
    if distractor_slots > 0:
        # Get identities not used for queries  
        distractor_identities = [id for id in test_identities if id not in query_identities_used]
        np.random.seed(random_seed + 2)
        np.random.shuffle(distractor_identities)
        
        distractor_samples_collected = 0
        for i, identity in enumerate(distractor_identities):
            if distractor_samples_collected >= distractor_slots:
                break
                
            identity_samples = df[df['identity'] == identity]
            samples_to_take = min(3, len(identity_samples), distractor_slots - distractor_samples_collected)
            
            if samples_to_take > 0:
                distractor_samples = identity_samples.sample(n=samples_to_take, random_state=random_seed + 300 + i)
                gallery_data.append(distractor_samples)
                distractor_samples_collected += samples_to_take
    
    # Create client folders and CSV files
    for client_id in range(num_clients):
        client_dir = os.path.join(output_dir, str(client_id))
        os.makedirs(client_dir)
        
        # Get training data for this client
        client_data = []
        samples_per_id_for_client = client_samples_per_id[client_id]
        for identity in client_identities[client_id]:
            identity_samples = df[df['identity'] == identity]
            # Take minimum of requested samples or available samples
            samples_to_take = min(samples_per_id_for_client, len(identity_samples))
            sampled_identity_samples = identity_samples.sample(n=samples_to_take, random_state=random_seed + client_id * 1000 + hash(identity) % 1000)
            client_data.append(sampled_identity_samples)
        
        # Create train.csv
        if client_data:
            train_df = pd.concat(client_data, ignore_index=True)
            train_df.to_csv(os.path.join(client_dir, 'train.csv'), index=False)
        
        # Save same query.csv and gallery.csv for all clients
        if query_data:
            query_df = pd.concat(query_data, ignore_index=True)
            query_df.to_csv(os.path.join(client_dir, 'query.csv'), index=False)
        
        if gallery_data:
            gallery_df = pd.concat(gallery_data, ignore_index=True)
            gallery_df.to_csv(os.path.join(client_dir, 'gallery.csv'), index=False)
        
        actual_samples = len(train_df) if client_data else 0
        expected_samples = client_configs[client_id]
        samples_per_id_used = client_samples_per_id[client_id]
        print(f"Client {client_id}: {actual_samples}/{expected_samples} train ({samples_per_id_used} samples/id), {len(query_df) if query_data else 0} query, {len(gallery_df) if gallery_data else 0} gallery")

# Keep original function for backward compatibility
def create_client_partition(dataset_name, num_clients, samples_per_client, samples_per_id, num_queries=30, num_gallery=200, output_dir="clients"):
    """Original function - creates equal-sized clients"""
    client_configs = [samples_per_client] * num_clients
    return create_client_partition_hardcoded(dataset_name, client_configs, samples_per_id, num_queries, num_gallery, output_dir)

# Example usage
if __name__ == "__main__":
    # Improved configuration for better ReID evaluation metrics
    dataset_name = "LeopardID2022"
    client_configs = [800, 30]  # Specify target samples per client
    random_seed = 42  # For reproducible results
    
    # Use improved query/gallery configuration
    create_client_partition_hardcoded(
        dataset_name=dataset_name,
        client_configs=client_configs,
        target_samples_per_id=15,
        # Legacy parameters (will be overridden by improved config)
        num_queries=30,  
        num_gallery=250,
        output_dir="../baselines/baseline800.30_improved",
        random_seed=random_seed,
        # Improved configuration parameters
        query_ids_per_client=15,           # 15 identities for queries
        queries_per_id=2,                  # 2 queries per identity = 30 total queries
        gallery_samples_per_query_id=12    # 12 gallery per query ID = 180 + 70 distractors
    )