from wildlife_datasets.datasets import MacaqueFaces, Cows2021v2, LeopardID2022
import os
import pandas as pd
import numpy as np
import shutil

def create_client_partition(dataset_name, num_clients, samples_per_client, samples_per_id, num_queries=30, num_gallery=200, output_dir="clients"):
    """
    Partition dataset into client folders with train.csv, query.csv, and gallery.csv
    
    Args:
        dataset_name: Name of the dataset to use
        num_clients: Number of clients (x)
        samples_per_client: Number of samples per client (s)
        samples_per_id: Number of samples per identity (z)
        num_queries: Number of query samples (default 30)
        num_gallery: Number of gallery samples (default 200)
        output_dir: Output directory name
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
    
    # Group by identity
    identity_groups = df.groupby('identity')
    
    # Filter identities that have at least samples_per_id samples
    valid_identities = []
    for identity, group in identity_groups:
        if len(group) >= samples_per_id:
            valid_identities.append(identity)
    
    print(f"Found {len(valid_identities)} identities with at least {samples_per_id} samples")
    
    # Create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Calculate how many identities per client
    identities_per_client = samples_per_client // samples_per_id
    
    # Check if we have enough identities
    total_identities_needed = num_clients * identities_per_client
    if total_identities_needed > len(valid_identities):
        raise ValueError(f"Not enough identities. Need {total_identities_needed}, have {len(valid_identities)}")
    
    print(f"Each client will have {identities_per_client} identities")
    
    # Randomly shuffle identities
    np.random.seed(42)
    np.random.shuffle(valid_identities)
    
    # Partition identities to clients (ensuring no overlap)
    client_identities = []
    used_identities = set()
    
    for i in range(num_clients):
        start_idx = i * identities_per_client
        end_idx = start_idx + identities_per_client
        client_ids = valid_identities[start_idx:end_idx]
        client_identities.append(client_ids)
        used_identities.update(client_ids)
    
    # Get unused identities for test set
    unused_identities = [id for id in valid_identities if id not in used_identities]
    print(f"Unused identities for test set: {len(unused_identities)}")
    
    # Prepare test identities for query and gallery
    query_identities = []
    gallery_identities = []
    
    # Filter identities for query (need at least 2 samples)
    for identity in unused_identities:
        identity_samples = df[df['identity'] == identity]
        if len(identity_samples) >= 2:
            query_identities.append(identity)
    
    # Filter identities for gallery (need at least 3 samples)
    for identity in unused_identities:
        identity_samples = df[df['identity'] == identity]
        if len(identity_samples) >= 3:
            gallery_identities.append(identity)
    
    print(f"Available for query: {len(query_identities)} identities")
    print(f"Available for gallery: {len(gallery_identities)} identities")
    
    # Create global query and gallery sets from unused identities
    query_data = []
    gallery_data = []
    
    # Create query set with 30 samples, min 2 samples per identity
    query_samples_collected = 0
    np.random.shuffle(query_identities)
    query_identities_used = []
    
    for identity in query_identities:
        if query_samples_collected >= num_queries:
            break
        identity_samples = df[df['identity'] == identity]
        samples_to_take = min(2, len(identity_samples), num_queries - query_samples_collected)
        query_samples = identity_samples.sample(n=samples_to_take, random_state=42)
        query_data.append(query_samples)
        query_samples_collected += samples_to_take
        query_identities_used.append(identity)
    
    # Create gallery set ensuring all query IDs are included
    gallery_samples_collected = 0
    
    # First, add samples from all query identities to gallery
    for identity in query_identities_used:
        if gallery_samples_collected >= num_gallery:
            break
        identity_samples = df[df['identity'] == identity]
        samples_to_take = min(3, len(identity_samples), num_gallery - gallery_samples_collected)
        gallery_samples = identity_samples.sample(n=samples_to_take, random_state=42)
        gallery_data.append(gallery_samples)
        gallery_samples_collected += samples_to_take
    
    # Then fill remaining gallery slots with other identities
    remaining_gallery_identities = [id for id in gallery_identities if id not in query_identities_used]
    np.random.shuffle(remaining_gallery_identities)
    
    for identity in remaining_gallery_identities:
        if gallery_samples_collected >= num_gallery:
            break
        identity_samples = df[df['identity'] == identity]
        samples_to_take = min(3, len(identity_samples), num_gallery - gallery_samples_collected)
        gallery_samples = identity_samples.sample(n=samples_to_take, random_state=42)
        gallery_data.append(gallery_samples)
        gallery_samples_collected += samples_to_take
    
    # Create client folders and CSV files
    for client_id in range(num_clients):
        client_dir = os.path.join(output_dir, str(client_id))
        os.makedirs(client_dir)
        
        # Get training data for this client
        client_data = []
        for identity in client_identities[client_id]:
            identity_samples = df[df['identity'] == identity].sample(n=samples_per_id, random_state=42)
            client_data.append(identity_samples)
        
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
        
        print(f"Client {client_id}: {len(train_df) if client_data else 0} train, {len(query_df) if query_data else 0} query, {len(gallery_df) if gallery_data else 0} gallery samples")

# Example usage
if __name__ == "__main__":
    # Parameters
    dataset_name = "LeopardID2022"
    num_clients = 3
    samples_per_client = 300
    samples_per_id = 10
    num_queries = 30
    num_gallery = 200
    
    create_client_partition(
        dataset_name=dataset_name,
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        samples_per_id=samples_per_id,
        num_queries=num_queries,
        num_gallery=num_gallery,
        output_dir="clients"
    )