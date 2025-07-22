import os
import pandas as pd
import numpy as np
import shutil
from wildlife_datasets.datasets import MacaqueFaces, Cows2021v2, LeopardID2022


def create_federated_partition(dataset_name, num_clients, alpha=0.1,
                              output_dir="federated_clients", random_seed=42):
    """
    Create federated partition using Dirichlet distribution for training data only.
    
    Args:
        dataset_name: Name of the dataset ('LeopardID2022', 'MacaqueFaces', 'Cows2021v2')
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter for training data (lower = more heterogeneous)
        min_samples_per_id: Minimum samples per identity to include in training
        output_dir: Output directory name
        random_seed: Random seed for reproducibility
    """
    
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
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Filter identities by minimum sample count (exclude 'unknown')
    df = df.groupby('identity', group_keys=False).apply(lambda x: x.sample(frac=0.6, random_state=42))
    identity_groups = df.groupby('identity')
    valid_identities = []
    identity_sample_counts = {}
    
    for identity, group in identity_groups:
        count = len(group)
        identity_sample_counts[identity] = count
        valid_identities.append(identity)
    
    print(f"Found {len(valid_identities)} valid identities for training")
    
    if len(valid_identities) < num_clients:
        raise ValueError(f"Not enough identities ({len(valid_identities)}) for {num_clients} clients")
    
    # Create training dataframe from all valid identities
    train_data = []
    for identity in valid_identities:
        identity_samples = df[df['identity'] == identity].copy()
        train_data.append(identity_samples)
    
    train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
    print(f"Total training samples: {len(train_df)}")
    
    # Apply Dirichlet partitioning to all valid identities
    client_proportions = np.random.dirichlet([alpha] * num_clients)
    
    # Calculate target number of identities per client
    total_identities = len(valid_identities)
    target_ids_per_client = (client_proportions * total_identities).astype(int)
    
    # Adjust for rounding errors
    remaining_ids = total_identities - np.sum(target_ids_per_client)
    for i in range(remaining_ids):
        target_ids_per_client[i % num_clients] += 1
    
    # Allocate identities to clients
    np.random.shuffle(valid_identities)
    client_identities = [[] for _ in range(num_clients)]
    
    current_idx = 0
    for client_id in range(num_clients):
        num_ids_for_client = target_ids_per_client[client_id]
        client_identities[client_id] = valid_identities[current_idx:current_idx + num_ids_for_client]
        current_idx += num_ids_for_client
    
    # Ensure each client has at least one identity
    for client_id in range(num_clients):
        if len(client_identities[client_id]) == 0:
            max_client = max(range(num_clients), key=lambda x: len(client_identities[x]))
            if len(client_identities[max_client]) > 1:
                identity_to_move = client_identities[max_client].pop()
                client_identities[client_id].append(identity_to_move)
    
    # Create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Save client data
    client_stats = []
    
    for client_id in range(num_clients):
        client_dir = os.path.join(output_dir, str(client_id))
        os.makedirs(client_dir)
        
        # Create training data for this client
        client_train_data = []
        for identity in client_identities[client_id]:
            identity_samples = train_df[train_df['identity'] == identity]
            if len(identity_samples) > 0:
                client_train_data.append(identity_samples)
        
        if client_train_data:
            client_train_df = pd.concat(client_train_data, ignore_index=True)
            client_train_df.to_csv(os.path.join(client_dir, 'train.csv'), index=False)
        else:
            # Create empty train file if no data
            pd.DataFrame().to_csv(os.path.join(client_dir, 'train.csv'), index=False)
        
        
        # Store statistics
        train_samples = len(client_train_df) if client_train_data else 0
        client_stats.append({
            'client_id': client_id,
            'train': train_samples,
            'identities': len(client_identities[client_id])
        })
    
    # Display statistics
    print(f"\nClient Data Distribution:")
    print("=" * 30)
    print(f"{'Client':<8} {'Train':<8} {'IDs':<8}")
    print("-" * 30)
    
    total_train = total_ids = 0
    
    for stats in client_stats:
        print(f"{stats['client_id']:<8} {stats['train']:<8} {stats['identities']:<8}")
        total_train += stats['train']
        total_ids += stats['identities']
    
    print("-" * 30)
    print(f"{'TOTAL:':<8} {total_train:<8} {total_ids:<8}")
    print("=" * 30)
    
    return client_identities


if __name__ == "__main__":
    # Example usage
    dataset_name = "LeopardID2022"
    
    # Create partition
    create_federated_partition(
        dataset_name=dataset_name,
        num_clients=3,
        alpha=0.7,
        output_dir="federated_clients",
        random_seed=56
    )