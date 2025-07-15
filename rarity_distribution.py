import pandas as pd
import numpy as np
from collections import defaultdict
import os
from wildlife_datasets.datasets import LeopardID2022

def analyze_and_distribute_rarity(source_df, num_clients=7, richness_sections=5, rare_threshold_percentile=20, 
                                  test_set_percentage=0.3, output_folder='client_rarity_data', min_samples_threshold=5):
    """
    Analyze identity counts and distribute them based on rarity and richness levels
    
    Args:
        source_df: Source DataFrame with columns like ['image_id', 'identity', 'path', ...]
        num_clients: Number of clients to distribute to
        richness_sections: Number of richness sections to divide data into
        rare_threshold_percentile: Percentile threshold for rare cases (lower = more rare)
        test_set_percentage: Percentage of samples for test set (query + gallery)
        output_folder: Folder name to save client data
        min_samples_threshold: Minimum samples per identity (filter out below this)
    
    Returns:
        DataFrame with client assignments and rarity classifications
    """
    
    # Create identity counts from source DataFrame
    identity_counts = source_df['identity'].value_counts()
    
    # Remove 'unknown' entries if present
    identity_counts = identity_counts[identity_counts.index != 'unknown']
    
    # Filter out identities below minimum threshold
    print(f"Before filtering: {len(identity_counts)} identities")
    identity_counts = identity_counts[identity_counts >= min_samples_threshold]
    print(f"After filtering (>= {min_samples_threshold} samples): {len(identity_counts)} identities")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({'identity_id': identity_counts.index, 'sample_count': identity_counts.values})
    
    print(f"Total samples: {df['sample_count'].sum()}")
    print(f"Sample count statistics:")
    print(df['sample_count'].describe())
    
    # Determine rarity threshold
    rare_threshold = np.percentile(df['sample_count'], rare_threshold_percentile)
    print(f"\nRare threshold (bottom {rare_threshold_percentile}%): <= {rare_threshold} samples")
    
    # Classify rarity
    def classify_rarity(count):
        if count <= rare_threshold:
            return 'rare'
        elif count <= np.percentile(df['sample_count'], 50):
            return 'common'
        else:
            return 'abundant'
    
    df['rarity_class'] = df['sample_count'].apply(classify_rarity)
    
    # Create richness sections based on sample count quantiles
    richness_boundaries = np.percentile(df['sample_count'], 
                                      np.linspace(0, 100, richness_sections + 1))
    
    def classify_richness(count):
        for i in range(len(richness_boundaries) - 1):
            if richness_boundaries[i] <= count <= richness_boundaries[i + 1]:
                return f'section_{i + 1}'  # section_1 = poorest, section_n = richest
        return f'section_{richness_sections}'
    
    df['richness_section'] = df['sample_count'].apply(classify_richness)
    
    # Display distribution
    print(f"\nRarity distribution:")
    print(df['rarity_class'].value_counts())
    print(f"\nRichness section distribution:")
    print(df['richness_section'].value_counts())
    
    # Create client richness levels (some poor, some medium, some rich)
    client_richness = {}
    clients_per_section = max(1, num_clients // richness_sections)
    
    for i in range(richness_sections):
        section_name = f'section_{i + 1}'
        start_client = i * clients_per_section
        end_client = min((i + 1) * clients_per_section, num_clients)
        
        for client_id in range(start_client, end_client):
            client_richness[client_id] = section_name
    
    # Handle remaining clients (assign to richest section)
    for client_id in range(len(client_richness), num_clients):
        client_richness[client_id] = f'section_{richness_sections}'
    
    print(f"\nClient richness assignments:")
    for client_id, richness in client_richness.items():
        print(f"Client {client_id}: {richness}")
    
    # Initialize client assignments
    client_assignments = defaultdict(list)
    
    # Separate identities by rarity
    rare_identities = df[df['rarity_class'] == 'rare'].copy()
    non_rare_identities = df[df['rarity_class'] != 'rare'].copy()
    
    print(f"\nDistributing {len(rare_identities)} rare identities...")
    
    # Distribute rare identities: 70% to poor clients, 30% to rich clients
    np.random.seed(42)
    rare_identities = rare_identities.sample(frac=1).reset_index(drop=True)
    
    poor_clients = [cid for cid, richness in client_richness.items() if richness == 'section_1']
    rich_clients = [cid for cid, richness in client_richness.items() if richness == f'section_{richness_sections}']
    
    # Distribute rare cases
    for i, (_, identity_row) in enumerate(rare_identities.iterrows()):
        if i < len(rare_identities) * 0.7:  # 70% to poor clients
            if poor_clients:
                client_id = poor_clients[i % len(poor_clients)]
            else:
                client_id = i % num_clients
        else:  # 30% to rich clients
            if rich_clients:
                client_id = rich_clients[i % len(rich_clients)]
            else:
                client_id = i % num_clients
        
        client_assignments[client_id].append(identity_row)
    
    print(f"Distributing {len(non_rare_identities)} non-rare identities...")
    
    # Distribute non-rare identities using weighted distribution
    # Rich clients get more identities, poor clients get fewer
    client_weights = []
    for client_id in range(num_clients):
        richness = client_richness[client_id]
        section_num = int(richness.split('_')[1])
        # Higher section number = richer = higher weight
        weight = section_num ** 2  # Exponential weighting
        client_weights.append(weight)
    
    # Normalize weights
    client_weights = np.array(client_weights)
    client_weights = client_weights / client_weights.sum()
    
    # Distribute non-rare identities
    non_rare_identities = non_rare_identities.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for _, identity_row in non_rare_identities.iterrows():
        client_id = np.random.choice(num_clients, p=client_weights)
        client_assignments[client_id].append(identity_row)
    
    # Create final assignment DataFrame
    assignment_records = []
    for client_id, identities in client_assignments.items():
        for identity_row in identities:
            assignment_records.append({
                'client_id': client_id,
                'identity_id': identity_row['identity_id'],
                'sample_count': identity_row['sample_count'],
                'rarity_class': identity_row['rarity_class'],
                'richness_section': identity_row['richness_section'],
                'client_richness': client_richness[client_id]
            })
    
    assignment_df = pd.DataFrame(assignment_records)
    
    # Display final distribution
    print(f"\nFinal client distribution:")
    client_stats = assignment_df.groupby('client_id').agg({
        'identity_id': 'count',
        'sample_count': 'sum',
        'rarity_class': lambda x: (x == 'rare').sum()
    }).rename(columns={
        'identity_id': 'num_identities',
        'sample_count': 'total_samples',
        'rarity_class': 'rare_count'
    })
    
    client_stats['client_richness'] = [client_richness[cid] for cid in client_stats.index]
    client_stats['rare_percentage'] = (client_stats['rare_count'] / client_stats['num_identities'] * 100).round(2)
    
    print(client_stats)
    
    # Create client folders and CSV files
    create_client_folders(assignment_df, source_df, test_set_percentage, output_folder)
    
    return assignment_df, client_stats

def create_client_folders(assignment_df, source_df, test_set_percentage, output_folder):
    """
    Create client folders with train, query, and gallery CSV files matching source structure
    """
    # Create main output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\nCreating client folders in: {output_folder}")
    
    for client_id in assignment_df['client_id'].unique():
        client_data = assignment_df[assignment_df['client_id'] == client_id].copy()
        
        # Create client folder
        client_folder = os.path.join(output_folder, str(client_id))
        os.makedirs(client_folder, exist_ok=True)
        
        train_records = []
        query_records = []
        gallery_records = []
        
        for _, identity_row in client_data.iterrows():
            identity_id = identity_row['identity_id']
            total_samples = identity_row['sample_count']
            rarity_class = identity_row['rarity_class']
            
            # Get all samples for this identity from source data
            identity_samples = source_df[source_df['identity'] == identity_id].copy()
            identity_samples = identity_samples.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Calculate splits ensuring minimum requirements
            test_set_size = max(3, int(total_samples * test_set_percentage))  # Minimum 3 for test
            train_size = total_samples - test_set_size
            
            # Ensure minimum 2 training samples
            if train_size < 2:
                train_size = 2
                test_set_size = total_samples - train_size
            
            # For test set: minimum 1 query, 2 gallery
            if test_set_size >= 3:
                query_size = 1
                gallery_size = max(2, test_set_size - query_size)
            else:
                # If not enough for minimum, skip this identity
                print(f"Warning: Identity {identity_id} has only {total_samples} samples, skipping")
                continue
            
            # Split the actual samples
            train_samples = identity_samples[:train_size].copy()
            query_samples = identity_samples[train_size:train_size + query_size].copy()
            gallery_samples = identity_samples[train_size + query_size:train_size + query_size + gallery_size].copy()
            
            # Add rarity class to all samples
            train_samples['rarity_class'] = rarity_class
            query_samples['rarity_class'] = rarity_class
            gallery_samples['rarity_class'] = rarity_class
            
            train_records.append(train_samples)
            query_records.append(query_samples)
            gallery_records.append(gallery_samples)
        
        # Create DataFrames and save CSV files
        if train_records:
            train_df = pd.concat(train_records, ignore_index=True)
            query_df = pd.concat(query_records, ignore_index=True)
            gallery_df = pd.concat(gallery_records, ignore_index=True)
            
            # Save CSV files
            train_df.to_csv(os.path.join(client_folder, 'train.csv'), index=False)
            query_df.to_csv(os.path.join(client_folder, 'query.csv'), index=False)
            gallery_df.to_csv(os.path.join(client_folder, 'gallery.csv'), index=False)
            
            print(f"Client {client_id}:")
            print(f"  Train: {len(train_df)} samples")
            print(f"  Query: {len(query_df)} samples") 
            print(f"  Gallery: {len(gallery_df)} samples")
            print(f"  Test percentage: {((len(query_df) + len(gallery_df)) / (len(train_df) + len(query_df) + len(gallery_df)) * 100):.1f}%")
            
            # Verify rarity distribution
            rare_train = (train_df['rarity_class'] == 'rare').sum()
            rare_total = (train_df['rarity_class'] == 'rare').sum() + (query_df['rarity_class'] == 'rare').sum() + (gallery_df['rarity_class'] == 'rare').sum()
            print(f"  Rare cases: {rare_total} total, {rare_train} in training")
        else:
            print(f"Warning: Client {client_id} has no valid data after filtering")
    
    print(f"\nClient folders created successfully in: {output_folder}")


# Example usage with the provided data
if __name__ == "__main__":
    # Load dataset
    dataset = LeopardID2022('/home/wellvw12/leopard')
    source_df = dataset.df
    
    print(f"Loaded dataset with {len(source_df)} samples")
    print(f"Dataset columns: {source_df.columns.tolist()}")
    
    # Run the analysis
    assignment_df, client_stats = analyze_and_distribute_rarity(
        source_df, 
        num_clients=7, 
        richness_sections=4, 
        rare_threshold_percentile=10,
        test_set_percentage=0.1,
        output_folder='client_rarity_data9ß',
        min_samples_threshold=5
    )
    
    # Save results
    assignment_df.to_csv('/home/wellvw12/fedReID/client_assignments_with_rarity.csv', index=False)
    client_stats.to_csv('/home/wellvw12/fedReID/client_statistics.csv')
    
    print(f"\nResults saved to:")
    print(f"- Client assignments: /home/wellvw12/fedReID/client_assignments_with_rarity.csv")
    print(f"- Client statistics: /home/wellvw12/fedReID/client_statistics.csv")
    print(f"- Client folders: /home/wellvw12/fedReID/client_rarity_data_9/")