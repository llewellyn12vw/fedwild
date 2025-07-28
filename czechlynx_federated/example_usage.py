#!/usr/bin/env python3
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
    
    print(f"\n{client_name.upper()} Client:")
    print(f"  Training: {len(train_data)} samples ({train_data['identity'].nunique()} IDs)")
    print(f"  Query: {len(query_data)} samples ({query_data['identity'].nunique()} IDs)")
    print(f"  Gallery: {len(gallery_data)} samples ({gallery_data['identity'].nunique()} IDs)")

# Example: Filter by specific client and split
beskydy_train = metadata[(metadata['client'] == 'beskydy') & (metadata['split'] == 'train')]
print(f"\nBeskydy training data: {len(beskydy_train)} samples")
