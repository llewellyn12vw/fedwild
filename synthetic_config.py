#!/usr/bin/env python3
"""
Configuration file for synthetic data generation

Modify the parameters below to customize your synthetic dataset generation.
Then run: python generate_synthetic_data.py
"""

# Example configurations for different testing scenarios

# Balanced clients (good for standard federated learning)
BALANCED_CONFIG = {
    'train_samples_per_id': [5, 5, 5],  # Same samples per ID for all clients
    'train_ids_per_client': [15, 15, 15],  # Same number of IDs for all clients
    'query_ids_per_client': 5,
    'queries_per_id': 1,
    'gallery_samples_per_query_id': 3,
}

# Imbalanced clients (good for testing robustness)
IMBALANCED_CONFIG = {
    'train_samples_per_id': [10, 5, 2],  # Client 0 has more samples per ID
    'train_ids_per_client': [20, 15, 8],  # Client 0 has more identities
    'query_ids_per_client': 15,
    'queries_per_id': 2,
    'gallery_samples_per_query_id': 5,
}

# Small test config (for quick testing)
QUICK_TEST_CONFIG = {
    'train_samples_per_id': [3, 4, 2],  # Few samples per ID
    'train_ids_per_client': [5, 6, 4],  # Few identities per client
    'query_ids_per_client': 2,  # Few query IDs
    'queries_per_id': 1,
    'gallery_samples_per_query_id': 2,  # Few gallery samples
}

# Knowledge distillation focused (one large, one small client)
KD_TEST_CONFIG = {
    'train_samples_per_id': [8, 3],  # Large vs small client
    'train_ids_per_client': [25, 8],  # Many vs few identities  
    'query_ids_per_client': 5,
    'queries_per_id': 1,
    'gallery_samples_per_query_id': 3,
}

# Multi-client scenario (many small clients)
MULTI_CLIENT_CONFIG = {
    'train_samples_per_id': [4, 3, 5, 2, 4],  # 5 clients with varying samples
    'train_ids_per_client': [10, 8, 12, 6, 9],  # Different ID counts
    'query_ids_per_client': 4,
    'queries_per_id': 1,
    'gallery_samples_per_query_id': 3,
}

# Current active configuration (change this to switch configs)
ACTIVE_CONFIG = IMBALANCED_CONFIG

def print_config_summary():
    """Print summary of current configuration"""
    print("Current Synthetic Data Configuration:")
    print("=" * 40)
    print(f"Clients: {len(ACTIVE_CONFIG['train_samples_per_id'])}")
    print(f"Train samples per ID: {ACTIVE_CONFIG['train_samples_per_id']}")
    print(f"Train IDs per client: {ACTIVE_CONFIG['train_ids_per_client']}")
    print(f"Query IDs per client: {ACTIVE_CONFIG['query_ids_per_client']}")
    print(f"Queries per ID: {ACTIVE_CONFIG['queries_per_id']}")
    print(f"Gallery samples per query ID: {ACTIVE_CONFIG['gallery_samples_per_query_id']}")
    print("=" * 40)
    
    # Calculate totals
    total_train = sum(s * i for s, i in zip(ACTIVE_CONFIG['train_samples_per_id'], ACTIVE_CONFIG['train_ids_per_client']))
    n_clients = len(ACTIVE_CONFIG['train_samples_per_id'])
    total_query = n_clients * ACTIVE_CONFIG['query_ids_per_client'] * ACTIVE_CONFIG['queries_per_id']
    total_gallery = n_clients * ACTIVE_CONFIG['query_ids_per_client'] * ACTIVE_CONFIG['gallery_samples_per_query_id']
    
    print(f"Total train images: {total_train}")
    print(f"Total query images: {total_query}")
    print(f"Total gallery images: {total_gallery}")
    print(f"Total images: {total_train + total_query + total_gallery}")

if __name__ == '__main__':
    print_config_summary()