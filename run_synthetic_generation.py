#!/usr/bin/env python3
"""
Quick script to generate synthetic data with common configurations
"""

import os
import sys
from generate_synthetic_data import SyntheticDataGenerator

def generate_kd_test_data():
    """Generate data specifically for knowledge distillation testing"""
    print("Generating Knowledge Distillation Test Dataset")
    print("Configuration: 2 clients - one large (100 samples), one small (10 samples)")
    
    # Override config for KD testing
    generator = SyntheticDataGenerator(
        output_dir="synthetic_kd_test",
        n_clients=2,
        species='leopard'
    )
    
    # Custom config for your specific use case
    generator.config.update({
        'train_samples_per_id': [10, 1],  # Client 0: 10 samples/ID, Client 1: 1 sample/ID  
        'train_ids_per_client': [10, 10],  # Both clients have same number of IDs (10 each)
        'query_ids_per_client': 3,  # 3 query identities per client
        'queries_per_id': 1,  # 1 query image per identity
        'gallery_samples_per_query_id': 2,  # 2 gallery images per query identity
    })
    
    generator.generate()
    print(f"\nDataset created! Client 0: ~100 train images, Client 1: ~10 train images")
    print("Use with: python main.py --data_dir synthetic_kd_test --datasets 0,1")

def generate_quick_test():
    """Generate minimal dataset for quick testing"""
    print("Generating Quick Test Dataset")
    
    generator = SyntheticDataGenerator(
        output_dir="synthetic_quick",
        n_clients=2,
        species='leopard'
    )
    
    generator.config.update({
        'train_samples_per_id': [3, 2],  # Few samples per ID
        'train_ids_per_client': [5, 4],  # Few identities
        'query_ids_per_client': 2,  # Just 2 query IDs
        'queries_per_id': 1,
        'gallery_samples_per_query_id': 2,
    })
    
    generator.generate()
    print(f"\nMinimal dataset created for quick testing!")

def generate_balanced():
    """Generate balanced dataset for standard federated learning"""
    print("Generating Balanced Dataset")
    
    generator = SyntheticDataGenerator(
        output_dir="synthetic_balanced",
        n_clients=3,
        species='leopard'
    )
    
    generator.config.update({
        'train_samples_per_id': [5, 5, 5],
        'train_ids_per_client': [12, 12, 12],
        'query_ids_per_client': 4,
        'queries_per_id': 1,
        'gallery_samples_per_query_id': 3,
    })
    
    generator.generate()
    print(f"\nBalanced dataset created!")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'kd':
            generate_kd_test_data()
        elif mode == 'quick':
            generate_quick_test()
        elif mode == 'balanced':
            generate_balanced()
        else:
            print("Usage: python run_synthetic_generation.py [kd|quick|balanced]")
    else:
        print("Available modes:")
        print("  kd      - Generate KD test data (2 clients: 100 vs 10 samples)")
        print("  quick   - Generate minimal data for quick testing")
        print("  balanced- Generate balanced data for standard FL")
        print("\nExample: python run_synthetic_generation.py kd")