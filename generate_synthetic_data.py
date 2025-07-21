#!/usr/bin/env python3
"""
Synthetic Data Generator for FedReID Testing

Generates synthetic leopard-like dataset for quick testing of federated learning
and knowledge distillation methods. Creates multiple clients with non-overlapping
identities and configurable data distribution.

Usage:
    python generate_synthetic_data.py --output_dir synthetic_data --n_clients 3
"""

import os
import csv
import uuid
import argparse
import random
from PIL import Image
import numpy as np
from pathlib import Path


class SyntheticDataGenerator:
    def __init__(self, output_dir, n_clients=3, species='leopard'):
        self.output_dir = Path(output_dir)
        self.n_clients = n_clients
        self.species = species
        
        # Import configuration
        try:
            from synthetic_config import ACTIVE_CONFIG
            self.config = ACTIVE_CONFIG.copy()
        except ImportError:
            print("Warning: synthetic_config.py not found, using default configuration")
            self.config = {
                'train_samples_per_id': [4, 6, 8],
                'train_ids_per_client': [20, 15, 10],
                'query_ids_per_client': 5,
                'queries_per_id': 1,
                'gallery_samples_per_query_id': 3,
            }
        
        # Add image properties
        self.config.update({
            'image_width': 256,
            'image_height': 256,
            'orientations': ['left', 'right', 'front', 'back'],
        })
        
        # Ensure n_clients matches config
        if n_clients != len(self.config['train_samples_per_id']):
            print(f"Warning: n_clients ({n_clients}) doesn't match config length ({len(self.config['train_samples_per_id'])})")
            print(f"Using config length: {len(self.config['train_samples_per_id'])}")
            self.n_clients = len(self.config['train_samples_per_id'])
        
        print(f"Generating synthetic data for {n_clients} clients")
        print(f"Train config: {self.config['train_samples_per_id']} samples/ID, {self.config['train_ids_per_client']} IDs/client")
        
    def generate_synthetic_image(self, width=256, height=256):
        """Generate a synthetic image with random patterns"""
        # Create random noise image with some structure
        image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Add some structure (circular patterns to mimic spots)
        center_x, center_y = width // 2, height // 2
        for _ in range(random.randint(5, 15)):  # Random number of "spots"
            spot_x = random.randint(0, width-1)
            spot_y = random.randint(0, height-1)
            spot_radius = random.randint(10, 30)
            
            # Create circular region
            y, x = np.ogrid[:height, :width]
            mask = (x - spot_x)**2 + (y - spot_y)**2 <= spot_radius**2
            
            # Darken the spot area
            if np.any(mask):
                image[mask] = image[mask] * 0.3
        
        return Image.fromarray(image)
    
    def generate_bbox(self, width=256, height=256):
        """Generate random bounding box"""
        x = random.randint(10, width//4)
        y = random.randint(10, height//4)
        w = random.randint(width//2, width - x - 10)
        h = random.randint(height//2, height - y - 10)
        return f"[{x}, {y}, {w}, {h}]"
    
    def generate_identity_pool(self):
        """Generate unique identity UUIDs for all clients"""
        total_train_ids = sum(self.config['train_ids_per_client'])
        total_test_ids = self.n_clients * self.config['query_ids_per_client']
        
        # Generate unique identities (no overlap between clients or train/test)
        all_identities = [str(uuid.uuid4()) for _ in range(total_train_ids + total_test_ids)]
        
        identity_pool = {
            'train': [],
            'test': []
        }
        
        # Assign training identities to clients
        idx = 0
        for client_id in range(self.n_clients):
            n_ids = self.config['train_ids_per_client'][client_id % len(self.config['train_ids_per_client'])]
            client_identities = all_identities[idx:idx + n_ids]
            identity_pool['train'].append(client_identities)
            idx += n_ids
        
        # Assign test identities to clients
        for client_id in range(self.n_clients):
            n_ids = self.config['query_ids_per_client']
            client_identities = all_identities[idx:idx + n_ids]
            identity_pool['test'].append(client_identities)
            idx += n_ids
        
        print(f"Generated {total_train_ids} training identities and {total_test_ids} test identities")
        return identity_pool
    
    def create_client_directories(self):
        """Create directory structure for all clients"""
        for client_id in range(self.n_clients):
            client_dir = self.output_dir / str(client_id)
            client_dir.mkdir(parents=True, exist_ok=True)
            
            # Create images subdirectory
            images_dir = client_dir / 'images'
            images_dir.mkdir(exist_ok=True)
            
            print(f"Created directories for client {client_id}")
    
    def generate_client_data(self, client_id, identity_pool):
        """Generate training and testing data for a specific client"""
        client_dir = self.output_dir / str(client_id)
        images_dir = client_dir / 'images'
        
        # Get identities for this client
        train_identities = identity_pool['train'][client_id]
        test_identities = identity_pool['test'][client_id]
        
        # Generate training data
        train_data = []
        image_counter = client_id * 10000  # Offset to avoid ID conflicts
        
        for identity in train_identities:
            n_samples = self.config['train_samples_per_id'][client_id % len(self.config['train_samples_per_id'])]
            
            for sample_idx in range(n_samples):
                image_id = image_counter
                image_filename = f"synthetic_{client_id}_{image_id:06d}.jpg"
                image_path = images_dir / image_filename
                
                # Generate and save synthetic image
                synthetic_image = self.generate_synthetic_image()
                synthetic_image.save(image_path)
                
                # Create CSV entry
                train_data.append({
                    'image_id': image_id,
                    'identity': identity,
                    'path': f"synthetic_data/{client_id}/images/{image_filename}",
                    'bbox': self.generate_bbox(),
                    'date': '',
                    'orientation': random.choice(self.config['orientations']),
                    'segmentation': '',
                    'species': self.species
                })
                
                image_counter += 1
        
        # Generate test data (query + gallery)
        query_data = []
        gallery_data = []
        
        for identity in test_identities:
            # Generate query image(s)
            for query_idx in range(self.config['queries_per_id']):
                image_id = image_counter
                image_filename = f"synthetic_{client_id}_{image_id:06d}.jpg"
                image_path = images_dir / image_filename
                
                synthetic_image = self.generate_synthetic_image()
                synthetic_image.save(image_path)
                
                query_data.append({
                    'image_id': image_id,
                    'identity': identity,
                    'path': f"synthetic_data/{client_id}/images/{image_filename}",
                    'bbox': self.generate_bbox(),
                    'date': '',
                    'orientation': random.choice(self.config['orientations']),
                    'segmentation': '',
                    'species': self.species
                })
                
                image_counter += 1
            
            # Generate gallery images
            for gallery_idx in range(self.config['gallery_samples_per_query_id']):
                image_id = image_counter
                image_filename = f"synthetic_{client_id}_{image_id:06d}.jpg"
                image_path = images_dir / image_filename
                
                synthetic_image = self.generate_synthetic_image()
                synthetic_image.save(image_path)
                
                gallery_data.append({
                    'image_id': image_id,
                    'identity': identity,
                    'path': f"synthetic_data/{client_id}/images/{image_filename}",
                    'bbox': self.generate_bbox(),
                    'date': '',
                    'orientation': random.choice(self.config['orientations']),
                    'segmentation': '',
                    'species': self.species
                })
                
                image_counter += 1
        
        # Save CSV files
        self.save_csv(client_dir / 'train.csv', train_data)
        self.save_csv(client_dir / 'query.csv', query_data)
        self.save_csv(client_dir / 'gallery.csv', gallery_data)
        
        print(f"Client {client_id}: {len(train_data)} train, {len(query_data)} query, {len(gallery_data)} gallery")
        return len(train_data), len(query_data), len(gallery_data)
    
    def save_csv(self, filepath, data):
        """Save data to CSV file with proper headers"""
        fieldnames = ['image_id', 'identity', 'path', 'bbox', 'date', 'orientation', 'segmentation', 'species']
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    def generate_dataset_summary(self):
        """Generate a summary of the created dataset"""
        summary_path = self.output_dir / 'dataset_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("Synthetic Dataset Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of clients: {self.n_clients}\n")
            f.write(f"Species: {self.species}\n\n")
            
            f.write("Training Configuration:\n")
            f.write(f"  - Samples per ID: {self.config['train_samples_per_id']}\n")
            f.write(f"  - IDs per client: {self.config['train_ids_per_client']}\n\n")
            
            f.write("Testing Configuration:\n")
            f.write(f"  - Query IDs per client: {self.config['query_ids_per_client']}\n")
            f.write(f"  - Queries per ID: {self.config['queries_per_id']}\n")
            f.write(f"  - Gallery samples per query ID: {self.config['gallery_samples_per_query_id']}\n\n")
            
            f.write("Directory Structure:\n")
            for client_id in range(self.n_clients):
                f.write(f"  {client_id}/\n")
                f.write(f"    ├── train.csv\n")
                f.write(f"    ├── query.csv\n")
                f.write(f"    ├── gallery.csv\n")
                f.write(f"    └── images/\n")
        
        print(f"Dataset summary saved to {summary_path}")
    
    def generate(self):
        """Generate complete synthetic dataset"""
        print(f"Creating synthetic dataset in {self.output_dir}")
        
        # Create directories
        self.create_client_directories()
        
        # Generate identity pool
        identity_pool = self.generate_identity_pool()
        
        # Generate data for each client
        total_stats = {'train': 0, 'query': 0, 'gallery': 0}
        
        for client_id in range(self.n_clients):
            print(f"\nGenerating data for client {client_id}...")
            train_count, query_count, gallery_count = self.generate_client_data(client_id, identity_pool)
            
            total_stats['train'] += train_count
            total_stats['query'] += query_count
            total_stats['gallery'] += gallery_count
        
        # Generate summary
        self.generate_dataset_summary()
        
        print(f"\n" + "="*50)
        print("Dataset generation completed!")
        print(f"Total: {total_stats['train']} train, {total_stats['query']} query, {total_stats['gallery']} gallery")
        print(f"Output directory: {self.output_dir}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset for FedReID testing')
    parser.add_argument('--output_dir', default='synthetic_data', help='Output directory')
    parser.add_argument('--n_clients', type=int, default=3, help='Number of federated clients')
    parser.add_argument('--species', default='leopard', help='Species name for metadata')
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = SyntheticDataGenerator(
        output_dir=args.output_dir,
        n_clients=args.n_clients,
        species=args.species
    )
    
    generator.generate()


if __name__ == '__main__':
    main()