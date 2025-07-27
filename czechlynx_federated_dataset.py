#!/usr/bin/env python3
"""
Czech Lynx Federated Dataset classes for wildlife_datasets integration.

This module provides WildlifeDataset subclasses for each federated client
optimized for FedGDK knowledge distillation experiments.
"""

import pandas as pd
from pathlib import Path
from wildlife_datasets import datasets
import os


class CzechLynxFederatedBase(datasets.WildlifeDataset):
    """Base class for Czech Lynx federated clients."""
    
    def __init__(self, data_dir: str, client_metadata_file: str):
        """
        Initialize federated client dataset.
        
        Args:
            data_dir: Directory containing the Czech Lynx images
            client_metadata_file: Path to client-specific metadata CSV
        """
        self.client_metadata_file = client_metadata_file
        super().__init__(data_dir)
    
    def create_catalogue(self) -> pd.DataFrame:
        """Create catalogue from client metadata file."""
        if not os.path.exists(self.client_metadata_file):
            raise FileNotFoundError(f"Client metadata file not found: {self.client_metadata_file}")
        
        df = pd.read_csv(self.client_metadata_file)
        
        # Ensure required columns exist
        required_cols = ['image_id', 'identity', 'path']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate paths exist (optional, can be slow for large datasets)
        if hasattr(self, 'validate_paths') and self.validate_paths:
            missing_files = []
            for path in df['path'].head(10):  # Check first 10 files only
                full_path = Path(self.data_dir) / path
                if not full_path.exists():
                    missing_files.append(str(full_path))
            
            if missing_files:
                print(f"Warning: Some image files not found. First few missing: {missing_files[:3]}")
        
        return self.finalize_catalogue(df)


class CzechLynxBeskydyClient(CzechLynxFederatedBase):
    """
    Client 0: Beskydy region (Northern/Eastern Czech Republic)
    
    Characteristics:
    - Mountainous terrain with diverse elevations
    - Different vegetation and climate conditions
    - Full age range of lynx individuals
    - Serves as geographic test region in original splits
    """
    
    def __init__(self, data_dir: str, metadata_dir: str = "/home/wellvw12/fedwild/czechlynx_federated"):
        client_metadata = Path(metadata_dir) / "client_0_beskydy_metadata.csv"
        super().__init__(data_dir, str(client_metadata))


class CzechLynxNPSClient(CzechLynxFederatedBase):
    """
    Client 1: NPS region (Southern Czech Republic)
    
    Characteristics:
    - Different ecosystem and forest types
    - Distinct environmental conditions from Beskydy
    - Complementary geographic knowledge
    - Part of original training split
    """
    
    def __init__(self, data_dir: str, metadata_dir: str = "/home/wellvw12/fedwild/czechlynx_federated"):
        client_metadata = Path(metadata_dir) / "client_1_nps_metadata.csv"
        super().__init__(data_dir, str(client_metadata))


class CzechLynxSumavaClient(CzechLynxFederatedBase):
    """
    Client 2: Šumava region (Western Czech Republic)
    
    Characteristics:
    - Western geographic region with unique conditions
    - Completes the geographic diversity triangle
    - Different habitat characteristics
    - Part of original training split
    """
    
    def __init__(self, data_dir: str, metadata_dir: str = "/home/wellvw12/fedwild/czechlynx_federated"):
        client_metadata = Path(metadata_dir) / "client_2_sumava_metadata.csv"
        super().__init__(data_dir, str(client_metadata))


class CzechLynxFederatedManager:
    """
    Manager class to handle all federated clients and provide unified access.
    """
    
    def __init__(self, data_dir: str, metadata_dir: str = "/home/wellvw12/fedwild/czechlynx_federated"):
        """
        Initialize all federated clients.
        
        Args:
            data_dir: Directory containing Czech Lynx images
            metadata_dir: Directory containing client metadata files
        """
        self.data_dir = data_dir
        self.metadata_dir = metadata_dir
        self.clients = {}
        
        # Initialize all clients
        self.clients['beskydy'] = CzechLynxBeskydyClient(data_dir, metadata_dir)
        self.clients['nps'] = CzechLynxNPSClient(data_dir, metadata_dir)
        self.clients['sumava'] = CzechLynxSumavaClient(data_dir, metadata_dir)
    
    def get_client(self, client_name: str) -> CzechLynxFederatedBase:
        """Get specific client dataset."""
        if client_name not in self.clients:
            available = list(self.clients.keys())
            raise ValueError(f"Client '{client_name}' not found. Available: {available}")
        return self.clients[client_name]
    
    def get_all_clients(self) -> dict:
        """Get all client datasets."""
        return self.clients
    
    def print_federation_summary(self):
        """Print summary of federated setup."""
        print("=== CZECH LYNX FEDERATED DATASET SUMMARY ===")
        total_samples = 0
        total_individuals = set()
        
        for client_name, client in self.clients.items():
            df = client.df
            samples = len(df)
            individuals = df['identity'].nunique()
            unique_individuals = set(df['identity'].unique())
            
            total_samples += samples
            total_individuals.update(unique_individuals)
            
            print(f"{client_name.upper()} Client:")
            print(f"  - Samples: {samples}")
            print(f"  - Individuals: {individuals}")
            print(f"  - Age range: {df['relative_age'].min()}-{df['relative_age'].max()}")
            
            if 'source_region' in df.columns:
                regions = df['source_region'].unique()
                print(f"  - Regions: {regions}")
        
        print(f"\nFEDERATION TOTALS:")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Total unique individuals: {len(total_individuals)}")
        
        # Check for individual overlap
        individual_overlaps = []
        client_names = list(self.clients.keys())
        for i in range(len(client_names)):
            for j in range(i+1, len(client_names)):
                client1_individuals = set(self.clients[client_names[i]].df['identity'])
                client2_individuals = set(self.clients[client_names[j]].df['identity'])
                overlap = len(client1_individuals & client2_individuals)
                individual_overlaps.append(f"{client_names[i]}-{client_names[j]}: {overlap}")
        
        print(f"  - Individual overlaps: {', '.join(individual_overlaps)}")


# Example usage and testing functions
def test_federated_dataset(data_dir: str = "/path/to/czechlynx/images"):
    """Test the federated dataset implementation."""
    try:
        # Initialize federation manager
        federation = CzechLynxFederatedManager(data_dir)
        
        # Print summary
        federation.print_federation_summary()
        
        # Test individual clients
        for client_name in ['beskydy', 'nps', 'sumava']:
            client = federation.get_client(client_name)
            print(f"\nTesting {client_name} client:")
            print(f"DataFrame shape: {client.df.shape}")
            print(f"Sample data:\n{client.df.head(2)}")
        
        print("\n✅ All federated clients initialized successfully!")
        return federation
        
    except Exception as e:
        print(f"❌ Error testing federated dataset: {e}")
        return None


if __name__ == "__main__":
    # Test with dummy data directory
    test_federated_dataset("/dummy/path")