#!/usr/bin/env python3
"""
Test script for Czech Lynx Federated metadata files (without WildlifeDataset validation).
"""

import pandas as pd
import os
from pathlib import Path

def test_metadata_files():
    """Test the generated metadata files directly."""
    print("=== TESTING GENERATED METADATA FILES ===")
    
    metadata_dir = "/home/wellvw12/fedwild/czechlynx_federated"
    
    expected_files = [
        "client_0_beskydy_metadata.csv",
        "client_1_nps_metadata.csv", 
        "client_2_sumava_metadata.csv"
    ]
    
    client_data = {}
    
    for filename in expected_files:
        filepath = Path(metadata_dir) / filename
        
        if not filepath.exists():
            print(f"❌ Missing file: {filepath}")
            continue
            
        try:
            df = pd.read_csv(filepath)
            client_name = filename.replace("_metadata.csv", "").replace("client_", "")
            client_data[client_name] = df
            
            print(f"✅ {client_name.upper()} client metadata loaded:")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Individuals: {df['identity'].nunique()}")
            print(f"   Age range: {df['relative_age'].min():.0f}-{df['relative_age'].max():.0f}")
            print(f"   Sample paths: {df['path'].iloc[0]}")
            print()
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
    
    return client_data


def analyze_federation_properties(client_data):
    """Analyze federated learning properties."""
    if not client_data:
        return
    
    print("=== FEDERATION ANALYSIS ===")
    
    total_samples = sum(len(df) for df in client_data.values())
    total_individuals = set()
    
    for client_name, df in client_data.items():
        total_individuals.update(df['identity'].unique())
    
    print(f"Total samples: {total_samples}")
    print(f"Total unique individuals: {len(total_individuals)}")
    
    # Check individual overlap between clients
    print("\nIndividual overlap analysis:")
    client_names = list(client_data.keys())
    for i, client1 in enumerate(client_names):
        for j, client2 in enumerate(client_names[i+1:], i+1):
            individuals1 = set(client_data[client1]['identity'])
            individuals2 = set(client_data[client2]['identity']) 
            overlap = len(individuals1 & individuals2)
            print(f"  {client1} ↔ {client2}: {overlap} shared individuals")
    
    # Geographic analysis
    print("\nGeographic distribution:")
    for client_name, df in client_data.items():
        regions = df['source_region'].unique()
        locations = df['location'].nunique()
        
        # Calculate coordinate ranges
        if not df['latitude'].isna().all():
            lat_range = df['latitude'].max() - df['latitude'].min()
            lon_range = df['longitude'].max() - df['longitude'].min()
            print(f"  {client_name}: {regions} region(s), {locations} locations, "
                  f"lat_range={lat_range:.3f}°, lon_range={lon_range:.3f}°")
    
    # Age distribution
    print("\nAge distribution:")
    for client_name, df in client_data.items():
        age_counts = df['relative_age'].value_counts().sort_index()
        age_stats = {
            'min': df['relative_age'].min(),
            'max': df['relative_age'].max(), 
            'mean': df['relative_age'].mean(),
            'most_common': age_counts.index[0] if not age_counts.empty else 'N/A'
        }
        print(f"  {client_name}: min={age_stats['min']:.0f}, max={age_stats['max']:.0f}, "
              f"mean={age_stats['mean']:.1f}, most_common={age_stats['most_common']}")
    
    # Knowledge distillation assessment
    print("\n=== KNOWLEDGE DISTILLATION SUITABILITY ===")
    
    # Check for non-IID characteristics
    non_iid_score = 0
    
    # Geographic separation
    geographic_separation = True
    regions_per_client = {}
    for client_name, df in client_data.items():
        regions = set(df['source_region'].unique())
        regions_per_client[client_name] = regions
        
    for i, client1 in enumerate(client_names):
        for j, client2 in enumerate(client_names[i+1:], i+1):
            if regions_per_client[client1] & regions_per_client[client2]:
                geographic_separation = False
                break
    
    if geographic_separation:
        non_iid_score += 3
        print("✅ Perfect geographic separation (Non-IID: HIGH)")
    else:
        print("⚠️  Some geographic overlap detected")
    
    # Age diversity
    age_ranges = []
    for client_name, df in client_data.items():
        age_range = df['relative_age'].max() - df['relative_age'].min()
        age_ranges.append(age_range)
    
    if all(r > 3 for r in age_ranges):
        non_iid_score += 2
        print("✅ Good age diversity across clients (Non-IID: MEDIUM)")
    else:
        print("⚠️  Limited age diversity in some clients")
    
    # Individual distribution
    min_individuals = min(df['identity'].nunique() for df in client_data.values())
    max_individuals = max(df['identity'].nunique() for df in client_data.values())
    
    if min_individuals > 50 and max_individuals/min_individuals < 3:
        non_iid_score += 1
        print("✅ Balanced individual distribution (Non-IID: LOW)")
    
    # Overall assessment
    print(f"\nNon-IID Score: {non_iid_score}/6")
    if non_iid_score >= 5:
        print("🎯 EXCELLENT for knowledge distillation experiments!")
    elif non_iid_score >= 3:
        print("👍 GOOD for knowledge distillation experiments")
    else:
        print("⚠️  May need additional non-IID characteristics")


def generate_federated_config():
    """Generate configuration for FedGDK experiments."""
    print("\n=== FEDGDK EXPERIMENT CONFIGURATION ===")
    
    config = """
# Recommended FedGDK Configuration for Czech Lynx Dataset

python main.py --fedgkd --ex_name czechlynx/GeoSplit \\
    --fedgkd_distillation_coeff 0.15 \\
    --fedgkd_temperature 2.0 \\
    --fedgkd_buffer_length 15 \\
    --fedgkd_start_round 20 \\
    --local_epoch 3 \\
    --num_of_clients 3 \\
    --total_rounds 100 \\
    --metadata_file_client0 /home/wellvw12/fedwild/czechlynx_federated/client_0_beskydy_metadata.csv \\
    --metadata_file_client1 /home/wellvw12/fedwild/czechlynx_federated/client_1_nps_metadata.csv \\
    --metadata_file_client2 /home/wellvw12/fedwild/czechlynx_federated/client_2_sumava_metadata.csv \\
    --dataset_type czechlynx_federated
    
# Client Characteristics:
# - Client 0 (Beskydy): 17,689 samples, 69 individuals, mountainous terrain
# - Client 1 (NPS): 13,309 samples, 106 individuals, southern forests  
# - Client 2 (Sumava): 6,442 samples, 82 individuals, western regions
"""
    
    print(config)


def main():
    """Run all tests."""
    print("CZECH LYNX FEDERATED METADATA TESTING")
    print("=" * 50)
    
    # Test metadata files
    client_data = test_metadata_files()
    
    # Analyze federation properties
    analyze_federation_properties(client_data)
    
    # Generate config
    generate_federated_config()
    
    print("=" * 50)
    print("METADATA TESTING COMPLETE!")
    
    if client_data and len(client_data) == 3:
        print("✅ All metadata files generated successfully!")
        print("✅ Ready for Czech Lynx federated learning experiments!")
    else:
        print("❌ Some metadata files missing or invalid")


if __name__ == "__main__":
    main()