import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
from math import floor
import math
import torch
import torchvision
from wildlife_tools.data.dataset import WildlifeDataset
from wildlife_datasets.datasets import MacaqueFaces, Cows2021v2, LeopardID2022, HyenaID2022
import torchvision.transforms as T
from wildlife_datasets import datasets, loader, metrics
from wildlife_datasets import splits
import os
from math import floor
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split

N_CLIENTS = 4  # 6 clients + 1 test set
OUTPUT_DIR = "/home/wellvw12/hyena_6/clients"
DIRICHLET_ALPHA = 0.5
MIN_SAMPLES_PER_CLIENT = 90
TEST_SIZE = 0.25
QUERY_RATIO = 0.2  # Ratio of query samples to gallery samples
MIN_GALLERY_PER_QUERY = 2
MAX_QUERIES_PER_ID = 8
KD_TARGET_SAMPLES = 600  # Total target samples for KD set
MIN_KD_SAMPLES_PER_ID = 3
MAX_KD_SAMPLES_PER_ID = 10
KD_ID_RATIO = 0.3  # Percentage of identities to include in KD set
SEPERATE_TEST = 0.085
DIRICHLET_ALPHAS = [0.45, 0.45, 0.6, 0.9]

# Global option to remove test sets (query/gallery) - set manually
REMOVE_TEST_SETS = True

# Global option to use orientation-based splits instead of identity-based Dirichlet
USE_ORIENTATION_SPLITS = True 

def check_identity_mapping(query_set, gallery_set, metadata_path):
    """Verify consistent identity indexing between query and gallery sets."""
    # metadata = LeopardID2022(metadata_path)
    metadata = HyenaID2022(metadata_path)
    q = WildlifeDataset(query_set, metadata.root)
    g = WildlifeDataset(gallery_set, metadata.root)
    
    q_identities = q.labels_map
    g_identities = g.labels_map
    
    # Check 1: Verify all query identities exist in gallery
    unique_q = np.unique(q_identities)
    unique_g = np.unique(g_identities)
    missing = set(unique_q) - set(unique_g)
    
    if missing:
        print(f"Error: {len(missing)} query identities missing from gallery")
        print("First 5 missing:", list(missing)[:5])
        return False, {"missing_identities": list(missing)}
    
    # Check 2: Verify index positions match
    mismatches = {}
    q_indices = {id_: np.where(q_identities == id_)[0] for id_ in unique_q}
    g_indices = {id_: np.where(g_identities == id_)[0] for id_ in unique_g}
    
    for id_ in set(unique_q) & set(unique_g):
        if not np.array_equal(q_indices[id_], g_indices[id_]):
            mismatches[id_] = {
                "query_indices": q_indices[id_].tolist(),
                "gallery_indices": g_indices[id_].tolist()
            }
    
    if mismatches:
        print(f"Error: {len(mismatches)} identities have index mismatches")
        for id_, idx in list(mismatches.items())[:3]:
            print(f"{id_}:\n  Query positions: {idx['query_indices']}\n  Gallery positions: {idx['gallery_indices']}")
        return False, {"index_mismatches": mismatches}
    
    # Check 3: Verify no duplicates
    q_dupes = [id_ for id_ in unique_q if len(q_indices[id_]) > 1]
    g_dupes = [id_ for id_ in unique_g if len(g_indices[id_]) > 1]
    
    if q_dupes or g_dupes:
        print("Error: Duplicate identity indices found")
        if q_dupes: print(f"Query duplicates ({len(q_dupes)}):", q_dupes[:5])
        if g_dupes: print(f"Gallery duplicates ({len(g_dupes)}):", g_dupes[:5])
        return False, {"query_duplicates": q_dupes, "gallery_duplicates": g_dupes}
    
    print("Success: Identity mappings are consistent")
    print(f"Query identities: {len(unique_q)}")
    print(f"Gallery identities: {len(unique_g)}")
    print(f"Shared identities: {len(set(unique_q) & set(unique_g))}")
    
    return True, {
        "query_identities": len(unique_q),
        "gallery_identities": len(unique_g),
        "shared_identities": len(set(unique_q) & set(unique_g))
    }

def create_query_gallery_splits(
    df, 
    max_queries, 
    min_gallery, 
    min_samples, 
    query_ratio=QUERY_RATIO, 
    random_state=42
):
    """
    Create query/gallery splits with guaranteed gallery support.
    Tries to ensure queries make up about `query_ratio` of gallery samples.
    """
    results = {'query': [], 'gallery': []}
    id_counts = df['identity'].value_counts()
    for identity, count in id_counts.items():
        samples = df[df['identity'] == identity]
        possible_queries = min(
            max_queries, 
            max(1, int(count * query_ratio))
        )
        if count >= min_samples and possible_queries > 0 and (count - possible_queries) >= min_gallery:
            query_samples = samples.sample(possible_queries, random_state=random_state)
            results['query'].append(query_samples)
            results['gallery'].append(samples.drop(query_samples.index))
        else:
            results['gallery'].append(samples)
    query_df = pd.concat(results['query']) if results['query'] else pd.DataFrame()
    gallery_df = pd.concat(results['gallery'])
    gallery_df = gallery_df[~gallery_df.index.isin(query_df.index)]
    query_df = query_df[query_df['identity'].isin(gallery_df['identity'])]
    return query_df, gallery_df

def process_test_set(test_set_dir, metadata_path='/home/wellvw12/leopard'):
    """
    Process an individual test set directory to create validated query/gallery splits.
    
    Args:
        test_set_dir: Path to directory containing test set CSV
        metadata_path: Path to dataset root for WildlifeDataset initialization
    
    Returns:
        tuple: (success_status, report_dict)
    """
    # Path setup
    test_csv_path = os.path.join(test_set_dir, 'test.csv')
    output_query_path = os.path.join(test_set_dir, 'query.csv')
    output_gallery_path = os.path.join(test_set_dir, 'gallery.csv')
    
    if not os.path.exists(test_csv_path):
        print(f"Error: test.csv not found in {test_set_dir}")
        return False, {"error": "test.csv not found"}
    
    try:
        # Load test set
        test_df = pd.read_csv(test_csv_path)
        
        # Create query/gallery splits
        query_df, gallery_df = create_query_gallery_splits(test_df)
        
        # Verify identity mappings
        print(f"\nVerifying identity mappings for {test_set_dir}:")
        is_consistent, verification_report = check_identity_mapping(
            query_df, gallery_df, metadata_path
        )
        
        if not is_consistent:
            print("Aborting due to identity mapping issues")
            return False, verification_report
        
        # Save the validated splits
        query_df.to_csv(output_query_path, index=False)
        gallery_df.to_csv(output_gallery_path, index=False)
        
        # Print summary
        print(f"\nSuccessfully processed {test_set_dir}:")
        print(f"  Query samples: {len(query_df)}")
        print(f"  Gallery samples: {len(gallery_df)}")
        print(f"  Query identities: {len(query_df['identity'].unique())}")
        print(f"  Gallery identities: {len(gallery_df['identity'].unique())}")
        
        return True, {
            "query_samples": len(query_df),
            "gallery_samples": len(gallery_df),
            "query_identities": len(query_df['identity'].unique()),
            "gallery_identities": len(gallery_df['identity'].unique()),
            "verification_report": verification_report
        }
        
    except Exception as e:
        print(f"Error processing {test_set_dir}: {str(e)}")
        return False, {"error": str(e)}

# Load dataset
# d = LeopardID2022('/home/wellvw12/leopard')
d = HyenaID2022('/home/wellvw12/hyenaid2022')
df = d.df[~(d.df['identity'] == 'unknown')]

# Parameters


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_dataframe(df, path):
    """Safely save DataFrame, creating empty file if DataFrame is empty"""
    if len(df) == 0:
        df.head(0).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)

def create_orientation_based_assignments(df, n_clients=N_CLIENTS):
    """
    Create client assignments based on orientation distribution.
    This creates meaningful non-IID splits that better demonstrate KD benefits.
    """
    print("Creating orientation-based client assignments...")
    
    # Check if orientation column exists
    if 'orientation' not in df.columns:
        print("Warning: No 'orientation' column found. Falling back to identity-based splits.")
        return None
    
    # Get orientation distribution
    orientation_counts = df['orientation'].value_counts()
    print("Orientation distribution in remaining data:")
    print(orientation_counts)
    
    # Create balanced orientation-based splits
    # Instead of strict orientation groups, sample proportionally to ensure balance
    
    client_assignments = {i: [] for i in range(1, n_clients+1)}
    all_identities = df['identity'].unique()
    
    # Group identities by their primary orientation
    identity_orientations = {}
    for identity in all_identities:
        identity_data = df[df['identity'] == identity]
        # Get the most common orientation for this identity
        primary_orientation = identity_data['orientation'].mode().iloc[0]
        identity_orientations[identity] = primary_orientation
    
    # Define orientation priorities for each client
    client_orientation_preferences = {
        1: ['right', 'frontright'],           # Right-side specialist
        2: ['left', 'frontleft'],             # Left-side specialist  
        3: ['back', 'backright', 'backleft'], # Back-view specialist
        4: ['front', 'up', 'down']            # Front/other specialist
    }
    
    # Adjust for different numbers of clients
    if n_clients == 3:
        client_orientation_preferences = {
            1: ['right', 'frontright'],
            2: ['left', 'frontleft'],
            3: ['back', 'backright', 'backleft', 'front', 'up', 'down']
        }
    elif n_clients == 2:
        client_orientation_preferences = {
            1: ['right', 'frontright', 'left', 'frontleft'],
            2: ['back', 'backright', 'backleft', 'front', 'up', 'down']
        }
    
    # First pass: assign identities based on primary orientation
    assigned_identities = set()
    
    for client_id in range(1, min(n_clients + 1, 5)):  # Limit to available preferences
        if client_id not in client_orientation_preferences:
            continue
            
        preferred_orientations = client_orientation_preferences[client_id]
        
        # Find identities that match this client's preferred orientations
        matching_identities = [
            identity for identity in all_identities 
            if identity not in assigned_identities and 
            identity_orientations.get(identity) in preferred_orientations
        ]
        
        client_assignments[client_id].extend(matching_identities)
        assigned_identities.update(matching_identities)
        
        # Count samples for this client
        client_data = df[df['identity'].isin(matching_identities)]
        print(f"Client {client_id} ({preferred_orientations}): {len(client_data)} samples, {len(matching_identities)} identities")
    
    # Second pass: distribute remaining identities to balance client sizes
    remaining_identities = [id_ for id_ in all_identities if id_ not in assigned_identities]
    
    if remaining_identities:
        print(f"Distributing {len(remaining_identities)} remaining identities...")
        
        # Sort clients by current sample count (ascending)
        client_sizes = {}
        for client_id in range(1, n_clients + 1):
            client_data = df[df['identity'].isin(client_assignments[client_id])]
            client_sizes[client_id] = len(client_data)
        
        # Distribute remaining identities to balance sizes
        for identity in remaining_identities:
            # Find client with fewest samples
            smallest_client = min(client_sizes, key=client_sizes.get)
            client_assignments[smallest_client].append(identity)
            
            # Update size count
            identity_data = df[df['identity'] == identity]
            client_sizes[smallest_client] += len(identity_data)
    
    # Final verification and balancing
    print("\nFinal client assignments:")
    min_samples = MIN_SAMPLES_PER_CLIENT
    
    for client_id in range(1, n_clients + 1):
        client_data = df[df['identity'].isin(client_assignments[client_id])]
        print(f"Client {client_id}: {len(client_data)} samples, {len(client_assignments[client_id])} identities")
        
        # If client has too few samples, transfer from richest client
        while len(client_data) < min_samples and len(client_assignments[client_id]) < len(all_identities):
            # Find richest client
            richest_client = None
            max_samples = 0
            for other_id in range(1, n_clients + 1):
                if other_id != client_id and len(client_assignments[other_id]) > 1:
                    other_data = df[df['identity'].isin(client_assignments[other_id])]
                    if len(other_data) > max_samples:
                        max_samples = len(other_data)
                        richest_client = other_id
            
            if richest_client is None:
                break
                
            # Transfer one identity from richest to current client
            transferred_identity = client_assignments[richest_client].pop()
            client_assignments[client_id].append(transferred_identity)
            
            # Recalculate client data
            client_data = df[df['identity'].isin(client_assignments[client_id])]
            print(f"  Transferred identity to Client {client_id}, now has {len(client_data)} samples")
    
    return client_assignments

# --- Step 1: First split into Test and Non-Test ---
print("Creating test set...")
test_identities = np.random.choice(
    df['identity'].unique(), 
    size=int(len(df['identity'].unique()) * SEPERATE_TEST),
    replace=False
)
test_df = df[df['identity'].isin(test_identities)]
non_test_df = df[~df['identity'].isin(test_identities)]

# --- Step 2: Create Balanced KD Set ---
print("\nCreating KD set...")
def create_balanced_kd_set(df, target_samples=KD_TARGET_SAMPLES, 
                          min_samples=MIN_KD_SAMPLES_PER_ID, 
                          max_samples=MAX_KD_SAMPLES_PER_ID,
                          id_ratio=KD_ID_RATIO):
    
    # If using orientation splits, create orientation-diverse KD set
    if USE_ORIENTATION_SPLITS and 'orientation' in df.columns:
        print("Creating orientation-diverse KD set...")
        return create_orientation_diverse_kd_set(df, target_samples, max_samples)
    
    # Original identity-based KD set creation
    # Select identities for KD set (prioritize those with more samples)
    id_counts = df['identity'].value_counts()
    kd_candidate_ids = id_counts[id_counts >= min_samples].index
    n_kd_ids = max(int(len(id_counts) * id_ratio), 30)  # At least 30 identities
    
    # Sort by sample count and take top identities that give us target_samples
    sorted_ids = id_counts.sort_values(ascending=False).index
    selected_ids = []
    total_samples = 0
    
    for identity in sorted_ids:
        if identity in kd_candidate_ids and len(selected_ids) < n_kd_ids:
            group = df[df['identity'] == identity]
            samples = min(max_samples, len(group))
            if total_samples + samples <= target_samples or len(selected_ids) < 30:
                selected_ids.append(identity)
                total_samples += samples
    
    # Now sample from selected identities
    kd_samples = []
    remaining_samples = []
    
    for identity in df['identity'].unique():
        group = df[df['identity'] == identity]
        if identity in selected_ids:
            n_samples = min(max_samples, max(min_samples, len(group)))
            kd_samples.append(group.sample(n=n_samples, random_state=42))
            remaining_samples.append(group.drop(kd_samples[-1].index))
        else:
            remaining_samples.append(group)
    
    kd_df = pd.concat(kd_samples)
    remaining_df = pd.concat(remaining_samples)
    
    print(f"KD set: {len(kd_df)} samples, {kd_df['identity'].nunique()} identities")
    print(f"Avg samples/identity: {len(kd_df)/kd_df['identity'].nunique():.1f}")
    print(f"Remaining: {len(remaining_df)} samples, {remaining_df['identity'].nunique()} identities")
    return kd_df, remaining_df

def create_orientation_diverse_kd_set(df, target_samples=600, max_per_identity=10):
    """
    Create KD set with maximum orientation diversity for better knowledge transfer.
    Ensures all orientations are represented to help all clients.
    """
    orientation_counts = df['orientation'].value_counts()
    print("Available orientations:", orientation_counts.to_dict())
    
    # Define target samples per orientation (prioritize rare ones)
    orientation_targets = {
        'right': int(target_samples * 0.25),      # 150 samples
        'left': int(target_samples * 0.25),       # 150 samples
        'frontright': int(target_samples * 0.08), # 48 samples
        'frontleft': int(target_samples * 0.08),  # 48 samples
        'back': int(target_samples * 0.10),       # 60 samples
        'backright': int(target_samples * 0.06),  # 36 samples
        'backleft': int(target_samples * 0.06),   # 36 samples
        'front': int(target_samples * 0.08),      # 48 samples
        'up': int(target_samples * 0.02),         # 12 samples
        'down': int(target_samples * 0.02)        # 12 samples
    }
    
    kd_samples = []
    used_identities = set()
    
    for orientation in orientation_counts.index:
        orientation_data = df[df['orientation'] == orientation]
        target = orientation_targets.get(orientation, 20)  # Default 20 for unknown orientations
        target = min(target, len(orientation_data))  # Don't exceed available data
        
        if target > 0:
            # Sample diverse identities for this orientation
            available_identities = orientation_data['identity'].unique()
            
            sampled_data = []
            samples_collected = 0
            
            # Iterate through identities to collect diverse samples
            for identity in available_identities:
                if samples_collected >= target:
                    break
                    
                identity_data = orientation_data[orientation_data['identity'] == identity]
                # Take at most max_per_identity samples from each identity
                n_samples = min(max_per_identity, len(identity_data), target - samples_collected)
                
                if n_samples > 0:
                    sampled = identity_data.sample(n=n_samples, random_state=42)
                    sampled_data.append(sampled)
                    used_identities.update(sampled['identity'].unique())
                    samples_collected += n_samples
            
            if sampled_data:
                orientation_samples = pd.concat(sampled_data)
                kd_samples.append(orientation_samples)
                print(f"KD {orientation}: {len(orientation_samples)} samples, {orientation_samples['identity'].nunique()} identities")
    
    # Combine all KD samples
    kd_df = pd.concat(kd_samples) if kd_samples else pd.DataFrame()
    
    # Create remaining dataset (exclude KD samples)
    if len(kd_df) > 0:
        remaining_df = df.drop(kd_df.index)
    else:
        remaining_df = df.copy()
    
    # Final statistics
    print(f"\nKD set summary:")
    print(f"Total samples: {len(kd_df)}")
    print(f"Unique identities: {kd_df['identity'].nunique()}")
    print(f"Orientation distribution: {kd_df['orientation'].value_counts().to_dict()}")
    print(f"Avg samples/identity: {len(kd_df)/kd_df['identity'].nunique():.1f}")
    print(f"Remaining for clients: {len(remaining_df)} samples, {remaining_df['identity'].nunique()} identities")
    
    return kd_df, remaining_df

kd_df, client_df = create_balanced_kd_set(non_test_df)

# Save KD set as "client 0"
kd_dir = f"{OUTPUT_DIR}/0"
os.makedirs(kd_dir, exist_ok=True)
save_dataframe(kd_df, f"{kd_dir}/train.csv")
save_dataframe(pd.DataFrame(), f"{kd_dir}/query.csv")
save_dataframe(pd.DataFrame(), f"{kd_dir}/gallery.csv")

# --- Step 3: Distribute Remaining to Clients ---
print("\nDistributing to clients...")

if USE_ORIENTATION_SPLITS:
    # Use orientation-based splits
    client_assignments = create_orientation_based_assignments(client_df, N_CLIENTS)
    
    if client_assignments is None:
        print("Falling back to identity-based Dirichlet distribution...")
        USE_ORIENTATION_SPLITS = False
    else:
        print("Using orientation-based client assignments.")

if not USE_ORIENTATION_SPLITS:
    # Use identity-based Dirichlet distribution
    all_identities = client_df['identity'].unique()
    np.random.seed(42)
    client_assignments = {i: [] for i in range(1, N_CLIENTS+1)}  # Clients 1-6

    # Example: 5 clients, first two are "smaller"
    probs = np.random.dirichlet(DIRICHLET_ALPHAS, size=len(all_identities))

    client_assignments = {i: [] for i in range(1, N_CLIENTS+1)}
    for idx, identity in enumerate(all_identities):
        client_id = np.random.choice(N_CLIENTS, p=probs[idx]) + 1
        client_assignments[client_id].append(identity)

    # Rebalance clients
    for client_id in client_assignments:
        client_data = client_df[client_df['identity'].isin(client_assignments[client_id])]
        while len(client_data) < MIN_SAMPLES_PER_CLIENT:
            richest_client = max(client_assignments.items(), 
                               key=lambda x: len(client_df[client_df['identity'].isin(x[1])]))[0]
            transfer_ids = client_assignments[richest_client][-1:]  # Transfer one identity
            client_assignments[richest_client] = client_assignments[richest_client][:-1]
            client_assignments[client_id].extend(transfer_ids)
            client_data = client_df[client_df['identity'].isin(client_assignments[client_id])]

# --- Step 4: Save Test Set as Client 7 ---
test_dir = f"{OUTPUT_DIR}/7"
os.makedirs(test_dir, exist_ok=True)
save_dataframe(test_df, f"{test_dir}/train.csv")

# Create test query/gallery splits
test_query, test_gallery = create_query_gallery_splits(
    test_df,
    max_queries=MAX_QUERIES_PER_ID,
    min_gallery=MIN_GALLERY_PER_QUERY,
    min_samples=MIN_GALLERY_PER_QUERY+1,
    query_ratio=0.1,
    random_state=42
)
save_dataframe(test_query, f"{test_dir}/query.csv")
save_dataframe(test_gallery, f"{test_dir}/gallery.csv")

# --- Step 5: Process Each Client ---
print("\nProcessing clients...")
used_identities = set()
for client_id, identities in client_assignments.items():
    client_dir = f"{OUTPUT_DIR}/{client_id}"
    os.makedirs(client_dir, exist_ok=True)
    identities = [id_ for id_ in identities if id_ not in used_identities]
    used_identities.update(identities)
    client_data = client_df[client_df['identity'].isin(identities)]
    
    # Split into train/test (15% test)
    id_counts = client_data['identity'].value_counts()
    multi_sample_ids = id_counts[id_counts > 1].index
    
    if len(multi_sample_ids) == 0:
        train = client_data
        test = pd.DataFrame(columns=client_data.columns)
    else:
        df_multi = client_data[client_data['identity'].isin(multi_sample_ids)]
        df_single = client_data[~client_data['identity'].isin(multi_sample_ids)]
        
        if len(df_multi) < 2:
            train = client_data
            test = pd.DataFrame(columns=client_data.columns)
        else:
            try:
                train_multi, test_multi = train_test_split(
                    df_multi, test_size=TEST_SIZE, 
                    stratify=df_multi['identity'], random_state=42
                )
            except ValueError:
                train_multi, test_multi = train_test_split(
                    df_multi, test_size=TEST_SIZE, random_state=42
                )
            
            train = pd.concat([train_multi, df_single])
            test = test_multi
    
    # Create query/gallery from test set
    if len(test) > 0 and not REMOVE_TEST_SETS:
        query, gallery = create_query_gallery_splits(
            test,
            max_queries=MAX_QUERIES_PER_ID,
            min_gallery=MIN_GALLERY_PER_QUERY,
            min_samples=MIN_GALLERY_PER_QUERY+1,
            query_ratio=0.1,
            random_state=42
        )
        query = query[query['identity'].isin(gallery['identity'])]
        test_id_counts = test['identity'].value_counts()
        problematic = test_id_counts[test_id_counts == 1].index
        if len(problematic) > 0:
            train = pd.concat([train, test[test['identity'].isin(problematic)]])
            query = query[~query['identity'].isin(problematic)]
            gallery = gallery[~gallery['identity'].isin(problematic)]
        print(f"\nVerifying identity mapping for Client {client_id}:")
        is_consistent, report = check_identity_mapping(
            query, gallery, '/home/wellvw12/hyenaid2022'
        )
    else:
        # If removing test sets, add test data back to training
        if REMOVE_TEST_SETS and len(test) > 0:
            train = pd.concat([train, test])
        query = pd.DataFrame(columns=client_data.columns)
        gallery = pd.DataFrame(columns=client_data.columns)
    save_dataframe(train, f"{client_dir}/train.csv")
    save_dataframe(query, f"{client_dir}/query.csv")
    save_dataframe(gallery, f"{client_dir}/gallery.csv")
    print(f"Client {client_id} saved - Train: {len(train)}, Query: {len(query)}, Gallery: {len(gallery)}")

# --- Verification ---
print("\nFinal Distribution Summary:")
all_identities = defaultdict(list)

for client in sorted(os.listdir(OUTPUT_DIR)):
    if not client.isdigit():
        continue
        
    client_path = f"{OUTPUT_DIR}/{client}"
    try:
        train = pd.read_csv(f"{client_path}/train.csv")
        query = pd.read_csv(f"{client_path}/query.csv") if os.path.getsize(f"{client_path}/query.csv") > 0 else pd.DataFrame()
        gallery = pd.read_csv(f"{client_path}/gallery.csv") if os.path.getsize(f"{client_path}/gallery.csv") > 0 else pd.DataFrame()
        
        identities = train['identity'].unique() if 'identity' in train.columns else []
        
        for id_ in identities:
            all_identities[id_].append(client)
    except Exception as e:
        print(f"Error processing client {client}: {str(e)}")
        continue

# Check for overlapping identities
overlaps = {k:v for k,v in all_identities.items() if len(v) > 1}
if overlaps:
    print(f"Warning: {len(overlaps)} identities shared across clients")
    print("Sample overlaps:", dict(list(overlaps.items())[:3]))
else:
    print("Success: All identities are unique to their clients")

print("\nClient Statistics:")
stats = []
for client in sorted(os.listdir(OUTPUT_DIR)):
    if not client.isdigit():
        continue
        
    client_path = f"{OUTPUT_DIR}/{client}"
    try:
        train = pd.read_csv(f"{client_path}/train.csv")
        query = pd.read_csv(f"{client_path}/query.csv") if os.path.getsize(f"{client_path}/query.csv") > 0 else pd.DataFrame()
        gallery = pd.read_csv(f"{client_path}/gallery.csv") if os.path.getsize(f"{client_path}/gallery.csv") > 0 else pd.DataFrame()
        
        client_stats = {
            'Client': client,
            'Type': 'KD' if client == '0' else 'Test' if client == '7' else 'Train',
            'Identities': train['identity'].nunique() if 'identity' in train.columns else 0,
            'Train Samples': len(train),
            'Query Samples': len(query),
            'Gallery Samples': len(gallery),
            'Avg Samples/ID': round(len(train)/train['identity'].nunique(), 1) if 'identity' in train.columns and train['identity'].nunique() > 0 else 0
        }
        
        # Add orientation info if available and orientation splits were used
        if USE_ORIENTATION_SPLITS and 'orientation' in train.columns:
            top_orientations = train['orientation'].value_counts().head(3)
            if len(top_orientations) > 0:
                orientation_str = ', '.join([f"{orient}({count})" for orient, count in top_orientations.items()])
                client_stats['Top Orientations'] = orientation_str
        
        stats.append(client_stats)
    except Exception as e:
        print(f"Error gathering stats for client {client}: {str(e)}")

print(pd.DataFrame(stats).to_markdown(index=False))
print("\nKD Set Details:")
print(kd_df['identity'].value_counts().describe())
print("\nData preparation complete!")