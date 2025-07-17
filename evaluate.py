import scipy.io
import torch
import numpy as np
import argparse
import os
from datetime import datetime
from tabulate import tabulate
import pandas as pd

# Argument parsing
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--result_dir', default='.', type=str, help='Directory containing results')
parser.add_argument('--dataset', default='no_dataset', type=str, help='Dataset being evaluated')
parser.add_argument('--output_file', default='aggregated_result.csv', help='File to save aggregated results')
parser.add_argument('--loss', type=float, default=0.0, help='Loss value for the evaluation')
parser.add_argument('--enable_species_eval', action='store_true', help='Enable species-specific evaluation')
parser.add_argument('--species_a', default='leopard', type=str, help='First species to evaluate')
parser.add_argument('--species_b', default='hyena', type=str, help='Second species to evaluate')
args = parser.parse_args()

def get_species_labels():
    """Get species labels for query and gallery"""
    species_result_dir = os.path.join("/home/wellvw12/fedReID/lep_hyn_exact", args.dataset)
    query_csv = os.path.join(species_result_dir, 'query.csv')
    gallery_csv = os.path.join(species_result_dir, 'gallery.csv')
    
    query_species = None
    gallery_species = None
    
    if os.path.exists(query_csv):
        df = pd.read_csv(query_csv)
        if 'species' in df.columns:
            query_species = df['species'].values
    
    if os.path.exists(gallery_csv):
        df = pd.read_csv(gallery_csv)
        if 'species' in df.columns:
            gallery_species = df['species'].values
    print(species_result_dir)
    print(f"Query species: {query_species}, Gallery species: {gallery_species}")
    return query_species, gallery_species

def evaluate_by_species(qf, ql, gf, gl, query_species, gallery_species, target_species):
    """Evaluate only queries and galleries of a specific species"""
    # Filter query indices for target species
    query_indices = np.where(np.atleast_1d(query_species) == target_species)[0]
    
    # Filter indices that are within bounds of the feature arrays
    query_indices = query_indices[query_indices < len(ql)]
    
    if len(query_indices) == 0:
        return 0.0, torch.IntTensor([0]).zero_(), 0
    
    # Filter gallery indices for target species
    gallery_indices = np.where(np.atleast_1d(gallery_species) == target_species)[0]
    
    # Filter indices that are within bounds of the feature arrays
    gallery_indices = gallery_indices[gallery_indices < len(gl)]
    
    if len(gallery_indices) == 0:
        return 0.0, torch.IntTensor([0]).zero_(), 0
    
    # Extract features and labels for target species
    species_qf = qf[query_indices]
    species_ql = ql[query_indices]
    species_gf = gf[gallery_indices]
    species_gl = gl[gallery_indices]
    
    # Run evaluation
    CMC = torch.IntTensor(len(species_gl)).zero_()
    ap = 0.0
    valid_queries = 0
    
    for i in range(len(species_ql)):
        ap_tmp, CMC_tmp = evaluate(species_qf[i], species_ql[i], species_gf, species_gl)
        if CMC_tmp[0] == -1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp
        valid_queries += 1
    
    if valid_queries > 0:
        CMC = CMC.float() / valid_queries
        ap = ap / valid_queries
    
    return ap, CMC, valid_queries

def save_results(dataset, metrics):
    """Save results to CSV file, appending new entries"""
    output_file = os.path.join(args.result_dir, args.output_file)
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, 'a') as f:
        if not file_exists:
            header = "timestamp,dataset,rank1,mAP,loss\n"
            f.write(header)
        
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{dataset},"
        line += f"{metrics['rank1']:.4f},"
        # line += f"{metrics['rank5']:.4f}," if not np.isnan(metrics['rank5']) else "NA,"
        line += f"{metrics['mAP']:.4f},"
        if(args.loss is not None):
            line += f"{args.loss:.4f}\n"
        else:
            line += "NA\n"
        
        f.write(line)

def save_species_results(dataset, species_metrics):
    """Save species-specific results to separate CSV files"""
    for species, metrics in species_metrics.items():
        output_file = os.path.join(args.result_dir, f'{species}_evaluation_results.csv')
        file_exists = os.path.isfile(output_file)
        
        with open(output_file, 'a') as f:
            if not file_exists:
                header = "timestamp,dataset,rank1,mAP,valid_queries,loss\n"
                f.write(header)
            
            line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{dataset},"
            line += f"{metrics['rank1']:.4f},"
            line += f"{metrics['mAP']:.4f},"
            line += f"{metrics['valid_queries']},"
            if(args.loss is not None):
                line += f"{args.loss:.4f}\n"
            else:
                line += "NA\n"
            
            f.write(line)

def print_results_table(output_file):
    """Print all results in a formatted table"""
    if not os.path.exists(output_file):
        print("No results file found")
        return
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) <= 1:
        print("No results recorded yet")
        return
    
    headers = lines[0].strip().split(',')[1:]  # Skip timestamp
    data = []
    for line in lines[1:]:
        parts = line.strip().split(',')
        data.append([p if p != 'NA' else 'N/A' for p in parts[1:]])  # Skip timestamp
    
    print("\n=== Evaluation Results Summary ===")
    print(tabulate(data, headers=headers, floatfmt=".4f", tablefmt="grid"))

def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    
    index = np.argsort(score)[::-1]
    good_index = np.argwhere(gl == ql).flatten()
    junk_index = np.argwhere(gl == -1).flatten()
    
    ap, cmc = compute_mAP(index, good_index, junk_index)
    return ap, cmc

def compute_mAP(index, good_index, junk_index):
    ap = 0.0
    cmc = torch.IntTensor(len(index)).zero_()
    
    if len(good_index) == 0:
        cmc[0] = -1
        return ap, cmc

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask).flatten()
    
    if len(rows_good) > 0:
        cmc[rows_good[0]:] = 1
        
        for i in range(ngood):
            d_recall = 1.0 / ngood
            precision = (i + 1) / (rows_good[i] + 1)
            if rows_good[i] != 0:
                old_precision = i / rows_good[i]
            else:
                old_precision = 1.0
            ap += d_recall * (old_precision + precision) / 2

    return ap, cmc

# Main evaluation
result = scipy.io.loadmat(os.path.join(args.result_dir, 'pytorch_result.mat'))
query_feature = torch.FloatTensor(result['query_f']).cuda()
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f']).cuda()
gallery_label = result['gallery_label'][0]

# Get species labels if enabled
query_species = None
gallery_species = None

args.enable_species_eval = True
if args.enable_species_eval:
    query_species, gallery_species = get_species_labels()
    if query_species is not None and gallery_species is not None:
        print(f"Species-specific evaluation enabled for: {args.species_a}, {args.species_b}")
    else:
        print("No species data found, running standard evaluation only")
        args.enable_species_eval = False

# Standard evaluation (overall)
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
valid_queries = 0

for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
    if CMC_tmp[0] == -1:
        continue
    CMC += CMC_tmp
    ap += ap_tmp
    valid_queries += 1

# Calculate overall metrics
CMC = CMC.float() / valid_queries
metrics = {
    'rank1': CMC[0].item(),
    # 'rank5': CMC[4].item() if len(gallery_label) >= 5 else float('nan'),
    'mAP': ap / valid_queries
}

# Save overall results
save_results(args.dataset, metrics)

# Print overall results
print(f"\nOverall Evaluation ({args.dataset}):")
print(f"Rank-1:  {metrics['rank1']:.4f}")
print(f"mAP:     {metrics['mAP']:.4f}")
print('-' * 15)

# Species-specific evaluation
if args.enable_species_eval and query_species is not None and gallery_species is not None:
    print(f"\nSpecies-specific Evaluation:")
    print('=' * 40)
    
    species_metrics = {}
    species_to_evaluate = [args.species_a, args.species_b]
    
    for species in species_to_evaluate:
        print(f"\nEvaluating species: {species}")
        
        # Check if species exists in the data
        if species not in query_species and species not in gallery_species:
            print(f"  Species '{species}' not found in data, skipping...")
            continue
        
        # Run species-specific evaluation
        species_ap, species_CMC, species_valid = evaluate_by_species(
            query_feature, query_label, gallery_feature, gallery_label,
            query_species, gallery_species, species
        )
        
        if species_valid > 0:
            species_metrics[species] = {
                'rank1': species_CMC[0].item(),
                'mAP': species_ap,
                'valid_queries': species_valid
            }
            
            print(f"  Rank-1:  {species_metrics[species]['rank1']:.4f}")
            print(f"  mAP:     {species_metrics[species]['mAP']:.4f}")
            print(f"  Valid queries: {species_valid}")
        else:
            print(f"  No valid queries found for species: {species}")
    
    # Save species-specific results (only for species that exist)
    if species_metrics:
        save_species_results(args.dataset, species_metrics)
        saved_files = [f"{species}_evaluation_results.csv" for species in species_metrics.keys()]
        print(f"\nSpecies evaluation results saved to: {', '.join(saved_files)}")
    else:
        print(f"\nNo species data to save.")
    print('=' * 40)

print('-' * 15)