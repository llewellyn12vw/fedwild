import scipy.io
import torch
import numpy as np
import argparse
import os
from datetime import datetime
from tabulate import tabulate

# Argument parsing
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--result_dir', default='.', type=str, help='Directory containing results')
parser.add_argument('--dataset', default='no_dataset', type=str, help='Dataset being evaluated')
parser.add_argument('--output_file', default='aggregated_result.csv', help='File to save aggregated results')
args = parser.parse_args()

def save_results(dataset, metrics):
    """Save results to CSV file, appending new entries"""
    output_file = os.path.join(args.result_dir, args.output_file)
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, 'a') as f:
        if not file_exists:
            header = "timestamp,dataset,rank1,rank5,mAP\n"
            f.write(header)
        
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{dataset},"
        line += f"{metrics['rank1']:.4f},"
        # line += f"{metrics['rank5']:.4f}," if not np.isnan(metrics['rank5']) else "NA,"
        line += f"{metrics['mAP']:.4f}\n"
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

# Calculate metrics
CMC = CMC.float() / valid_queries
metrics = {
    'rank1': CMC[0].item(),
    # 'rank5': CMC[4].item() if len(gallery_label) >= 5 else float('nan'),
    'mAP': ap / valid_queries
}

# Save and display results
save_results(args.dataset, metrics)
# print_results_table(args.output_file)

# Print current results
print(f"\nCurrent Evaluation ({args.dataset}):")
print(f"Rank-1:  {metrics['rank1']:.4f}")
# if not np.isnan(metrics['rank5']):
#     print(f"Rank-5:  {metrics['rank5']:.4f}")
print(f"mAP:     {metrics['mAP']:.4f}")
print('-' * 15)