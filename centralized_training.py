# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import time
import os
import numpy as np
import scipy.io
import pathlib
import sys
import json
import copy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset
import torch.utils.data as torch_data
import pandas as pd
from datetime import datetime

from utils import set_random_seed, get_optimizer, get_model, extract_feature
from data_utils import Data

parser = argparse.ArgumentParser(description='Centralized Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--model_name',default='centralized_model', type=str, help='output model name')
parser.add_argument('--ex_name',default='centralized_experiment', type=str, help='output result name')
parser.add_argument('--project_dir',default='.', type=str, help='project path')
parser.add_argument('--data_dir',default='/home/wellvw12/fedwild/MacaqueFaces',type=str, help='training dir path')
parser.add_argument('--datasets',default='beskydy,nps,sumava',type=str, help='datasets used')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--dataset_type', default='czechlynx', type=str, choices=['leopard', 'macaque', 'hyena', 'cow','czechlynx'], help='dataset type to use')
parser.add_argument('--metadata_file', default="/home/wellvw12/fedwild/czechlynx_federated", type=str, help='path to unified metadata.csv file with client allocation')
parser.add_argument('--image_size', default=128, type=int, help='input image size for training and testing')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--drop_rate', default=0.03, type=float, help='drop rate')
parser.add_argument('--model', default='resnet18_ft_net', type=str, help='model name')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--use_original_transform_only', action='store_true', help='use only original transform "0" for testing, default false' )
parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--test_interval', default=5, type=int, help='test every x epochs')


class AdjustedLabelDataset:
    """Wrapper dataset to adjust labels by adding an offset"""
    def __init__(self, dataset, label_offset):
        self.dataset = dataset
        self.label_offset = label_offset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        adjusted_label = label + self.label_offset
        return data, adjusted_label


def create_combined_dataloader(data, batch_size):
    """Create a combined dataloader from all clients' training data"""
    all_datasets = []
    label_offset = 0
    
    for cid in data.client_list:
        client_loader = data.train_loaders[cid]
        client_dataset = client_loader.dataset
        
        # Create a wrapper to adjust labels for global uniqueness
        adjusted_dataset = AdjustedLabelDataset(client_dataset, label_offset)
        all_datasets.append(adjusted_dataset)
        
        # Update offset for next client
        label_offset += data.train_class_sizes[cid]
    
    # Combine all datasets
    combined_dataset = ConcatDataset(all_datasets)
    
    # Create combined dataloader
    combined_loader = torch_data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    return combined_loader


def evaluate_single_query(qf, ql, gf, gl):
    """Evaluate a single query against gallery"""
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
    """Compute mean Average Precision and Cumulative Matching Curve"""
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


def evaluate_features(query_feature, query_labels, gallery_feature, gallery_labels):
    """Evaluate query features against gallery features"""
    query_feature = torch.FloatTensor(query_feature).cuda()
    gallery_feature = torch.FloatTensor(gallery_feature).cuda()
    
    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    ap = 0.0
    valid_queries = 0

    for i in range(len(query_labels)):
        ap_tmp, CMC_tmp = evaluate_single_query(query_feature[i], query_labels[i], gallery_feature, gallery_labels)
        if CMC_tmp[0] == -1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp
        valid_queries += 1

    # Calculate overall metrics
    CMC = CMC.float() / valid_queries if valid_queries > 0 else CMC.float()
    rank1 = CMC[0].item() if valid_queries > 0 else 0.0
    mAP = ap / valid_queries if valid_queries > 0 else 0.0
    
    return rank1, mAP


def test_centralized_on_all_clients(model, data, device, save_dir, epoch):
    """Test centralized model on each client's test data and save to summary CSV"""
    model.eval()
    model = model.to(device)
    
    # Temporarily remove classifier for feature extraction
    original_classifier = model.classifier.classifier
    model.classifier.classifier = nn.Sequential()
    
    summary_data = []
    
    try:
        for cid in data.client_list:
            if cid not in data.test_loaders:
                print(f"No test data for client {cid}")
                continue
            
            test_loaders = data.test_loaders[cid]
            if 'query' not in test_loaders or 'gallery' not in test_loaders:
                print(f"No query/gallery split for client {cid}")
                continue
            
            print(f"Testing centralized model on client {cid} data...")
            
            with torch.no_grad():
                gallery_feature = extract_feature(model, test_loaders['gallery'], [1.0], data.image_size)
                query_feature = extract_feature(model, test_loaders['query'], [1.0], data.image_size)
            
            # Calculate metrics using our evaluation functions
            gallery_labels = np.array(data.gallery_meta[cid]['labels'])
            query_labels = np.array(data.query_meta[cid]['labels'])
            
            rank1, mAP = evaluate_features(query_feature.cpu().numpy(), query_labels, 
                                         gallery_feature.cpu().numpy(), gallery_labels)
            
            summary_data.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'epoch': epoch,
                'client_id': cid,
                'rank1': rank1,
                'mAP': mAP
            })
            
            print(f"Client {cid}: Rank-1: {rank1:.4f}, mAP: {mAP:.4f}")
        
        # Save results directly to centralized_summary.csv
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_csv = os.path.join(save_dir, 'centralized_summary.csv')
            
            # Append to existing file or create new one
            if os.path.exists(summary_csv):
                summary_df.to_csv(summary_csv, mode='a', header=False, index=False)
            else:
                summary_df.to_csv(summary_csv, index=False)
            
            print(f"Results for epoch {epoch} saved to centralized_summary.csv")
        
    finally:
        # Restore classifier
        model.classifier.classifier = original_classifier




def train_centralized_model(model, combined_loader, combined_dataset_size, data, device, args, save_dir):
    """Train the centralized model on combined data"""
    model.train()
    
    optimizer = get_optimizer(model, args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    epochs = args.epochs
    test_interval = args.test_interval
    best_acc = 0.0
    
    print(f"Training centralized model for {epochs} epochs, testing every {test_interval} epochs...")
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)
        
        model.train(True)
        running_loss = 0.0
        running_corrects = 0.0
        
        for batch_data in combined_loader:
            inputs, labels = batch_data
            b, c, h, w = inputs.shape
            if b < args.batch_size:
                continue
                
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * b
            running_corrects += float(torch.sum(preds == labels.data))
        
        scheduler.step()
        
        # Calculate epoch metrics
        used_data_sizes = (combined_dataset_size - combined_dataset_size % args.batch_size)
        epoch_loss = running_loss / used_data_sizes if used_data_sizes > 0 else 0
        epoch_acc = running_corrects / used_data_sizes if used_data_sizes > 0 else 0
        
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_path = os.path.join(save_dir, 'best_centralized_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'New best accuracy: {best_acc:.4f} - Model saved')

        # Test every test_interval epochs
        if (epoch + 1) % test_interval == 0 or epoch == 0:
            print(f"\n--- Testing centralized model at epoch {epoch + 1} ---")
            test_centralized_on_all_clients(model, data, device, save_dir, epoch + 1)
            print("--- Resuming training ---\n")
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'final_centralized_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Centralized training completed. Best accuracy: {best_acc:.4f}')


def centralized_training():
    """Train a single model on all clients' data combined"""
    args = parser.parse_args()
    print("=====Starting Centralized Training=====")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    set_random_seed(1)
    
    # Initialize data using data_utils
    data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, 
                args.train_all, args.dataset_type, args.metadata_file, args.image_size, 
                args.use_original_transform_only)
    data.preprocess()
    
    # Create centralized model directory
    centralized_dir = os.path.join(args.project_dir, 'model', args.ex_name, 'centralized')
    os.makedirs(centralized_dir, exist_ok=True)
    
    # Calculate total number of classes across all clients
    total_classes = sum(data.train_class_sizes.values())
    print(f"Total classes across all clients: {total_classes}")
    
    # Create centralized model
    centralized_model = get_model(total_classes, args.drop_rate, args.stride, args.model)
    centralized_model = centralized_model.to(device)
    
    # Create combined data loader using data_utils
    combined_loader = create_combined_dataloader(data, args.batch_size)
    combined_dataset_size = sum(data.train_dataset_sizes.values())
    
    print(f"Combined dataset size: {combined_dataset_size}")
    print(f"Number of clients: {len(data.client_list)}")
    print(f"Client list: {data.client_list}")
    
    # Train centralized model
    train_centralized_model(centralized_model, combined_loader, combined_dataset_size, 
                          data, device, args, centralized_dir)


if __name__ == '__main__':
    centralized_training()