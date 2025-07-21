# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import time
import os
import yaml
import random
import numpy as np
import scipy.io
import pathlib
import sys
import json
import copy
import multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
from client import Client
from server import Server
from utils import set_random_seed
from data_utils import Data

mp.set_start_method('spawn', force=True)
sys.setrecursionlimit(10000)
version =  torch.__version__

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--model_name',default='federated_model', type=str, help='output model name')
parser.add_argument('--ex_name',default='3LeopardSame', type=str, help='output result name')
parser.add_argument('--project_dir',default='.', type=str, help='project path')
parser.add_argument('--data_dir',default='/home/wellvw12/baselines/baseline3.3.2',type=str, help='training dir path')
# parser.add_argument('--datasets',default='Market,DukeMTMC-reID,cuhk03-np-detected,cuhk01,MSMT17,viper,prid,3dpes,ilids',type=str, help='datasets used')
parser.add_argument('--datasets',default='0,1',type=str, help='datasets used')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--multi_species', action='store_true', help='use multiple species')
parser.add_argument('--metadata', default=None, type=str, help='metadata type: leopard, hyena, sea_turtle, cow')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--drop_rate', default=0.03, type=float, help='drop rate')
parser.add_argument('--model', default='resnet18_ft_net', type=str, help='model name')

# arguments for federated setting
parser.add_argument('--local_epoch', default=1, type=int, help='number of local epochs')
parser.add_argument('--batch_size', default=30, type=int, help='batch size')
parser.add_argument('--num_of_clients', default=3, type=int, help='number of clients')

# arguments for data transformation
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )

# arguments for testing federated model
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--multiple_scale',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--test_dir',default='all',type=str, help='./test_data')

# arguments for optimization
parser.add_argument('--cdw', action='store_true', help='use cosine distance weight for model aggregation, default false' )
parser.add_argument('--kd', action='store_true', help='apply knowledge distillation, default false' )
parser.add_argument('--regularization', action='store_true', help='use regularization during distillation, default false' )

# arguments for FedGKD
parser.add_argument('--fedgkd', action='store_true', help='enable FedGKD (Global Knowledge Distillation), default false')
parser.add_argument('--fedgkd_buffer_length', default=3, type=int, help='number of historical models to keep in FedGKD buffer')
parser.add_argument('--fedgkd_distillation_coeff', default=0.1, type=float, help='coefficient for FedGKD distillation loss')
parser.add_argument('--fedgkd_temperature', default=2.0, type=float, help='temperature for FedGKD distillation')
parser.add_argument('--fedgkd_avg_param', action='store_true', help='use FedGKD with parameter averaging (default), if false uses FedGKD-VOTE')

# arguments for cosine annealing learning rate scheduling
parser.add_argument('--cosine_annealing', default=False, help='use cosine annealing learning rate scheduling, default false' )
parser.add_argument('--total_rounds', default=50, type=int, help='total number of federated rounds for cosine annealing')
parser.add_argument('--eta_min', default=1e-5, type=float, help='minimum learning rate for cosine annealing')
parser.add_argument('--kd_lr_ratio', default=0.1, type=float, help='knowledge distillation learning rate as ratio of client LR')


def train_fd():
    args = parser.parse_args()
    print(args)
    print('regs',args.regularization)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    set_random_seed(1)

    data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, args.train_all,args.multi_species, args.metadata)
    data.preprocess()
    
    clients = {}
    for cid in data.client_list:
        clients[cid] = Client(
            cid, 
            data, 
            device, 
            args.project_dir, 
            args.model_name, 
            args.local_epoch, 
            args.lr, 
            args.batch_size, 
            args.drop_rate, 
            args.stride,
            args.ex_name,
            args.model,
            args.cosine_annealing,
            args.total_rounds,
            args.eta_min,
            )

    server = Server(
        clients, 
        data, 
        device, 
        args.project_dir, 
        args.model_name, 
        args.num_of_clients, 
        args.lr, 
        args.drop_rate, 
        args.stride, 
        args.multiple_scale,
        args.ex_name,
        args.model,
        args.kd_lr_ratio,
    )
    
    # Configure FedGKD if enabled
    if args.fedgkd:
        server.configure_fedgkd(
            buffer_length=args.fedgkd_buffer_length,
            distillation_coeff=args.fedgkd_distillation_coeff,
            temperature=args.fedgkd_temperature,
            avg_param=args.fedgkd_avg_param
        )

    dir_name = os.path.join(args.project_dir, 'model', args.ex_name)
    os.makedirs(dir_name, exist_ok=True)  # Creates parent dirs if needed

    print("=====training start!========")
    print(f"FedGKD enabled: {args.fedgkd}")
    if args.fedgkd:
        print(f"FedGKD settings - Buffer: {args.fedgkd_buffer_length}, Coeff: {args.fedgkd_distillation_coeff}, Temp: {args.fedgkd_temperature}, Avg: {args.fedgkd_avg_param}")
    print(f"Knowledge Distillation enabled: {args.kd}")
    
    if args.fedgkd and args.kd:
        print("FedGKD and Knowledge Distillation are both enabled")
    
    rounds = args.total_rounds
    for i in range(rounds):
        print('='*10)
        print("Round Number {}".format(i))
        print('='*10)
        if i==0:
            server.test(use_cuda)
        server.train(i, args.cdw, use_cuda,i)
        save_path = os.path.join(dir_name, 'federated_model.pth')
        torch.save(server.federated_model.cpu().state_dict(), save_path)
        if (i+1)%10 == 0:
            if args.kd:
                server.knowledge_distillation(args.regularization, round=i+1)
            server.test(use_cuda)
        server.draw_curve()


def test_standalone_client(client, model, device, save_dir):
    """Test standalone model and save results"""
    import scipy.io
    from utils import extract_feature
    
    # Check if test data exists for this client
    if client.cid not in client.data.test_loaders:
        print(f"No test data for client {client.cid}")
        return False
    
    test_loaders = client.data.test_loaders[client.cid]
    if 'query' not in test_loaders or 'gallery' not in test_loaders:
        print(f"No query/gallery split for client {client.cid}")
        return False
    
    print(f"Testing standalone model for client {client.cid}...")
    
    model.eval()
    model = model.to(device)
    
    # Temporarily remove classifier for feature extraction (like in federated setup)
    original_classifier = model.classifier.classifier
    model.classifier.classifier = nn.Sequential()  # Remove classifier for feature extraction
    
    
    with torch.no_grad():
        gallery_feature = extract_feature(model, test_loaders['gallery'], [1.0])
        query_feature = extract_feature(model, test_loaders['query'], [1.0])

    model.classifier.classifier = original_classifier

    result = {
        'gallery_f': gallery_feature.cpu().numpy(),
        'gallery_label': client.data.gallery_meta[client.cid]['labels'],
        'query_f': query_feature.cpu().numpy(),
        'query_label': client.data.query_meta[client.cid]['labels'],
    }
    
    # Save .mat file
    mat_path = os.path.join(save_dir, 'pytorch_result.mat')
    scipy.io.savemat(mat_path, result)
    
    # Evaluate and save CSV results
    output_file = os.path.join(save_dir, 'standalone_result.csv')
    cmd = (
        f"python evaluate.py --result_dir {save_dir} "
        f"--dataset client_{client.cid} --output_file standalone_result.csv"
    )
    os.system(cmd)
    
    print(f"Standalone test results for client {client.cid} saved to {output_file}")
    return True


def train_standalone_client(client, model, device, args, save_dir):
    """Train a single client's model independently"""
    import torch.nn as nn
    from torch.optim import lr_scheduler
    from utils import get_optimizer
    import time

    
    model.train()
    
    optimizer = get_optimizer(model, args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 30  # More epochs for standalone training
    best_acc = 0.0
    
    print(f"Training standalone model for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)
        
        model.train(True)
        running_loss = 0.0
        running_corrects = 0.0
        
        for data in client.train_loader:
            inputs, labels = data
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
        
        used_data_sizes = (client.dataset_sizes - client.dataset_sizes % args.batch_size)
        epoch_loss = running_loss / used_data_sizes
        epoch_acc = running_corrects / used_data_sizes
        
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_path = os.path.join(save_dir, 'best_standalone_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'New best accuracy: {best_acc:.4f} - Model saved')

        # Test every 10th epoch
        if (epoch + 1) % 10 == 0:
            print(f"\n--- Testing at epoch {epoch + 1} ---")
            test_result = test_standalone_client(client, model, device, save_dir)
            if test_result:
                print(f"Test completed at epoch {epoch + 1}")
            print("--- Resuming training ---\n")
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'final_standalone_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final standalone training completed. Best accuracy: {best_acc:.4f}')


def standalone_training():
    """Train each client independently on their own dataset"""
    args = parser.parse_args()
    print("=====Starting Standalone Training=====")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    set_random_seed(1)
    
    data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, args.train_all)
    data.preprocess()
    
    # Train each client independently
    for cid in data.client_list:
        print(f"\n{'='*20}")
        print(f"Standalone training for Client {cid}")
        print(f"{'='*20}")
        if cid in ['1', '2', 'test']:
            print(f"Skipping client {cid} as it is not a standalone client.")
            continue
        # Create a fresh client instance to get the proper classifier
        client = Client(
            cid, 
            data, 
            device, 
            args.project_dir, 
            f'standalone_model_{cid}', 
            args.local_epoch, 
            args.lr, 
            args.batch_size, 
            args.drop_rate, 
            args.stride,
            experiment_name=args.ex_name
        )
        
        # Create standalone model directory
        standalone_dir = os.path.join(args.project_dir, 'model', args.ex_name, f'client_{cid}', 'standalone')
        os.makedirs(standalone_dir, exist_ok=True)
        
        # Use the client's model which already has proper setup
        # Create a copy of the full model with classifier
        from utils import get_model
        standalone_model = get_model(data.train_class_sizes[cid], args.drop_rate, args.stride, args.model)
        # Set the classifier from the client
        # standalone_model.classifier.classifier = client.classifier
        standalone_model = standalone_model.to(device)
        
        # Train standalone model
        train_standalone_client(client, standalone_model, device, args, standalone_dir)


def centralized_training():
    """Train a single model on all clients' data combined"""
    args = parser.parse_args()
    print("=====Starting Centralized Training=====")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    set_random_seed(1)
    
    data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, args.train_all)
    data.preprocess()
    
    # Create centralized model directory
    centralized_dir = os.path.join(args.project_dir, 'model', args.ex_name, 'centralized')
    os.makedirs(centralized_dir, exist_ok=True)
    
    # Calculate total number of classes across all clients
    total_classes = sum(data.train_class_sizes.values())
    print(f"Total classes across all clients: {total_classes}")
    
    # Create centralized model
    from utils import get_model
    centralized_model = get_model(total_classes, args.drop_rate, args.stride, args.model)
    centralized_model = centralized_model.to(device)
    
    # Create combined data loader
    combined_loader = create_combined_dataloader(data, args.batch_size)
    combined_dataset_size = sum(data.train_dataset_sizes.values())
    
    print(f"Combined dataset size: {combined_dataset_size}")
    
    # Train centralized model
    train_centralized_model(centralized_model, combined_loader, combined_dataset_size, 
                          data, device, args, centralized_dir)


def create_combined_dataloader(data, batch_size):
    """Create a combined dataloader from all clients' training data"""
    import torch.utils.data as torch_data
    from torch.utils.data import ConcatDataset
    
    # Collect all datasets from clients
    all_datasets = []
    label_offset = 0
    
    for cid in data.client_list:
        client_loader = data.train_loaders[cid]
        
        # Get the dataset from the loader
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


def train_centralized_model(model, combined_loader, combined_dataset_size, data, device, args, save_dir):
    """Train the centralized model on combined data"""
    import torch.nn as nn
    from torch.optim import lr_scheduler
    from utils import get_optimizer
    import time
    
    model.train()
    
    optimizer = get_optimizer(model, args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100  # Same as federated training rounds
    best_acc = 0.0
    
    print(f"Training centralized model for {epochs} epochs...")
    
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
        epoch_loss = running_loss / used_data_sizes
        epoch_acc = running_corrects / used_data_sizes
        
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_path = os.path.join(save_dir, 'best_centralized_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'New best accuracy: {best_acc:.4f} - Model saved')

        # Test every 10th epoch on each client's test data
        if (epoch + 1) % 10 == 0:
            print(f"\n--- Testing centralized model at epoch {epoch + 1} ---")
            test_centralized_on_all_clients(model, data, device, save_dir, epoch + 1)
            print("--- Resuming training ---\n")
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'final_centralized_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Centralized training completed. Best accuracy: {best_acc:.4f}')


def test_centralized_on_all_clients(model, data, device, save_dir, epoch):
    """Test centralized model on each client's test data using existing evaluate.py"""
    import scipy.io
    from utils import extract_feature
    
    model.eval()
    model = model.to(device)
    
    # Temporarily remove classifier for feature extraction
    original_classifier = model.classifier.classifier
    model.classifier.classifier = nn.Sequential()
    
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
                gallery_feature = extract_feature(model, test_loaders['gallery'], [1.0])
                query_feature = extract_feature(model, test_loaders['query'], [1.0])
            
            # Prepare result for evaluate.py
            result = {
                'gallery_f': gallery_feature.cpu().numpy(),
                'gallery_label': data.gallery_meta[cid]['labels'],
                'query_f': query_feature.cpu().numpy(),
                'query_label': data.query_meta[cid]['labels'],
            }
            
            # Create client-specific directory for this epoch
            client_epoch_dir = os.path.join(save_dir, f'epoch_{epoch}', f'client_{cid}')
            os.makedirs(client_epoch_dir, exist_ok=True)
            
            # Save .mat file for evaluate.py
            mat_path = os.path.join(client_epoch_dir, 'pytorch_result.mat')
            scipy.io.savemat(mat_path, result)
            
            # Use existing evaluate.py to calculate metrics
            output_file = f'centralized_epoch_{epoch}_client_{cid}.csv'
            cmd = (
                f"python evaluate.py --result_dir {client_epoch_dir} "
                f"--dataset client_{cid}_epoch_{epoch} --output_file {output_file}"
            )
            os.system(cmd)
            
            print(f"Client {cid} centralized test results for epoch {epoch} saved")
        
        # Create summary CSV file combining all client results for this epoch
        create_centralized_summary(save_dir, epoch, data.client_list)
        
    finally:
        # Restore classifier
        model.classifier.classifier = original_classifier


def create_centralized_summary(save_dir, epoch, client_list):
    """Create a summary CSV file combining all client results for this epoch"""
    import pandas as pd
    from datetime import datetime
    
    summary_data = []
    
    for cid in client_list:
        client_epoch_dir = os.path.join(save_dir, f'epoch_{epoch}', f'client_{cid}')
        csv_file = os.path.join(client_epoch_dir, f'centralized_epoch_{epoch}_client_{cid}.csv')
        
        if os.path.exists(csv_file):
            try:
                # Read the individual client results
                df = pd.read_csv(csv_file)
                if not df.empty:
                    # Get the latest (last) entry
                    latest_result = df.iloc[-1]
                    summary_data.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'epoch': epoch,
                        'client_id': cid,
                        'rank1': latest_result['rank1'],
                        'rank5': latest_result['rank5'] if 'rank5' in latest_result else 'NA',
                        'mAP': latest_result['mAP']
                    })
            except Exception as e:
                print(f"Error reading results for client {cid}: {e}")
    
    # Save summary to main centralized directory
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(save_dir, 'centralized_summary.csv')
        
        # Append to existing file or create new one
        if os.path.exists(summary_csv):
            summary_df.to_csv(summary_csv, mode='a', header=False, index=False)
        else:
            summary_df.to_csv(summary_csv, index=False)
        
        print(f"Centralized summary for epoch {epoch} saved to centralized_summary.csv")
        
        # Print epoch summary
        print(f"\nCentralized Model Performance Summary - Epoch {epoch}:")
        print("-" * 60)
        for _, row in summary_df.iterrows():
            print(f"Client {row['client_id']}: Rank-1: {row['rank1']:.4f}, mAP: {row['mAP']:.4f}")
        print("-" * 60)

if __name__ == '__main__':
    train_fd()
    # standalone_training()
    # centralized_training()


