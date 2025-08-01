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
from experiment_config import save_experiment_config, save_client_info, save_metadata_info

mp.set_start_method('spawn', force=True)
sys.setrecursionlimit(10000)
version =  torch.__version__

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--model_name',default='federated_model', type=str, help='output model name')
parser.add_argument('--ex_name',default='3LeopardSame', type=str, help='output result name')
parser.add_argument('--project_dir',default='.', type=str, help='project path')
parser.add_argument('--data_dir',default='/home/wellvw12/fedwild/MacaqueFaces',type=str, help='training dir path')
# parser.add_argument('--datasets',default='Market,DukeMTMC-reID,cuhk03-np-detected,cuhk01,MSMT17,viper,prid,3dpes,ilids',type=str, help='datasets used')
parser.add_argument('--datasets',default='beskydy,nps,sumava',type=str, help='datasets used')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--dataset_type', default='czechlynx', type=str, choices=['leopard', 'macaque', 'hyena', 'cow','czechlynx'], help='dataset type to use')
parser.add_argument('--metadata_file', default="/home/wellvw12/fedwild/czechlynx_federated", type=str, help='path to unified metadata.csv file with client allocation')
parser.add_argument('--image_size', default=128, type=int, help='input image size for training and testing')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--drop_rate', default=0.03, type=float, help='drop rate')
parser.add_argument('--model', default='resnet18_ft_net', type=str, help='model name')

# arguments for federated setting
parser.add_argument('--local_epoch', default=1, type=int, help='number of local epochs')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_of_clients', default=2, type=int, help='number of clients')

# arguments for data transformation
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--use_original_transform_only', action='store_true', help='use only original transform "0" for testing, default false' )


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
parser.add_argument('--fedgkd_start_round', default=5, type=int, help='round to start applying FedGKD distillation loss (buffer still fills from round 0)')

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

    data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, args.train_all, args.dataset_type, args.metadata_file, args.image_size,args.use_original_transform_only)
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
        args.use_original_transform_only,
    )
    
    # Configure FedGKD if enabled
    if args.fedgkd:
        server.configure_fedgkd(
            buffer_length=args.fedgkd_buffer_length,
            distillation_coeff=args.fedgkd_distillation_coeff,
            temperature=args.fedgkd_temperature,
            avg_param=args.fedgkd_avg_param,
            start_round=args.fedgkd_start_round
        )

    dir_name = os.path.join(args.project_dir, 'model', args.ex_name)
    os.makedirs(dir_name, exist_ok=True)  # Creates parent dirs if needed

    # Save experiment configuration and metadata
    print("Saving experiment configuration...")
    save_experiment_config(args, dir_name)
    save_client_info(data, dir_name)
    save_metadata_info(args.metadata_file, dir_name)
    print("Experiment configuration saved successfully!")

    print("=====training start!========")
    print(f"FedGKD enabled: {args.fedgkd}")
    if args.fedgkd:
        print(f"FedGKD settings - Buffer: {args.fedgkd_buffer_length}, Coeff: {args.fedgkd_distillation_coeff}, Temp: {args.fedgkd_temperature}, Avg: {args.fedgkd_avg_param}, Start Round: {args.fedgkd_start_round}")
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
        gallery_feature = extract_feature(model, test_loaders['gallery'], [1.0], data.image_size)
        query_feature = extract_feature(model, test_loaders['query'], [1.0], data.image_size)

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
    
    data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, args.train_all, args.dataset_type, args.metadata_file, args.image_size)
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





if __name__ == '__main__':
    train_fd()
    # standalone_training()


