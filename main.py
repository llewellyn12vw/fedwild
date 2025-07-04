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
parser.add_argument('--ex_name',default='experimentX', type=str, help='output result name')
parser.add_argument('--project_dir',default='.', type=str, help='project path')
parser.add_argument('--data_dir',default='/home/wellvw12/full_leopard_4/clients',type=str, help='training dir path')
# parser.add_argument('--datasets',default='Market,DukeMTMC-reID,cuhk03-np-detected,cuhk01,MSMT17,viper,prid,3dpes,ilids',type=str, help='datasets used')
parser.add_argument('--datasets',default='1,2,3,4,5,test',type=str, help='datasets used')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--drop_rate', default=0.5, type=float, help='drop rate')

# arguments for federated setting
parser.add_argument('--local_epoch', default=1, type=int, help='number of local epochs')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_of_clients', default=5, type=int, help='number of clients')

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


def train_fd():
    args = parser.parse_args()
    print(args)
    print('regs',args.regularization)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    set_random_seed(1)

    data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, args.train_all)
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
            experiment_name=args.ex_name)

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
        experiment_name=args.ex_name)

    dir_name = os.path.join(args.project_dir, 'model', args.ex_name)
    os.makedirs(dir_name, exist_ok=True)  # Creates parent dirs if needed

    print("=====training start!========")
    
    rounds = 100
    for i in range(rounds):
        print('='*10)
        print("Round Number {}".format(i))
        print('='*10)
        server.train(i, args.cdw, use_cuda,i)
        save_path = os.path.join(dir_name, 'federated_model.pth')
        torch.save(server.federated_model.cpu().state_dict(), save_path)
        if i==0:
            server.test(use_cuda)
        if (i+1)%10 == 0:
            if args.kd:
                server.knowledge_distillation(args.regularization)
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
        standalone_model = get_model(data.train_class_sizes[cid], args.drop_rate, args.stride)
        # Set the classifier from the client
        # standalone_model.classifier.classifier = client.classifier
        standalone_model = standalone_model.to(device)
        
        # Train standalone model
        train_standalone_client(client, standalone_model, device, args, standalone_dir)

if __name__ == '__main__':
    # train_fd()
    standalone_training()



