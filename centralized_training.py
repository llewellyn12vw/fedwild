#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone centralized training script based on train_standalone_client from main.py
Takes a single client folder containing train.csv, query.csv, and gallery.csv
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader
from wildlife_tools.data import WildlifeDataset
import torchvision.transforms as T
from utils import get_model, get_optimizer, extract_feature, set_random_seed
import scipy.io

def parse_args():
    parser = argparse.ArgumentParser(description='Centralized Training')
    parser.add_argument('--metadata_file', default='/home/wellvw12/fedwild/data/0/metadata.csv', type=str, help='Path to metadata file with split column')
    parser.add_argument('--root_dir', default='/home/wellvw12/leopard', type=str, help='Root directory for images')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--model_name', default='centralized_model', type=str, help='output model name')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--drop_rate', default=0.03, type=float, help='drop rate')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--model', default='resnet18_ft_net', type=str, help='model name: resnet18_ft_net, resnet50_arcface, dinov2_arcface, megadescriptor_ft_net')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--test_interval', default=5, type=int, help='test every x epochs (default: 10)')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
    
    return parser.parse_args()

def load_datasets_from_metadata(metadata_file, root_dir, batch_size, erasing_p, color_jitter):
    """Load train, query, and gallery datasets from metadata file with split column"""
    
    # Load metadata
    metadata = pd.read_csv(metadata_file)
    print(f"Loaded metadata with {len(metadata)} samples")
    
    # Check for required columns
    required_cols = ['split', 'identity']
    for col in required_cols:
        if col not in metadata.columns:
            raise ValueError(f"Required column '{col}' not found in metadata file")
    
    # Split data based on 'split' column
    train_metadata = metadata[metadata['split'] == 'train'].copy()
    query_metadata = metadata[metadata['split'] == 'query'].copy()
    gallery_metadata = metadata[metadata['split'] == 'gallery'].copy()
    
    print(f"Split distribution:")
    print(f"  Train: {len(train_metadata)} samples ({train_metadata['identity'].nunique()} IDs)")
    print(f"  Query: {len(query_metadata)} samples ({query_metadata['identity'].nunique()} IDs)")
    print(f"  Gallery: {len(gallery_metadata)} samples ({gallery_metadata['identity'].nunique()} IDs)")
    
    if len(train_metadata) == 0:
        raise ValueError("No training data found in metadata file")
    if len(query_metadata) == 0:
        raise ValueError("No query data found in metadata file")
    if len(gallery_metadata) == 0:
        raise ValueError("No gallery data found in metadata file")
    
    # Create transforms
    train_transform = []
    if color_jitter:
        train_transform.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
    
    train_transform.extend([
        T.Resize((224, 224), interpolation=3),
        T.Pad(10),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if erasing_p > 0:
        from random_erasing import RandomErasing
        train_transform.append(RandomErasing(probability=erasing_p, mean=[0.0, 0.0, 0.0]))
    
    eval_transform = T.Compose([
        T.Resize(size=(224, 224), interpolation=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_transform = T.Compose(train_transform)
    
    # Create datasets
    train_dataset = WildlifeDataset(
        metadata=train_metadata,
        root=root_dir,
        transform=train_transform
    )
    
    query_dataset = WildlifeDataset(
        metadata=query_metadata,
        root=root_dir,
        transform=eval_transform
    )
    
    gallery_dataset = WildlifeDataset(
        metadata=gallery_metadata,
        root=root_dir,
        transform=eval_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Store metadata for evaluation
    gallery_meta = {
        'sizes': len(gallery_dataset),
        'labels': gallery_dataset.labels
    }
    
    query_meta = {
        'sizes': len(query_dataset),
        'labels': query_dataset.labels
    }
    
    return {
        'train_loader': train_loader,
        'query_loader': query_loader,
        'gallery_loader': gallery_loader,
        'num_classes': train_dataset.num_classes,
        'dataset_size': len(train_dataset),
        'query_meta': query_meta,
        'gallery_meta': gallery_meta
    }

def test_model(model, data_info, device, save_dir, epoch=None):
    """Test model and save results"""
    
    print(f"Testing model...")
    
    model.eval()
    model = model.to(device)
    
    # Temporarily remove classifier for feature extraction
    original_classifier = model.classifier.classifier
    model.classifier.classifier = nn.Sequential()
    
    with torch.no_grad():
        gallery_feature = extract_feature(model, data_info['gallery_loader'], [1.0])
        query_feature = extract_feature(model, data_info['query_loader'], [1.0])

    # Restore classifier
    model.classifier.classifier = original_classifier

    result = {
        'gallery_f': gallery_feature.cpu().numpy(),
        'gallery_label': data_info['gallery_meta']['labels'],
        'query_f': query_feature.cpu().numpy(),
        'query_label': data_info['query_meta']['labels'],
    }
    
    # Save .mat file
    mat_path = os.path.join(save_dir, 'pytorch_result.mat')
    scipy.io.savemat(mat_path, result)
    
    # Evaluate and save CSV results
    epoch_suffix = f"_epoch_{epoch}" if epoch is not None else ""
    output_file = f'centralized_result{epoch_suffix}.csv'
    cmd = (
        f"python evaluate.py --result_dir {save_dir} "
        f"--dataset centralized{epoch_suffix} --output_file {output_file}"
    )
    os.system(cmd)
    
    print(f"Test results saved to {output_file}")
    return True

def train_centralized_model():
    args = parse_args()
    print("=====Starting Centralized Training=====")
    print(f"Metadata file: {args.metadata_file}")
    print(f"Root directory: {args.root_dir}")
    
    # Validate metadata file exists
    if not os.path.exists(args.metadata_file):
        print(f"Error: Metadata file {args.metadata_file} not found")
        sys.exit(1)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    set_random_seed(1)
    
    # Load datasets from metadata file
    print("Loading datasets from metadata file...")
    data_info = load_datasets_from_metadata(
        args.metadata_file,
        args.root_dir,
        args.batch_size,
        args.erasing_p,
        args.color_jitter
    )
    
    # Get number of classes and dataset info
    num_classes = data_info['num_classes']
    dataset_size = data_info['dataset_size']
    print(f"Number of classes: {num_classes}")
    print(f"Training dataset size: {dataset_size}")
    
    # Create model
    model = get_model(num_classes, args.drop_rate, args.stride, args.model)
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = get_optimizer(model, args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Create save directory
    save_dir = os.path.join('results', 'centralized', args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    best_acc = 0.0
    train_loader = data_info['train_loader']
    
    print(f"Training centralized model for {args.epochs} epochs...")
    print(f"Testing at epoch 0 and then every {args.test_interval} epochs")
    
    # Test at epoch 0 (before training)
    print(f"\n--- Testing at epoch 0 (initial) ---")
    test_result = test_model(model, data_info, device, save_dir, 0)
    if test_result:
        print(f"Initial test completed")
    print("--- Starting training ---\n")
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        print('-' * 10)
        
        model.train(True)
        running_loss = 0.0
        running_corrects = 0.0
        
        for batch_data in train_loader:
            inputs, labels = batch_data
            b = inputs.shape[0]
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
        used_data_size = (dataset_size - dataset_size % args.batch_size)
        epoch_loss = running_loss / used_data_size
        epoch_acc = running_corrects / used_data_size
        
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'New best accuracy: {best_acc:.4f} - Model saved')

        # Test at specified intervals
        if (epoch + 1) % args.test_interval == 0:
            print(f"\n--- Testing at epoch {epoch + 1} ---")
            test_result = test_model(model, data_info, device, save_dir, epoch + 1)
            if test_result:
                print(f"Test completed at epoch {epoch + 1}")
            print("--- Resuming training ---\n")
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # Final test
    print(f"\n--- Final Testing ---")
    test_model(model, data_info, device, save_dir)
    
    print(f'Centralized training completed. Best accuracy: {best_acc:.4f}')
    print(f'Results saved to: {save_dir}')

if __name__ == '__main__':
    train_centralized_model()