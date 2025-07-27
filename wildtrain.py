from itertools import chain
import torch
import timm
import pandas as pd
import torchvision.transforms as T
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm

from wildlife_tools.data import WildlifeDataset
from wildlife_tools.train import ArcFaceLoss, BasicTrainer
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='Wildlife Training with Evaluation')
    parser.add_argument('--metadata_file', default='/home/wellvw12/fedwild/data/0/metadata.csv', type=str, help='Path to metadata file with split column')
    parser.add_argument('--root_dir', default='/home/wellvw12/leopard', type=str, help='Image root directory')
    parser.add_argument('--model', default="swin_tiny_patch4_window7_224", type=str, help='Model name')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--eval_interval', default=10, type=int, help='Evaluation interval')
    parser.add_argument('--device', default='cuda', type=str, help='Device')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers')
    parser.add_argument('--accumulation_steps', default=2, type=int, help='Gradient accumulation steps')
    return parser.parse_args()

def compute_mAP(distances, query_labels, gallery_labels):
    """Compute mean Average Precision"""
    sorted_indices = np.argsort(distances, axis=1)
    num_queries = distances.shape[0]
    AP_sum = 0.0
    valid_queries = 0
    
    for i in range(num_queries):
        query_label = query_labels[i]
        sorted_gallery_labels = gallery_labels[sorted_indices[i]]
        
        # Find matching gallery images
        matches = (sorted_gallery_labels == query_label)
        if np.sum(matches) == 0:
            continue
            
        # Compute AP for this query
        cum_matches = np.cumsum(matches)
        precision = cum_matches / np.arange(1, len(matches) + 1)
        AP = np.sum(precision * matches) / np.sum(matches)
        AP_sum += AP
        valid_queries += 1
    
    return AP_sum / valid_queries if valid_queries > 0 else 0.0

def compute_cmc(distances, query_labels, gallery_labels, max_rank=10):
    """Compute Cumulative Matching Characteristics (CMC)"""
    sorted_indices = np.argsort(distances, axis=1)
    num_queries = distances.shape[0]
    
    cmc = np.zeros(max_rank)
    valid_queries = 0
    
    for i in range(num_queries):
        query_label = query_labels[i]
        sorted_gallery_labels = gallery_labels[sorted_indices[i]]
        
        # Find first match
        match_indices = np.where(sorted_gallery_labels == query_label)[0]
        if len(match_indices) == 0:
            continue
            
        first_match = match_indices[0]
        valid_queries += 1
        
        # Update CMC
        for k in range(min(first_match + 1, max_rank)):
            cmc[k] += 1
    
    return cmc / valid_queries if valid_queries > 0 else cmc

def evaluate_model(backbone, objective, query_dataset, gallery_dataset, device='cuda'):
    """Evaluate model using Top-1 accuracy and mAP"""
    backbone.eval()
    objective.eval()
    
    # Extract query features
    query_loader = DataLoader(query_dataset, batch_size=64, shuffle=False, num_workers=2)
    query_features = []
    query_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(query_loader, desc="Extracting query features"):
            images = images.to(device)
            features = backbone(images)
            features = F.normalize(features, p=2, dim=1)
            query_features.append(features.cpu())
            query_labels.extend(labels.numpy())
    
    query_features = torch.cat(query_features, dim=0).numpy()
    query_labels = np.array(query_labels)
    
    # Extract gallery features
    gallery_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False, num_workers=2)
    gallery_features = []
    gallery_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(gallery_loader, desc="Extracting gallery features"):
            images = images.to(device)
            features = backbone(images)
            features = F.normalize(features, p=2, dim=1)
            gallery_features.append(features.cpu())
            gallery_labels.extend(labels.numpy())
    
    gallery_features = torch.cat(gallery_features, dim=0).numpy()
    gallery_labels = np.array(gallery_labels)
    
    # Compute cosine distances (1 - cosine similarity)
    similarities = np.dot(query_features, gallery_features.T)
    distances = 1 - similarities
    
    # Compute metrics
    mAP = compute_mAP(distances, query_labels, gallery_labels)
    cmc = compute_cmc(distances, query_labels, gallery_labels, max_rank=10)
    top1_acc = cmc[0]
    
    return {
        'mAP': mAP,
        'top1': top1_acc,
        'top5': cmc[4] if len(cmc) > 4 else 0.0,
        'cmc': cmc
    }

def custom_training_loop(args):
    """Custom training loop with evaluation at round 0 and every x rounds"""
    
    # Dataset configuration
    transform = T.Compose([
        T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        T.RandAugment(num_ops=2, magnitude=20),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    eval_transform = T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # Load unified metadata and filter by split
    print(f"Loading metadata from: {args.metadata_file}")
    metadata = pd.read_csv(args.metadata_file)
    print(f"Total metadata samples: {len(metadata)}")
    
    # Filter datasets by split column
    train_metadata = metadata[metadata['split'] == 'train'].copy()
    query_metadata = metadata[metadata['split'] == 'query'].copy()
    gallery_metadata = metadata[metadata['split'] == 'gallery'].copy()
    
    print(f"Split distribution:")
    print(f"  Train: {len(train_metadata)} samples ({train_metadata['identity'].nunique()} IDs)")
    print(f"  Query: {len(query_metadata)} samples ({query_metadata['identity'].nunique()} IDs)")
    print(f"  Gallery: {len(gallery_metadata)} samples ({gallery_metadata['identity'].nunique()} IDs)")
    
    # Create datasets
    train_dataset = WildlifeDataset(
        metadata=train_metadata,
        root=args.root_dir,
        transform=transform
    )
    
    query_dataset = WildlifeDataset(
        metadata=query_metadata,
        root=args.root_dir,
        transform=eval_transform
    )
    
    gallery_dataset = WildlifeDataset(
        metadata=gallery_metadata,
        root=args.root_dir,
        transform=eval_transform
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Query samples: {len(query_dataset)}")
    print(f"Gallery samples: {len(gallery_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")
    
    # Model configuration
    backbone = timm.create_model(args.model, num_classes=0, pretrained=True)
    backbone = backbone.to(args.device)
    
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(args.device)
        embedding_size = backbone(dummy_input).shape[1]
    
    objective = ArcFaceLoss(
        num_classes=train_dataset.num_classes,
        embedding_size=embedding_size,
        margin=0.5,
        scale=64
    ).to(args.device)
    
    # Optimizer and scheduler
    params = chain(backbone.parameters(), objective.parameters())
    optimizer = SGD(params=params, lr=args.lr, momentum=0.9)
    min_lr = args.lr * 1e-3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=min_lr)
    
    # Data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Evaluation at epoch 0 and every {args.eval_interval} epochs")
    
    # Evaluate at round 0 (before training)
    print(f"\n{'='*50}")
    print(f"Evaluation at epoch 0 (initial)")
    print(f"{'='*50}")
    
    metrics = evaluate_model(backbone, objective, query_dataset, gallery_dataset, args.device)
    print(f"Initial Results:")
    print(f"  Top-1 Accuracy: {metrics['top1']:.4f}")
    print(f"  mAP: {metrics['mAP']:.4f}")
    print(f"  Top-5 Accuracy: {metrics['top5']:.4f}")
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\n{'='*30}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*30}")
        
        backbone.train()
        objective.train()
        
        running_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            # Forward pass
            features = backbone(images)
            loss = objective(features, labels)
            
            # Backward pass
            loss.backward()
            
            if (batch_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/num_batches:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Update scheduler
        scheduler.step()
        
        avg_loss = running_loss / num_batches
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
        
        # Evaluate at specified intervals
        if (epoch + 1) % args.eval_interval == 0:
            print(f"\n{'='*50}")
            print(f"Evaluation at epoch {epoch+1}")
            print(f"{'='*50}")
            
            metrics = evaluate_model(backbone, objective, query_dataset, gallery_dataset, args.device)
            print(f"Results after epoch {epoch+1}:")
            print(f"  Top-1 Accuracy: {metrics['top1']:.4f}")
            print(f"  mAP: {metrics['mAP']:.4f}")
            print(f"  Top-5 Accuracy: {metrics['top5']:.4f}")
    
    # Final evaluation
    print(f"\n{'='*50}")
    print(f"Final Evaluation")
    print(f"{'='*50}")
    
    final_metrics = evaluate_model(backbone, objective, query_dataset, gallery_dataset, args.device)
    print(f"Final Results:")
    print(f"  Top-1 Accuracy: {final_metrics['top1']:.4f}")
    print(f"  mAP: {final_metrics['mAP']:.4f}")
    print(f"  Top-5 Accuracy: {final_metrics['top5']:.4f}")
    
    return backbone, objective, final_metrics

if __name__ == "__main__":
    args = parse_args()
    backbone, objective, metrics = custom_training_loop(args)
