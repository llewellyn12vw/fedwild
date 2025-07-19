#!/usr/bin/env python3
"""
Centralized Testing Framework for FedReID
Combines model architecture, training, evaluation, and data loading components
for centralized testing of selected datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import os
import sys
import argparse
import time
import copy
import numpy as np
import pandas as pd
from datetime import datetime
import scipy.io
from torchvision import transforms
from torch.utils.data import DataLoader

# Import components from existing modules
from model import ft_net, megadescriptor
from utils import get_optimizer, get_model, extract_feature, set_random_seed
from data_utils import Data
from optimization import Optimization
from wildlife_tools.data.dataset import WildlifeDataset
from wildlife_datasets.datasets import LeopardID2022, HyenaID2022

class CentralizedTester:
    def __init__(self, 
                 dataset_name='LeopardID2022',
                 data_dir='/home/wellvw12/client_data_non_iid',
                 model_type='resnet18_ft_net',
                 batch_size=32,
                 learning_rate=0.001,
                 num_epochs=10,
                 device='cuda',
                 experiment_name='centralized_test',
                 random_seed=42,
                 enable_species_eval=True,
                 species_a_value='leopard',
                 species_b_value='hyena'):
        
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        self.experiment_name = experiment_name
        self.enable_species_eval = enable_species_eval
        self.species_a_value = species_a_value
        self.species_b_value = species_b_value
        
        # Set random seed for reproducibility
        set_random_seed(random_seed)
        
        # Initialize data loader
        self.data_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Results storage
        self.results = []
        
        print(f"Initialized CentralizedTester for {dataset_name}")
        print(f"Model: {model_type}, Device: {device}")
        
    def load_dataset(self, client_ids=None):
        """Load specified datasets/clients for centralized testing"""
        if client_ids is None:
            # Auto-detect available clients
            client_ids = []
            for i in range(10):  # Check for clients 0-9
                client_dir = os.path.join(self.data_dir, str(i))
                if os.path.exists(client_dir):
                    client_ids.append(str(i))
        
        # Convert to comma-separated string for Data class
        datasets_str = ','.join(client_ids)
        
        print(f"Loading clients: {client_ids}")
        
        # Initialize data loader using existing Data class
        self.data_loader = Data(
            datasets=datasets_str,
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            erasing_p=0.0,  # No random erasing for centralized testing
            color_jitter=False,
            train_all=False,
            multi_s=True,
            metadata='/home/wellvw12/lep_hyn'
        )
        
        # Preprocess data
        self.data_loader.preprocess()
        
        # Calculate total number of classes across all clients
        all_labels = set()
        for client_id in client_ids:
            if client_id in self.data_loader.train_loaders:
                dataset = self.data_loader.train_loaders[client_id].dataset
                all_labels.update(dataset.labels)
        self.total_classes = len(all_labels)
        print(f"Total unique classes across all clients: {self.total_classes}")
        
        return client_ids
    
    def detect_species(self, client_id):
        """Detect species from client query and gallery CSV files"""
        client_dir = os.path.join(self.data_dir, client_id)
        query_csv = os.path.join(client_dir, 'query.csv')
        gallery_csv = os.path.join(client_dir, 'gallery.csv')
        
        species_found = set()
        
        # Check query CSV
        if os.path.exists(query_csv):
            df = pd.read_csv(query_csv)
            if 'species' in df.columns:
                species_found.update(df['species'].unique())
        
        # Check gallery CSV
        if os.path.exists(gallery_csv):
            df = pd.read_csv(gallery_csv)
            if 'species' in df.columns:
                species_found.update(df['species'].unique())
        
        return list(species_found)
    
    def get_species_labels(self, client_id):
        """Get species labels for query and gallery"""
        client_dir = os.path.join(self.data_dir, client_id)
        query_csv = os.path.join(client_dir, 'query.csv')
        gallery_csv = os.path.join(client_dir, 'gallery.csv')
        
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
        
        return query_species, gallery_species
    
    def initialize_model(self):
        """Initialize model architecture"""
        print(f"Initializing {self.model_type} model with {self.total_classes} classes")
        
        # Create model using existing get_model function
        self.model = get_model(
            class_sizes=self.total_classes,
            drop_rate=0.5,
            stride=2,
            model_type=self.model_type
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = get_optimizer(self.model, self.learning_rate)
        
        # Initialize scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def train_centralized(self, client_ids):
        """Train model in centralized fashion using data from selected clients"""
        print(f"\nStarting centralized training for {self.num_epochs} epochs")
        
        # Combine data from all selected clients
        combined_loader = self._combine_client_data(client_ids)
        
        # Training loop
        train_losses = []
        train_accuracies = []
        
        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.num_epochs}')
            print('-' * 40)
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            for batch_idx, (inputs, labels) in enumerate(combined_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
                if batch_idx % 10 == 0:
                    print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Calculate epoch metrics
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc.item())
            
            print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
            
            # Update scheduler
            self.scheduler.step()
        
        print(f"\nTraining completed!")
        return train_losses, train_accuracies
    
    def _combine_client_data(self, client_ids):
        """Combine training data from multiple clients"""
        combined_datasets = []
        
        for client_id in client_ids:
            if client_id in self.data_loader.train_loaders:
                # Get the dataset from the loader
                dataset = self.data_loader.train_loaders[client_id].dataset
                combined_datasets.append(dataset)
        
        # Combine datasets
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset(combined_datasets)
        
        # Create combined loader
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True  # Drop incomplete batches to avoid batch norm issues
        )
        
        print(f"Combined dataset size: {len(combined_dataset)} samples")
        return combined_loader
    
    def evaluate_clients(self, client_ids):
        """Evaluate trained model on individual clients"""
        print(f"\nEvaluating model on {len(client_ids)} clients")
        
        self.model.eval()
        client_results = []
        
        for client_id in client_ids:
            if client_id not in self.data_loader.test_loaders:
                print(f"No test data for client {client_id}")
                continue
            
            print(f"\nEvaluating client {client_id}...")
            
            # Extract features
            with torch.no_grad():
                gallery_features = extract_feature(
                    self.model, 
                    self.data_loader.test_loaders[client_id]['gallery'], 
                    [1.0]
                )
                query_features = extract_feature(
                    self.model, 
                    self.data_loader.test_loaders[client_id]['query'], 
                    [1.0]
                )
            
            # Prepare evaluation data
            result = {
                'gallery_f': gallery_features.cpu().numpy(),
                'gallery_label': self.data_loader.gallery_meta[client_id]['labels'],
                'query_f': query_features.cpu().numpy(),
                'query_label': self.data_loader.query_meta[client_id]['labels'],
            }
            
            # Save results for evaluation
            result_dir = os.path.join('results', self.experiment_name, f'client_{client_id}')
            os.makedirs(result_dir, exist_ok=True)
            
            mat_path = os.path.join(result_dir, 'pytorch_result.mat')
            scipy.io.savemat(mat_path, result)
            
            # Run overall evaluation
            metrics = self._evaluate_client_performance(result)
            
            client_result = {
                'client_id': client_id,
                'rank1': metrics['rank1'],
                'mAP': metrics['mAP'],
                'num_queries': len(query_features),
                'num_gallery': len(gallery_features)
            }
            
            print(f"Client {client_id} - Overall Rank-1: {metrics['rank1']:.4f}, mAP: {metrics['mAP']:.4f}")
            
            # Species-specific evaluation
            if self.enable_species_eval:
                species_found = self.detect_species(client_id)
                if len(species_found) > 1:
                    print(f"  Multi-species detected: {species_found}")
                    query_species, gallery_species = self.get_species_labels(client_id)
                    
                    species_metrics = {}
                    for species in species_found:
                        species_perf = self._evaluate_species_performance(
                            result, query_species, gallery_species, species
                        )
                        species_metrics[species] = species_perf
                        
                        print(f"  {species} - Rank-1: {species_perf['rank1']:.4f}, mAP: {species_perf['mAP']:.4f}, Queries: {species_perf['valid_queries']}")
                    
                    client_result['species_metrics'] = species_metrics
                    
                    # Save species-specific results to client folder
                    species_results = []
                    for species, perf in species_metrics.items():
                        species_results.append({
                            'species': species,
                            'rank1': perf['rank1'],
                            'mAP': perf['mAP'],
                            'valid_queries': perf['valid_queries']
                        })
                    
                    species_df = pd.DataFrame(species_results)
                    species_df.to_csv(os.path.join(result_dir, 'species_evaluation.csv'), index=False)
                else:
                    print(f"  Single species detected: {species_found}")
                    client_result['species_metrics'] = None
            
            client_results.append(client_result)
        
        return client_results
    
    def _evaluate_client_performance(self, result):
        """Evaluate performance using the existing evaluation logic"""
        query_feature = torch.FloatTensor(result['query_f']).cuda()
        query_label = result['query_label']
        gallery_feature = torch.FloatTensor(result['gallery_f']).cuda()
        gallery_label = result['gallery_label']
        
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        valid_queries = 0
        
        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = self._evaluate_single_query(
                query_feature[i], query_label[i], gallery_feature, gallery_label
            )
            if CMC_tmp[0] == -1:
                continue
            CMC += CMC_tmp
            ap += ap_tmp
            valid_queries += 1
        
        CMC = CMC.float() / valid_queries
        metrics = {
            'rank1': CMC[0].item(),
            'mAP': ap / valid_queries
        }
        
        return metrics
    
    def _evaluate_single_query(self, qf, ql, gf, gl):
        """Evaluate single query using existing evaluation logic"""
        query = qf.view(-1, 1)
        score = torch.mm(gf, query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        
        index = np.argsort(score)[::-1]
        good_index = np.argwhere(gl == ql).flatten()
        junk_index = np.argwhere(gl == -1).flatten()
        
        ap, cmc = self._compute_mAP(index, good_index, junk_index)
        return ap, cmc
    
    def _compute_mAP(self, index, good_index, junk_index):
        """Compute mAP using existing logic"""
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
    
    def _evaluate_species_performance(self, result, query_species, gallery_species, target_species):
        """Evaluate performance for a specific species"""
        query_feature = torch.FloatTensor(result['query_f']).cuda()
        query_label = result['query_label']
        gallery_feature = torch.FloatTensor(result['gallery_f']).cuda()
        gallery_label = result['gallery_label']
        
        # Filter query indices for target species
        query_indices = np.where(query_species == target_species)[0]
        
        if len(query_indices) == 0:
            return {'rank1': 0.0, 'mAP': 0.0, 'valid_queries': 0}
        
        # Filter gallery indices for target species
        gallery_indices = np.where(gallery_species == target_species)[0]
        
        if len(gallery_indices) == 0:
            return {'rank1': 0.0, 'mAP': 0.0, 'valid_queries': 0}
        
        # Extract features and labels for target species
        species_qf = query_feature[query_indices]
        species_ql = query_label[query_indices]
        species_gf = gallery_feature[gallery_indices]
        species_gl = gallery_label[gallery_indices]
        
        # Run evaluation
        CMC = torch.IntTensor(len(species_gl)).zero_()
        ap = 0.0
        valid_queries = 0
        
        for i in range(len(species_ql)):
            ap_tmp, CMC_tmp = self._evaluate_single_query(
                species_qf[i], species_ql[i], species_gf, species_gl
            )
            if CMC_tmp[0] == -1:
                continue
            CMC += CMC_tmp
            ap += ap_tmp
            valid_queries += 1
        
        if valid_queries > 0:
            CMC = CMC.float() / valid_queries
            ap = ap / valid_queries
        
        return {
            'rank1': CMC[0].item() if valid_queries > 0 else 0.0,
            'mAP': ap,
            'valid_queries': valid_queries
        }
    
    def save_results(self, client_results, train_losses, train_accuracies):
        """Save comprehensive results"""
        results_dir = os.path.join('results', self.experiment_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save training history
        training_history = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'model_type': self.model_type
        }
        
        train_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'train_accuracy': train_accuracies
        })
        train_df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
        
        # Save client evaluation results
        client_df = pd.DataFrame(client_results)
        client_df.to_csv(os.path.join(results_dir, 'client_evaluation.csv'), index=False)
        
        # Save species-specific summary if enabled
        if self.enable_species_eval:
            species_summary = []
            for result in client_results:
                if result.get('species_metrics'):
                    for species, metrics in result['species_metrics'].items():
                        species_summary.append({
                            'client_id': result['client_id'],
                            'species': species,
                            'rank1': metrics['rank1'],
                            'mAP': metrics['mAP'],
                            'valid_queries': metrics['valid_queries']
                        })
            
            if species_summary:
                species_df = pd.DataFrame(species_summary)
                species_df.to_csv(os.path.join(results_dir, 'species_summary.csv'), index=False)
                
                # Calculate average metrics per species
                species_avg = species_df.groupby('species').agg({
                    'rank1': 'mean',
                    'mAP': 'mean',
                    'valid_queries': 'sum'
                }).round(4)
                
                print(f"\nSpecies-specific Average Performance:")
                print(species_avg)
                
                species_avg.to_csv(os.path.join(results_dir, 'species_average.csv'))
        
        # Save summary
        summary = {
            'experiment_name': self.experiment_name,
            'dataset_name': self.dataset_name,
            'model_type': self.model_type,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'total_classes': self.total_classes,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'average_rank1': np.mean([r['rank1'] for r in client_results]),
            'average_mAP': np.mean([r['mAP'] for r in client_results]),
            'final_train_loss': train_losses[-1],
            'final_train_accuracy': train_accuracies[-1]
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(results_dir, 'experiment_summary.csv'), index=False)
        
        print(f"\nResults saved to {results_dir}/")
        print(f"Average Rank-1: {summary['average_rank1']:.4f}")
        print(f"Average mAP: {summary['average_mAP']:.4f}")
        
    def run_experiment(self, client_ids=None):
        """Run complete centralized testing experiment"""
        print(f"Starting centralized testing experiment: {self.experiment_name}")
        
        # Load datasets
        client_ids = self.load_dataset(client_ids)
        
        # Initialize model
        self.initialize_model()
        
        # Train centralized model
        train_losses, train_accuracies = self.train_centralized(client_ids)
        
        # Evaluate on individual clients
        client_results = self.evaluate_clients(client_ids)
        
        # Save results
        self.save_results(client_results, train_losses, train_accuracies)
        
        return client_results, train_losses, train_accuracies

def main():
    parser = argparse.ArgumentParser(description='Centralized Testing for FedReID')
    parser.add_argument('--dataset', default='LeopardID2022', help='Dataset name')
    parser.add_argument('--data_dir', default="/home/wellvw12/baselines/baseline3.3.2", help='Data directory')
    parser.add_argument('--model_type', default='resnet18_ft_net', 
                       choices=['resnet18_ft_net', 'megadescriptor'], help='Model type')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--experiment_name', default='centralized_test', help='Experiment name')
    parser.add_argument('--client_ids', nargs='+', default=None, 
                       help='Client IDs to use (e.g., 0 1 2)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--enable_species_eval', action='store_true', default=True,
                       help='Enable species-specific evaluation')
    parser.add_argument('--species_a_value', default='leopard', type=str, help='Species A value')
    parser.add_argument('--species_b_value', default='hyena', type=str, help='Species B value')
    
    args = parser.parse_args()
    
    # Convert client_ids to strings if provided
    client_ids = [str(cid) for cid in args.client_ids] if args.client_ids else None
    
    # Initialize tester
    tester = CentralizedTester(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        experiment_name=args.experiment_name,
        random_seed=args.random_seed,
        enable_species_eval=args.enable_species_eval,
        species_a_value=args.species_a_value,
        species_b_value=args.species_b_value
    )
    
    # Run experiment
    client_results, train_losses, train_accuracies = tester.run_experiment(client_ids)
    
    print("\nExperiment completed successfully!")

if __name__ == '__main__':
    main()