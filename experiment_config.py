"""
Experiment Configuration Saving Module

This module handles saving experiment configuration details and metadata
to the experiment directory for reproducibility and tracking.
"""

import os
import json
import datetime
import platform
import torch
import pandas as pd
from typing import Dict, Any

def save_experiment_config(args, experiment_dir: str, additional_info: Dict[str, Any] = None):
    """
    Save experiment configuration and metadata to the experiment directory.
    
    Args:
        args: Parsed command-line arguments
        experiment_dir: Directory where experiment files are stored
        additional_info: Optional dictionary with additional information to save
    """
    
    # Create experiment directory if it doesn't exist
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Prepare experiment configuration
    config = {
        "experiment_info": {
            "name": args.ex_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "total_rounds": args.total_rounds,
            "project_dir": args.project_dir,
            "data_dir": args.data_dir
        },
        "dataset_config": {
            "dataset_type": args.dataset_type,
            "datasets": args.datasets,
            "metadata_file": args.metadata_file,
            "train_all": args.train_all
        },
        "model_config": {
            "model": args.model,
            "model_name": args.model_name,
            "drop_rate": args.drop_rate,
            "stride": args.stride
        },
        "training_config": {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "local_epoch": args.local_epoch,
            "num_of_clients": args.num_of_clients,
            "cosine_annealing": args.cosine_annealing,
            "eta_min": args.eta_min
        },
        "data_augmentation": {
            "erasing_p": args.erasing_p,
            "color_jitter": args.color_jitter
        },
        "optimization": {
            "cdw": args.cdw,
            "kd": args.kd,
            "regularization": args.regularization,
            "kd_lr_ratio": args.kd_lr_ratio
        },
        "fedgkd_config": {
            "enabled": args.fedgkd,
            "buffer_length": args.fedgkd_buffer_length,
            "distillation_coeff": args.fedgkd_distillation_coeff,
            "temperature": args.fedgkd_temperature,
            "avg_param": args.fedgkd_avg_param,
            "start_round": args.fedgkd_start_round
        },
        "testing_config": {
            "which_epoch": args.which_epoch,
            "multi": args.multi,
            "multiple_scale": args.multiple_scale,
            "test_dir": args.test_dir
        },
        "system_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }
    
    # Add additional information if provided
    if additional_info:
        config["additional_info"] = additional_info
    
    # Save configuration as JSON
    config_path = os.path.join(experiment_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"Experiment configuration saved to: {config_path}")
    
    # Also save a human-readable summary
    save_experiment_summary(config, experiment_dir)
    
    return config_path

def save_experiment_summary(config: Dict[str, Any], experiment_dir: str):
    """
    Save a human-readable experiment summary.
    
    Args:
        config: Configuration dictionary
        experiment_dir: Directory where experiment files are stored
    """
    
    summary_path = os.path.join(experiment_dir, 'experiment_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEDERATED LEARNING EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Experiment Info
        exp_info = config["experiment_info"]
        f.write("EXPERIMENT INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Name: {exp_info['name']}\n")
        f.write(f"Timestamp: {exp_info['timestamp']}\n")
        f.write(f"Total Rounds: {exp_info['total_rounds']}\n\n")
        
        # Dataset Config
        dataset_info = config["dataset_config"]
        f.write("DATASET CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset Type: {dataset_info['dataset_type']}\n")
        f.write(f"Client Datasets: {dataset_info['datasets']}\n")
        f.write(f"Metadata File: {dataset_info['metadata_file']}\n")
        f.write(f"Use All Training Data: {dataset_info['train_all']}\n\n")
        
        # Model Config
        model_info = config["model_config"]
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Architecture: {model_info['model']}\n")
        f.write(f"Model Name: {model_info['model_name']}\n")
        f.write(f"Drop Rate: {model_info['drop_rate']}\n")
        f.write(f"Stride: {model_info['stride']}\n\n")
        
        # Training Config
        train_info = config["training_config"]
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Learning Rate: {train_info['learning_rate']}\n")
        f.write(f"Batch Size: {train_info['batch_size']}\n")
        f.write(f"Local Epochs: {train_info['local_epoch']}\n")
        f.write(f"Number of Clients: {train_info['num_of_clients']}\n")
        f.write(f"Cosine Annealing: {train_info['cosine_annealing']}\n")
        if train_info['cosine_annealing']:
            f.write(f"Eta Min: {train_info['eta_min']}\n")
        f.write("\n")
        
        # Optimization
        opt_info = config["optimization"]
        f.write("OPTIMIZATION TECHNIQUES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Cosine Distance Weighting (CDW): {opt_info['cdw']}\n")
        f.write(f"Knowledge Distillation (KD): {opt_info['kd']}\n")
        if opt_info['kd']:
            f.write(f"  - Regularization: {opt_info['regularization']}\n")
            f.write(f"  - KD LR Ratio: {opt_info['kd_lr_ratio']}\n")
        f.write("\n")
        
        # FedGKD Config
        fedgkd_info = config["fedgkd_config"]
        f.write("FEDGKD CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"FedGKD Enabled: {fedgkd_info['enabled']}\n")
        if fedgkd_info['enabled']:
            f.write(f"  - Buffer Length: {fedgkd_info['buffer_length']}\n")
            f.write(f"  - Distillation Coefficient: {fedgkd_info['distillation_coeff']}\n")
            f.write(f"  - Temperature: {fedgkd_info['temperature']}\n")
            f.write(f"  - Average Parameters: {fedgkd_info['avg_param']}\n")
            f.write(f"  - Start Round: {fedgkd_info['start_round']}\n")
            strategy = "FedGKD (simple averaging)" if fedgkd_info['avg_param'] else "FedGKD-VOTE (performance weighting)"
            f.write(f"  - Strategy: {strategy}\n")
        f.write("\n")
        
        # Data Augmentation
        aug_info = config["data_augmentation"]
        f.write("DATA AUGMENTATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Random Erasing Probability: {aug_info['erasing_p']}\n")
        f.write(f"Color Jitter: {aug_info['color_jitter']}\n\n")
        
        # System Info
        sys_info = config["system_info"]
        f.write("SYSTEM INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Platform: {sys_info['platform']}\n")
        f.write(f"Python Version: {sys_info['python_version']}\n")
        f.write(f"PyTorch Version: {sys_info['pytorch_version']}\n")
        f.write(f"CUDA Available: {sys_info['cuda_available']}\n")
        if sys_info['cuda_available']:
            f.write(f"CUDA Version: {sys_info['cuda_version']}\n")
            f.write(f"GPU Count: {sys_info['device_count']}\n")
        f.write("\n")
        
        # Additional info
        if "additional_info" in config:
            f.write("ADDITIONAL INFORMATION:\n")
            f.write("-" * 40 + "\n")
            for key, value in config["additional_info"].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Configuration saved at: " + datetime.datetime.now().isoformat() + "\n")
        f.write("=" * 80 + "\n")
    
    print(f"Experiment summary saved to: {summary_path}")

def save_client_info(data, experiment_dir: str):
    """
    Save information about clients and data distribution.
    
    Args:
        data: Data object containing client information
        experiment_dir: Directory where experiment files are stored
    """
    
    if not hasattr(data, 'client_list'):
        print("No client information available to save")
        return
    
    client_info_path = os.path.join(experiment_dir, 'client_information.txt')
    
    with open(client_info_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CLIENT INFORMATION\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Clients: {len(data.client_list)}\n")
        f.write(f"Client IDs: {data.client_list}\n\n")
        
        f.write("CLIENT DATA DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        
        total_samples = 0
        total_classes = 0
        
        for cid in data.client_list:
            if cid in data.train_dataset_sizes and cid in data.train_class_sizes:
                samples = data.train_dataset_sizes[cid]
                classes = data.train_class_sizes[cid]
                total_samples += samples
                total_classes += classes
                
                f.write(f"Client {cid}:\n")
                f.write(f"  - Training Samples: {samples}\n")
                f.write(f"  - Unique Classes: {classes}\n")
                f.write(f"  - Samples per Class (avg): {samples/classes:.1f}\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Training Samples: {total_samples}\n")
        f.write(f"Total Classes (sum): {total_classes}\n")
        f.write(f"Average Samples per Client: {total_samples/len(data.client_list):.1f}\n")
        f.write(f"Average Classes per Client: {total_classes/len(data.client_list):.1f}\n\n")
        
        # Test data info if available
        if hasattr(data, 'test_loaders') and '0' in data.test_loaders:
            f.write("TEST DATA INFORMATION:\n")
            f.write("-" * 30 + "\n")
            
            if hasattr(data, 'query_meta') and '0' in data.query_meta:
                query_size = data.query_meta['0']['sizes']
                f.write(f"Query Samples: {query_size}\n")
            
            if hasattr(data, 'gallery_meta') and '0' in data.gallery_meta:
                gallery_size = data.gallery_meta['0']['sizes']
                f.write(f"Gallery Samples: {gallery_size}\n")
            
            f.write("Note: All clients use shared query/gallery splits for evaluation\n\n")
        
        f.write("=" * 60 + "\n")
    
    print(f"Client information saved to: {client_info_path}")

def save_metadata_info(metadata_file: str, experiment_dir: str):
    """
    Save information about the metadata file used.
    
    Args:
        metadata_file: Path to metadata file or directory
        experiment_dir: Directory where experiment files are stored
    """
    
    if not metadata_file:
        return
    
    metadata_info_path = os.path.join(experiment_dir, 'metadata_information.txt')
    
    # Check if metadata_file is a directory containing metadata.csv
    if os.path.isdir(metadata_file):
        metadata_csv_path = os.path.join(metadata_file, 'metadata.csv')
    else:
        metadata_csv_path = metadata_file
    
    with open(metadata_info_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("METADATA INFORMATION\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Metadata Source: {metadata_file}\n")
        f.write(f"Metadata CSV Path: {metadata_csv_path}\n\n")
        
        if os.path.exists(metadata_csv_path):
            try:
                # Read metadata to get basic statistics
                df = pd.read_csv(metadata_csv_path)
                
                f.write("METADATA STATISTICS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Samples: {len(df)}\n")
                f.write(f"Total Unique IDs: {df['identity'].nunique() if 'identity' in df.columns else 'N/A'}\n")
                
                if 'split' in df.columns:
                    split_counts = df['split'].value_counts()
                    f.write("\nSplit Distribution:\n")
                    for split_name, count in split_counts.items():
                        if split_name == '':
                            f.write(f"  Unused: {count} samples\n")
                        else:
                            unique_ids = df[df['split'] == split_name]['identity'].nunique() if 'identity' in df.columns else 'N/A'
                            f.write(f"  {split_name}: {count} samples ({unique_ids} unique IDs)\n")
                
                if 'client' in df.columns:
                    client_counts = df['client'].value_counts().sort_index()
                    f.write("\nClient Distribution:\n")
                    for client_id, count in client_counts.items():
                        if client_id == -1:
                            f.write(f"  Shared (query/gallery): {count} samples\n")
                        else:
                            train_data = df[(df['client'] == client_id) & (df['split'] == 'train')]
                            unique_ids = train_data['identity'].nunique() if len(train_data) > 0 and 'identity' in df.columns else 0
                            f.write(f"  Client {client_id}: {len(train_data)} training samples ({unique_ids} unique IDs)\n")
                
            except Exception as e:
                f.write(f"Error reading metadata file: {str(e)}\n")
        else:
            f.write("Metadata file not found at expected location\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Metadata information saved to: {metadata_info_path}")