from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
import os
import json
import torch
from random_erasing import RandomErasing
from wildlife_tools.data.dataset import WildlifeDataset
from wildlife_datasets.datasets import Cows2021v2, LeopardID2022, HyenaID2022, MacaqueFaces
import pandas as pd
import torchvision.transforms as T

class ImageDataset(Dataset):
    def __init__(self, imgs,  transform = None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data,label = self.imgs[index]
        return self.transform(Image.open(data)), label


class Data():
    def __init__(self, datasets, data_dir, batch_size, erasing_p, color_jitter, train_all, dataset_type, metadata_file=None):
        self.datasets = datasets.split(',')
        self.batch_size = batch_size
        self.erasing_p = erasing_p
        self.color_jitter = color_jitter
        self.data_dir = data_dir
        self.train_all = '_all' if train_all else ''
        self.dataset_type = dataset_type
        self.metadata_file = metadata_file
        
        # Load unified metadata if provided
        if self.metadata_file and os.path.exists(self.metadata_file):
            print(f"Loading unified metadata from: {self.metadata_file}")
            self.unified_metadata = pd.read_csv(self.metadata_file)
            print(f"Loaded {len(self.unified_metadata)} samples from unified metadata")
            self.use_unified_metadata = True
        else:
            self.unified_metadata = None
            self.use_unified_metadata = False
            print("Using legacy CSV file approach")

    def transform(self):
        transform_train = [
                transforms.Resize((224,224), interpolation=3),
                transforms.Pad(10),
                transforms.RandomCrop((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]

        transform_val = [
                transforms.Resize(size=(224,224),interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]

        if self.erasing_p > 0:
            transform_train = transform_train + [RandomErasing(probability=self.erasing_p, mean=[0.0, 0.0, 0.0])]

        if self.color_jitter:
            transform_train = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train

        self.data_transforms = {
            'train': transforms.Compose(transform_train),
            'val': transforms.Compose(transform_val),
        }

    def _get_metadata_root(self):
        """Get the appropriate metadata root based on dataset type"""
        if self.dataset_type == 'leopard':
            return LeopardID2022('/home/wellvw12/leopard').root
        elif self.dataset_type == 'macaque':
            return MacaqueFaces('/home/wellvw12/fedwild/MacaqueFaces').root
        elif self.dataset_type == 'hyena':
            return HyenaID2022('/home/wellvw12/hyenaid2022').root
        elif self.dataset_type == 'cow':
            return Cows2021v2('/home/wellvw12/cows2021v2').root
        else:
            # Default to macaque if not specified
            return MacaqueFaces('/home/wellvw12/fedwild/MacaqueFaces').root

    def preprocess_kd_data(self, dataset):
        loader, image_dataset = self.preprocess_one_train_dataset(dataset)
        self.kd_loader = loader
        print(self.kd_loader)


    def preprocess_one_train_dataset(self, dataset):
        """preprocess a training dataset, construct a data loader.
        """
        transform = self.data_transforms['train']
        
        if self.use_unified_metadata:
            # Use unified metadata approach
            client_id = int(dataset)  # dataset should be client ID (0, 1, 2, ...)
            
            # Filter metadata for this client's training data
            client_train_data = self.unified_metadata[
                (self.unified_metadata['split'] == 'train') & 
                (self.unified_metadata['client'] == client_id)
            ].copy()
            
            print(f"Client {client_id}: {len(client_train_data)} training samples, "
                  f"{client_train_data['identity'].nunique()} unique IDs")
            
            if len(client_train_data) == 0:
                print(f"Warning: No training data found for client {client_id}")
                # Create empty dataset
                image_dataset = WildlifeDataset(pd.DataFrame(), None, transform=transform)
            else:
                # Determine metadata root based on dataset type
                metadata_root = self._get_metadata_root()
                image_dataset = WildlifeDataset(client_train_data, metadata_root, transform=transform)
        else:
            # Legacy approach: read from separate CSV files
            data_path = os.path.join(self.data_dir, dataset, 'train.csv')
            metadata_root = self._get_metadata_root()

            df = pd.read_csv(data_path)
            image_dataset = WildlifeDataset(df, metadata_root, transform=transform)

        loader = torch.utils.data.DataLoader(
            image_dataset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=0, 
            pin_memory=False)

        return loader, image_dataset

    def preprocess_train(self):
        """preprocess training data, constructing train loaders
        """
        self.train_loaders = {}
        self.train_dataset_sizes = {}
        self.train_class_sizes = {}
        self.client_list = []
        
        for dataset in self.datasets:
            
            if dataset == 'test': continue

            self.client_list.append(dataset)
            loader, image_dataset = self.preprocess_one_train_dataset(dataset)

            self.train_dataset_sizes[dataset] = len(image_dataset)
            self.train_class_sizes[dataset] = image_dataset.num_classes
            self.train_loaders[dataset] = loader
            


    def preprocess_test(self):
        """Test data preprocessing for query and gallery splits"""
        self.test_loaders = {}
        self.gallery_meta = {}
        self.query_meta = {}

        transform = self.data_transforms['val']

        if self.use_unified_metadata:
            # Use unified metadata approach - query/gallery are shared across all clients
            query_data = self.unified_metadata[self.unified_metadata['split'] == 'query'].copy()
            gallery_data = self.unified_metadata[self.unified_metadata['split'] == 'gallery'].copy()
            
            print(f"Query samples: {len(query_data)} ({query_data['identity'].nunique()} unique IDs)")
            print(f"Gallery samples: {len(gallery_data)} ({gallery_data['identity'].nunique()} unique IDs)")
            
            # Determine metadata root based on dataset type
            metadata_root = self._get_metadata_root()
            
            # Create datasets
            if len(query_data) > 0:
                query_dataset = WildlifeDataset(query_data, metadata_root, transform=transform)
            else:
                query_dataset = WildlifeDataset(pd.DataFrame(), None, transform=transform)
                
            if len(gallery_data) > 0:
                gallery_dataset = WildlifeDataset(gallery_data, metadata_root, transform=transform)
            else:
                gallery_dataset = WildlifeDataset(pd.DataFrame(), None, transform=transform)
            
            # Create test loaders for each client (all use same query/gallery)
            for test_dir in self.datasets:
                self.test_loaders[test_dir] = {
                    'gallery': torch.utils.data.DataLoader(
                        gallery_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True
                    ),
                    'query': torch.utils.data.DataLoader(
                        query_dataset,
                        batch_size=self.batch_size,
                        shuffle=False, 
                        num_workers=2,
                        pin_memory=True
                    )
                }

                # Store metadata
                self.gallery_meta[test_dir] = {
                    'sizes': len(gallery_dataset),
                    'labels': gallery_dataset.labels if len(gallery_dataset) > 0 else []
                }
                self.query_meta[test_dir] = {
                    'sizes': len(query_dataset),
                    'labels': query_dataset.labels if len(query_dataset) > 0 else []
                }
        else:
            # Legacy approach: read from separate CSV files
            for test_dir in self.datasets:
                metadata_root = self._get_metadata_root()

                query = pd.read_csv(f'{self.data_dir}/{test_dir}/query.csv')
                gallery = pd.read_csv(f'{self.data_dir}/{test_dir}/gallery.csv')
                gallery_dataset = WildlifeDataset(gallery, metadata_root, transform=transform)
                query_dataset = WildlifeDataset(query, metadata_root, transform=transform)

                self.test_loaders[test_dir] = {
                    'gallery': torch.utils.data.DataLoader(
                        gallery_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True
                    ),
                    'query': torch.utils.data.DataLoader(
                        query_dataset,
                        batch_size=self.batch_size,
                        shuffle=False, 
                        num_workers=2,
                        pin_memory=True
                    )
                }

                # Store metadata
                self.gallery_meta[test_dir] = {
                    'sizes': len(gallery_dataset),
                    'labels': gallery_dataset.labels
                }
                self.query_meta[test_dir] = {
                    'sizes': len(query_dataset),
                    'labels': query_dataset.labels
                }

    def preprocess(self):
        self.transform()
        self.preprocess_train()
        self.preprocess_test()
        # self.preprocess_kd_data('kd')

def get_camera_ids(img_paths):
    """get camera id and labels by image path
    """
    camera_ids = []
    labels = []
    for path, v in img_paths:
        filename = os.path.basename(path)
        if filename[:3]!='cam':
            label = filename[0:4]
            camera = filename.split('c')[1]
            camera = camera.split('s')[0]
        else:
            label = filename.split('_')[2]
            camera = filename.split('_')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_ids.append(int(camera[0]))
    return camera_ids, labels
