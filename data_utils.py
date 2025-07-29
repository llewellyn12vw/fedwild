from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import datasets, transforms
import os
import json
import torch
import numpy as np
from random_erasing import RandomErasing
from wildlife_tools.data.dataset import WildlifeDataset
from wildlife_datasets.datasets import Cows2021v2, LeopardID2022, HyenaID2022, MacaqueFaces
from czechlynx_dataset import CzechLynxDataset
import pandas as pd
import torchvision.transforms as T


# Custom transform classes for client-specific effects
class DarkenTransform:
    def __init__(self, factor=0.3):
        self.factor = factor
    
    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(self.factor)


class AddNoise:
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level
    
    def __call__(self, img):
        img_array = np.array(img)
        noise = np.random.normal(0, self.noise_level * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)


class BrightenTransform:
    def __init__(self, factor=1.7):
        self.factor = factor
    
    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(self.factor)


class WeatherEffect:
    def __init__(self):
        pass
    
    def __call__(self, img):
        # Simulate weather by reducing contrast and adding slight blur
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(0.7)
        return img.filter(ImageFilter.GaussianBlur(radius=0.5))


class CompressionArtifacts:
    def __init__(self, quality=30):
        self.quality = quality
    
    def __call__(self, img):
        # Simulate compression by saving/loading with low quality
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)



class Data():
    def __init__(self, datasets, data_dir, batch_size, erasing_p, color_jitter, train_all, dataset_type, metadata_file=None, image_size=128, use_original_transform_only=False):
        self.datasets = datasets.split(',')
        self.batch_size = batch_size
        self.erasing_p = erasing_p
        self.color_jitter = color_jitter
        self.data_dir = data_dir
        self.train_all = '_all' if train_all else ''
        self.dataset_type = dataset_type
        self.metadata_file = metadata_file
        self.image_size = image_size  # Dynamic image size
        self.use_original_transform_only = use_original_transform_only  # Option to use only transform '0'

            
        self.unified_metadata = pd.read_csv(self.metadata_file)
        print(f"Loaded {len(self.unified_metadata)} samples from unified metadata")
        
        # Get available clients from metadata
        available_clients = self.unified_metadata['client'].unique()
        print(f"Available clients: {available_clients}")
        
        # Update datasets list to match available clients
        self.datasets = [str(client) for client in available_clients]
        print(f"Client datasets: {self.datasets}")


    def transform(self):
        # Client-specific transform definitions
        self.transforms_dict = {
            '0': {  # "Original/Control" client
                'name': 'Original',
                'train_transform': transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size), interpolation=3),
                    transforms.Pad(10),
                    transforms.RandomCrop((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test_transform': transforms.Compose([
                    transforms.Resize(size=(self.image_size, self.image_size), interpolation=3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            },
            
            '1': {  # "Low Light/Night Vision" client
                'name': 'Low_Light', 
                'train_transform': transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size), interpolation=3),
                    transforms.Pad(10),
                    transforms.RandomCrop((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(),
                    DarkenTransform(factor=0.3),
                    AddNoise(noise_level=0.05),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test_transform': transforms.Compose([
                    transforms.Resize(size=(self.image_size, self.image_size), interpolation=3),
                    DarkenTransform(factor=0.3),
                    AddNoise(noise_level=0.05),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            },
            
            '2': {  # "Overexposed/Bright" client
                'name': 'Bright',
                'train_transform': transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size), interpolation=3),
                    transforms.Pad(10),
                    transforms.RandomCrop((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(),
                    BrightenTransform(factor=1.4),
                    transforms.ColorJitter(contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test_transform': transforms.Compose([
                    transforms.Resize(size=(self.image_size, self.image_size), interpolation=3),
                    BrightenTransform(factor=1.4),
                    transforms.ColorJitter(contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            },
            
            '3': {  # "Weather/Environmental" client  
                'name': 'Weather',
                'train_transform': transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size), interpolation=3),
                    transforms.Pad(10),
                    transforms.RandomCrop((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(),
                    WeatherEffect(),
                    transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test_transform': transforms.Compose([
                    transforms.Resize(size=(self.image_size, self.image_size), interpolation=3),
                    WeatherEffect(),
                    transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            },
            
            '4': {  # "Camera Quality/Compression" client
                'name': 'Low_Quality',
                'train_transform': transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size), interpolation=3),
                    transforms.Pad(10),
                    transforms.RandomCrop((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(),
                    CompressionArtifacts(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test_transform': transforms.Compose([
                    transforms.Resize(size=(self.image_size, self.image_size), interpolation=3),
                    CompressionArtifacts(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            }
        }

        # Add additional effects if specified
        if self.erasing_p > 0:
            for client_id in self.transforms_dict:
                train_transforms = list(self.transforms_dict[client_id]['train_transform'].transforms)
                train_transforms.append(RandomErasing(probability=self.erasing_p, mean=[0.0, 0.0, 0.0]))
                self.transforms_dict[client_id]['train_transform'] = transforms.Compose(train_transforms)
        
        # Legacy transforms for backward compatibility
        self.data_transforms = {
            'train': self.transforms_dict.get('0', self.transforms_dict['0'])['train_transform'],
            'val': self.transforms_dict.get('0', self.transforms_dict['0'])['test_transform'],
        }

    def _create_dataset(self, client_data, transform):
        """Factory function to create species-specific dataset instances"""
        if self.dataset_type == 'czechlynx':
            root = CzechLynxDataset('/home/wellvw12/.cache/kagglehub/datasets/picekl/czechlynx/versions/6').root
            return WildlifeDataset(client_data,root, transform=transform)
        elif self.dataset_type == 'leopard':
            root = LeopardID2022('/home/wellvw12/leopard').root
            return WildlifeDataset(client_data,root, transform=transform)
        elif self.dataset_type == 'macaque':
            root = MacaqueFaces('/home/wellvw12/fedwild/MacaqueFaces').root
            return WildlifeDataset(client_data,root, transform=transform)


    def preprocess_kd_data(self, dataset):
        loader, _ = self.preprocess_one_train_dataset(dataset)
        self.kd_loader = loader
        print(self.kd_loader)


    def preprocess_one_train_dataset(self, dataset):
        """preprocess a training dataset, construct a data loader.
        """
        # Use only original transform '0' if option is enabled
        if self.use_original_transform_only:
            transform = self.transforms_dict['0']['train_transform']
            print(f"Client {dataset}: Using original transform '0' for training")
        else:
            # Use client-specific transform if available, otherwise use default
            if dataset in self.transforms_dict:
                transform = self.transforms_dict[dataset]['train_transform']
            else:
                transform = self.data_transforms['train']
        
        client_train_data = self.unified_metadata[
            (self.unified_metadata['split'] == 'train') & 
            (self.unified_metadata['client'] ==int(dataset))
        ].copy()
        
        print(f"Client {dataset}: {len(client_train_data)} training samples, "
              f"{client_train_data['identity'].nunique()} unique IDs")

        image_dataset = self._create_dataset(client_train_data, transform)

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
            if dataset == '-1': continue
            print(dataset)
            self.client_list.append(dataset)
            loader, image_dataset = self.preprocess_one_train_dataset(dataset)

            self.train_dataset_sizes[dataset] = len(image_dataset)
            self.train_class_sizes[dataset] = image_dataset.num_classes
            self.train_loaders[dataset] = loader
            


    def preprocess_test(self):
        """Test data preprocessing - create client test sets from '-1' data with client transforms"""
        self.test_loaders = {}
        self.gallery_meta = {}
        self.query_meta = {}
        
        # Get global test data (client '-1') for base data
        global_query_data = self.unified_metadata[
            (self.unified_metadata['split'] == 'query') & 
            (self.unified_metadata['client'] == -1)
        ].copy()
        
        global_gallery_data = self.unified_metadata[
            (self.unified_metadata['split'] == 'gallery') & 
            (self.unified_metadata['client'] == -1)
        ].copy()
        
        
        # Create test loaders for each client using global data with client-specific transforms
        for client_name in self.datasets:
            if client_name == '-1':
                continue  # Skip the source data itself
            
            # Use only original transform '0' if option is enabled
            if self.use_original_transform_only:
                transform = self.transforms_dict['0']['test_transform']
                print(f"Client {client_name}: Using original transform '0' for testing")
            else:
                # Use client-specific transform if available, otherwise use default
                if client_name in self.transforms_dict:
                    transform = self.transforms_dict[client_name]['test_transform']
                else:
                    transform = self.data_transforms['val']
            
            
            # Use the same global data but with client-specific transforms
            query_dataset = self._create_dataset(global_query_data, transform)
            gallery_dataset = self._create_dataset(global_gallery_data, transform)
            
            self.test_loaders[client_name] = {
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
            self.gallery_meta[client_name] = {
                'sizes': len(gallery_dataset),
                'labels': gallery_dataset.labels_string if hasattr(gallery_dataset, 'labels_string') else []
            }
            self.query_meta[client_name] = {
                'sizes': len(query_dataset),
                'labels': query_dataset.labels_string if hasattr(query_dataset, 'labels_string') else []
            }

    def preprocess(self):
        self.transform()
        self.preprocess_train()
        self.preprocess_test()
        # self.preprocess_kd_data('kd')


