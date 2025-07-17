from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
import os
import json
import torch
from random_erasing import RandomErasing
from wildlife_tools.data.dataset import WildlifeDataset
from wildlife_datasets.datasets import Cows2021v2, LeopardID2022, HyenaID2022
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
    def __init__(self, datasets, data_dir, batch_size, erasing_p, color_jitter, train_all,multi_s, metadata):
        self.datasets = datasets.split(',')
        self.batch_size = batch_size
        self.erasing_p = erasing_p
        self.color_jitter = color_jitter
        self.data_dir = data_dir
        self.train_all = '_all' if train_all else ''
        self.multi_s = multi_s
        self.metadata = metadata

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

    def preprocess_kd_data(self, dataset):
        loader, image_dataset = self.preprocess_one_train_dataset(dataset)
        self.kd_loader = loader
        print(self.kd_loader)


    def preprocess_one_train_dataset(self, dataset):
        """preprocess a training dataset, construct a data loader.
        """
        data_path = os.path.join(self.data_dir, dataset, 'train.csv')
        # data_path = os.path.join(data_path, 'train' + self.train_all)
        

        transform = self.data_transforms['train']
        if self.multi_s:
            metadata = self.metadata
        else:
            # Use single species metadata
             metadata = LeopardID2022('/home/wellvw12/leopard').root

        # metadata = HyenaID2022('/home/wellvw12/hyenaid2022')
        df = pd.read_csv(data_path)
        image_dataset = WildlifeDataset(df,metadata, transform=transform)

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
        """Test data preprocessing with 20% query and 80% gallery split"""
        self.test_loaders = {}
        self.gallery_meta = {}
        self.query_meta = {}

        transform = self.data_transforms['val']

        for test_dir in self.datasets:
            # if test_dir != 'test': continue
            if self.multi_s:
                metadata = self.metadata
            else:
                # Use single species metadata
                metadata = LeopardID2022('/home/wellvw12/leopard').root
            # metadata = HyenaID2022('/home/wellvw12/hyenaid2022')

            # df = pd.read_csv(f'{self.data_dir}/{test_dir}/test.csv')          
            query = pd.read_csv(f'{self.data_dir}/{test_dir}/query.csv')
            gallery = pd.read_csv(f'{self.data_dir}/{test_dir}/gallery.csv')
            gallery_dataset = WildlifeDataset(gallery, metadata, transform=transform)
            query_dataset = WildlifeDataset(query, metadata, transform=transform)
            # Print distribution stats
            # print(f"Query images: {len(query)} ({len(query)/len(df)*100:.1f}%)")
            # print(f"Gallery images: {len(gallery)} ({len(gallery)/len(df)*100:.1f}%)")

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
