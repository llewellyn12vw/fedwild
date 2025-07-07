import time
import torch
from utils import get_optimizer, get_model
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from optimization import Optimization
import os
import scipy.io
import numpy as np

class Client():
    def __init__(self, cid, data, device, project_dir, model_name, local_epoch, lr, batch_size, drop_rate, stride, experiment_name='experimentX'):
        self.cid = cid
        self.project_dir = project_dir
        self.model_name = model_name
        self.data = data
        self.device = device
        self.local_epoch = local_epoch
        self.lr = lr
        self.batch_size = batch_size
        
        self.dataset_sizes = self.data.train_dataset_sizes[cid]
        self.train_loader = self.data.train_loaders[cid]

        self.full_model = get_model(self.data.train_class_sizes[cid], drop_rate, stride)
        # For MegaDescriptor, store the ArcFace head instead of classifier
        self.arcface_head = self.full_model.arcface_head
        self.full_model.arcface_head = nn.Sequential()  # Remove for federated aggregation
        self.model = self.full_model
        self.distance=0
        self.optimization = Optimization(self.train_loader, self.device)
        self.experiment_name = experiment_name
        # print("class name size",class_names_size[cid])

    def train(self, federated_model, use_cuda,round):
        self.y_err = []
        self.y_loss = []

        self.model.load_state_dict(federated_model.state_dict())
        self.model.arcface_head = self.arcface_head
        self.old_arcface_head = copy.deepcopy(self.arcface_head)
        self.model = self.model.to(self.device)

        optimizer = get_optimizer(self.model, self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        
        # ArcFace loss is computed within the model forward pass

        since = time.time()

        print('Client', self.cid, 'start training')
        for epoch in range(self.local_epoch):
            print('Epoch {}/{}'.format(epoch, self.local_epoch - 1))
            print('-' * 10)

            # scheduler.step()
            self.model.train(True)
            running_loss = 0.0
            running_corrects = 0.0
            
            for data in self.train_loader:
                inputs, labels = data
                b, c, h, w = inputs.shape
                if b < self.batch_size:
                    continue
                if use_cuda:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                optimizer.zero_grad()

                # Forward pass with ArcFace loss
                loss = self.model(inputs, labels)
                
                # For accuracy computation, get features and compute similarity-based predictions
                with torch.no_grad():
                    features = self.model(inputs)  # Get embeddings without labels
                    # Simple nearest neighbor in feature space for accuracy estimation
                    preds = self._compute_predictions(features, labels)
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * b
                running_corrects += float(torch.sum(preds == labels.data))
            scheduler.step()
            used_data_sizes = (self.dataset_sizes - self.dataset_sizes % self.batch_size)
            epoch_loss = running_loss / used_data_sizes
            epoch_acc = running_corrects / used_data_sizes

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'train', epoch_loss, epoch_acc))

            self.y_loss.append(epoch_loss)
            self.y_err.append(1.0-epoch_acc)

            time_elapsed = time.time() - since
            print('Client', self.cid, ' Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        time_elapsed = time.time() - since
        print('Client', self.cid, 'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
        # save_network(self.model, self.cid, 'last', self.project_dir, self.model_name, gpu_ids)
        
        self.arcface_head = self.model.arcface_head
        self.distance = self.optimization.cdw_feature_distance(federated_model, self.old_arcface_head, self.model)
        self.model.arcface_head = nn.Sequential()  # Remove for federated aggregation

        if round == 0 or (round+1)%10 == 0:
            print("Round 1: Client", self.cid, "local model trained, distance:", self.distance)
            self.test(use_cuda)

    def test(self, use_cuda=True):
        """
        Test the local model on the client's query/gallery set and store results in local_result.csv.
        """
        # Check if test loaders exist for this client
        if not hasattr(self.data, "test_loaders") or self.cid not in self.data.test_loaders:
            print(f"No test data for client {self.cid}")
            return

        test_loaders = self.data.test_loaders[self.cid]
        if 'query' not in test_loaders or 'gallery' not in test_loaders:
            print(f"No query/gallery split for client {self.cid}")
            return

        model = self.model.eval()
        if use_cuda:
            model = model.cuda()
        else:
            model = model.cpu()

        from utils import extract_feature

        with torch.no_grad():
            gallery_feature = extract_feature(model, test_loaders['gallery'], [1.0])
            query_feature = extract_feature(model, test_loaders['query'], [1.0])

        result = {
            'gallery_f': gallery_feature.cpu().numpy(),
            'gallery_label': self.data.gallery_meta[self.cid]['labels'],
            'query_f': query_feature.cpu().numpy(),
            'query_label': self.data.query_meta[self.cid]['labels'],
        }

        # Save .mat file for compatibility with evaluate.py
        result_dir = os.path.join(self.project_dir, 'model',self.experiment_name, f'client_{self.cid}')
        os.makedirs(result_dir, exist_ok=True)
        mat_path = os.path.join(result_dir, 'pytorch_result.mat')
        scipy.io.savemat(mat_path, result)

        # Call evaluate.py to compute metrics and store in local_result.csv
        output_file = os.path.join(result_dir, 'local_result.csv')
        cmd = (
            f"python /home/wellvw12/fedReID/evaluate.py --result_dir {result_dir} "
            f"--dataset client_{self.cid} --output_file local_result.csv"
        )
        os.system(cmd)
        print(f"Client {self.cid} local test results saved to {output_file}")

    def generate_soft_label(self, x, regularization, temperature=4.0):
        self.model = self.model.to(x.device)  # Ensure model matches input device
        return self.optimization.kd_generate_soft_label(self.model, x, regularization, temperature)

    def get_model(self):
        return self.model

    def get_data_sizes(self):
        return self.dataset_sizes

    def get_train_loss(self):
        return self.y_loss[-1]

    def get_cos_distance_weight(self):
        return self.distance
    
    def _compute_predictions(self, features, labels):
        """
        Compute predictions for ArcFace training by using cosine similarity.
        This provides a rough accuracy estimate during training.
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(features, features.t())
        
        # Get the most similar sample (excluding self)
        similarity_matrix.fill_diagonal_(-float('inf'))
        _, indices = torch.max(similarity_matrix, dim=1)
        
        # Return predicted labels based on most similar samples
        return labels[indices]