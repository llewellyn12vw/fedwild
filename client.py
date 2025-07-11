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
import math

class Client():
    def __init__(self, cid, data, device, project_dir, model_name, local_epoch, lr, batch_size, drop_rate, stride, experiment_name, model, cosine_annealing=True, total_rounds=100, eta_min=1e-6):
        self.cid = cid
        self.project_dir = project_dir
        self.model_name = model_name
        self.data = data
        self.device = device
        self.local_epoch = local_epoch
        self.initial_lr = lr
        self.current_lr = lr
        self.batch_size = batch_size
        self.model = model
        self.dataset_sizes = self.data.train_dataset_sizes[cid]
        self.train_loader = self.data.train_loaders[cid]
        self.cosine_annealing = cosine_annealing
        self.total_rounds = total_rounds
        self.eta_min = eta_min

        self.full_model = get_model(self.data.train_class_sizes[cid], drop_rate, stride, model)
        self.model_type = model
        
        # Store the classifier and remove for federated aggregation
        self.classifier = self.full_model.classifier
        self.full_model.classifier = nn.Sequential()  # Remove for federated aggregation
            
        self.model = self.full_model
        self.distance=0
        self.optimization = Optimization(self.train_loader, self.device)
        self.experiment_name = experiment_name
        
        # FedGKD-specific parameters
        self.fedgkd_enabled = False
        self.fedgkd_ensemble_teacher = None
        self.fedgkd_distillation_coeff = 0.1
        self.fedgkd_temperature = 1.0
        # print("class name size",class_names_size[cid])

    def update_learning_rate(self, round_num):
        """Update learning rate based on scheduling strategy"""
        if self.cosine_annealing:
            # Cosine annealing scheduling
            self.current_lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                (1 + math.cos(math.pi * round_num / self.total_rounds)) / 2
        else:
            # Original exponential decay
            self.current_lr = self.initial_lr * (0.98 ** round_num)
        
        if round_num % 10 == 0:  # Print every 10 rounds
            print(f"Client {self.cid}, Round {round_num}: LR = {self.current_lr:.8f}")
    
    def receive_fedgkd_teacher(self, ensemble_teacher, distillation_coeff, temperature):
        """Receive ensemble teacher model from server for FedGKD"""
        self.fedgkd_enabled = True
        self.fedgkd_ensemble_teacher = copy.deepcopy(ensemble_teacher)
        self.fedgkd_distillation_coeff = distillation_coeff
        self.fedgkd_temperature = temperature
        print(f"Client {self.cid} received FedGKD ensemble teacher with coeff={distillation_coeff}, temp={temperature}")
    
    def compute_fedgkd_distillation_loss(self, student_output, inputs):
        """Compute FedGKD distillation loss using feature-level distillation"""
        if not self.fedgkd_enabled or self.fedgkd_ensemble_teacher is None:
            return 0.0
            
        # Get teacher features (teacher should already have classifier removed)
        self.fedgkd_ensemble_teacher.eval()
        self.fedgkd_ensemble_teacher = self.fedgkd_ensemble_teacher.to(self.device)
        
        with torch.no_grad():
            teacher_features = self.fedgkd_ensemble_teacher(inputs)
        
        # For ResNet, if student_output is logits, we need to get features
        if student_output.shape[1] != teacher_features.shape[1]:
            # Get features by running input through model without classifier
            # We need to maintain gradients, so don't use torch.no_grad()
            original_classifier = self.model.classifier
            self.model.classifier = nn.Sequential()
            student_features = self.model(inputs)
            self.model.classifier = original_classifier
        else:
            student_features = student_output
        
        # Ensure both have same dimensions
        if student_features.shape != teacher_features.shape:
            print(f"Warning: Feature dimension mismatch - Student: {student_features.shape}, Teacher: {teacher_features.shape}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Use MSE loss for feature-level distillation (more stable than KL for features)
        mse_loss = F.mse_loss(student_features, teacher_features)
        distillation_loss = mse_loss * self.fedgkd_distillation_coeff
        
        return distillation_loss

    def train(self, federated_model, use_cuda,round):
        self.y_err = []
        self.y_loss = []

        # Update learning rate based on round number
        self.update_learning_rate(round)
        self.model.load_state_dict(federated_model.state_dict())
        
        # Restore classifier
        self.model.classifier = self.classifier
        self.old_classifier = copy.deepcopy(self.classifier)
            
        self.model = self.model.to(self.device)

        optimizer = get_optimizer(self.model, self.current_lr)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        
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

                # Forward pass - Standard CrossEntropy loss
                outputs = self.model(inputs)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)
                
                # FedGKD distillation loss
                if self.fedgkd_enabled:
                    fedgkd_loss = self.compute_fedgkd_distillation_loss(outputs, inputs)
                    loss += fedgkd_loss
                
                # Standard predictions
                _, preds = torch.max(outputs.data, 1)
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * b
                running_corrects += float(torch.sum(preds == labels.data))

            # scheduler.step()
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
        
        # Store classifier and remove for federated aggregation
        self.classifier = self.model.classifier
        self.distance = self.optimization.cdw_feature_distance(federated_model, self.old_classifier, self.model)
        self.model.classifier = nn.Sequential()  # Remove for federated aggregation

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
            f"python evaluate.py --result_dir {result_dir} "
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