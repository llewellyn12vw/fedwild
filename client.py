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
import csv
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
        self.fedgkd_temperature = 2.0
        # print("class name size",class_names_size[cid])

    def update_learning_rate(self, round_num):
        """Update learning rate based on scheduling strategy"""
        if self.cosine_annealing:
            # Cosine annealing scheduling
            self.current_lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                (1 + math.cos(math.pi * round_num / self.total_rounds)) / 2
        else:
            # Original exponential decay
            # self.current_lr = self.initial_lr * (0.98 ** round_num)
            pass
        
        if round_num % 10 == 0:  # Print every 10 rounds
            print(f"Client {self.cid}, Round {round_num}: LR = {self.current_lr:.8f}")
    
    def receive_fedgkd_teacher(self, ensemble_teacher, distillation_coeff, temperature):
        """Receive ensemble teacher model from server for FedGKD with feature-level distillation"""
        self.fedgkd_enabled = True
        self.fedgkd_ensemble_teacher = copy.deepcopy(ensemble_teacher)
        self.fedgkd_distillation_coeff = distillation_coeff
        self.fedgkd_temperature = temperature
        print(f"Client {self.cid} received FedGKD ensemble teacher for feature distillation with coeff={distillation_coeff}, temp={temperature}")
    
    def normalize_features(self, features):
        """Normalize features for stable distillation"""
        # L2 normalization for stable cosine similarity computation
        return F.normalize(features, p=2, dim=1)
    
    def compute_fedgkd_distillation_loss(self, student_features, inputs):
        """Compute FedGKD distillation loss using CSKD (Cosine Similarity-guided Knowledge Distillation)"""
        if not self.fedgkd_enabled or self.fedgkd_ensemble_teacher is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        # Proper teacher model gradient isolation
        self.fedgkd_ensemble_teacher.eval()
        self.fedgkd_ensemble_teacher = self.fedgkd_ensemble_teacher.to(self.device)
        
        # Ensure teacher gradients are blocked
        for param in self.fedgkd_ensemble_teacher.parameters():
            param.requires_grad = False
        
        with torch.no_grad():
            # Extract features from teacher model
            teacher_features = self.fedgkd_ensemble_teacher.backbone(inputs)
            teacher_features = self.fedgkd_ensemble_teacher.feature_head(teacher_features)
        
        if student_features.shape != teacher_features.shape:
            print(f"ERROR: Feature dimension mismatch - Student: {student_features.shape}, Teacher: {teacher_features.shape}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Normalize features for stable similarity computation
        student_features_norm = self.normalize_features(student_features)
        teacher_features_norm = self.normalize_features(teacher_features.detach())
        
        # CSKD Loss: L = λ_m * G * MSE + λ_s * (1 - CosineSim)
        cosine_sim = F.cosine_similarity(student_features_norm, teacher_features_norm, dim=1)
        cosine_sim_mean = cosine_sim.mean()
        
        # Guidance map G based on cosine similarities (reduces capacity gap issues)
        guidance_map = (1.0 + cosine_sim).unsqueeze(1)  # Shape: [batch_size, 1]
        
        # MSE loss with guidance
        mse_loss = F.mse_loss(student_features, teacher_features, reduction='none')  # Per-sample loss
        guided_mse = (guidance_map * mse_loss).mean()  # Apply guidance and average
        
        # Cosine similarity loss (encourages directional alignment)
        cosine_loss = (1.0 - cosine_sim).mean()
        
        # Combined CSKD loss with adaptive weighting
        lambda_m = 0.7  # MSE weight
        lambda_s = 0.3  # Cosine similarity weight
        
        cskd_loss = lambda_m * guided_mse + lambda_s * cosine_loss
        distillation_loss = cskd_loss * self.fedgkd_distillation_coeff
        
        # Enhanced debugging
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 5 == 0:  # Print every 30 batches
            student_norm = torch.norm(student_features, dim=1).mean()
            teacher_norm = torch.norm(teacher_features, dim=1).mean()

            result_dir = os.path.join(self.project_dir, 'model',self.experiment_name, f'client_{self.cid}')
            os.makedirs(result_dir, exist_ok=True)

            csv_file = os.path.join(result_dir, 'cosine_sim.csv')
            file_exists = os.path.isfile(csv_file)
            row_data = [f"{cosine_sim_mean:.4f}",f"{student_norm:.4f}", f"{teacher_norm:.4f}"]  # Store round and last loss value

            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = ['cosine_sim','student_norm','teacher_norm']  # Replace with your column names
                    writer.writerow(header)
                writer.writerow(row_data)

            print(f"  [CSKD DEBUG] Student norm: {student_norm:.4f}, Teacher norm: {teacher_norm:.4f}")
            print(f"  [CSKD DEBUG] Cosine sim: {cosine_sim_mean:.4f}, Guided MSE: {guided_mse:.6f}, Cosine loss: {cosine_loss:.6f}")
            print(f"  [CSKD DEBUG] Total CSKD: {cskd_loss:.6f}, Final loss: {distillation_loss:.6f}")
        
        return distillation_loss
    
    def compute_cosine_sim(self, student_features, teacher_features):
        """Compute cosine similarity between student and teacher features"""
        if student_features.shape != teacher_features.shape:
            print(f"ERROR: Feature dimension mismatch - Student: {student_features.shape}, Teacher: {teacher_features.shape}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Normalize features for stable similarity computation
        student_features_norm = self.normalize_features(student_features)
        teacher_features_norm = self.normalize_features(teacher_features.detach())
        
        cosine_sim = F.cosine_similarity(student_features_norm, teacher_features_norm, dim=1)
        return cosine_sim.mean()
    
    def train(self, federated_model, use_cuda,rnd):
        self.y_err = []
        self.y_loss = []
        #rnd = round
        self.update_learning_rate(rnd)
        
        # Store global model state for cosine similarity computation
        global_model_copy = copy.deepcopy(federated_model)
        global_model_copy = global_model_copy.to(self.device)
        
        self.model.load_state_dict(federated_model.state_dict())
    
        
        self.model.classifier = self.classifier
        self.old_classifier = copy.deepcopy(self.classifier)
            
        self.model = self.model.to(self.device)
        optimizer = get_optimizer(self.model, self.current_lr)

        since = time.time()
        print('Client', self.cid, 'start training')
        for epoch in range(self.local_epoch):
            print('Epoch {}/{}'.format(epoch, self.local_epoch - 1))
            print('-' * 10)
            # scheduler.step()
            self.model.train(True)
            running_loss = 0.0
            running_corrects = 0.0
            batch_count = 0
            
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

                # Extract features for FedGKD distillation
                student_features = self.model.backbone(inputs)
                student_features = self.model.feature_head(student_features)
                
                # Get final outputs for classification loss
                outputs = self.model(inputs)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)
                
                if self.fedgkd_enabled:
                    classification_loss = loss.item()
                    fedgkd_loss = self.compute_fedgkd_distillation_loss(student_features, inputs)
                    loss += fedgkd_loss
                    # print(f"Client {self.cid} - Classification: {classification_loss:.4f}, FedGKD: {fedgkd_loss.item():.4f}, Total: {loss.item():.4f}, Ratio: {fedgkd_loss.item()/classification_loss:.3f}")
                
                _, preds = torch.max(outputs.data, 1)
                                
                loss.backward()
                                
                optimizer.step()
                running_loss += loss.item() * b
                running_corrects += float(torch.sum(preds == labels.data))
                batch_count += 1

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
        
        # Compute cosine similarity between global and local model every 10 rounds when FedGKD is not active
        if (rnd == 0 or rnd % 10 == 0) and not self.fedgkd_enabled:
            self.model.eval()
            global_model_copy.eval()
            
            # Sample a batch from training data to compute feature similarity
            sample_data = next(iter(self.train_loader))
            sample_inputs, _ = sample_data
            if use_cuda:
                sample_inputs = sample_inputs.cuda()
            
            with torch.no_grad():
                # Extract features from both models
                local_features = self.model.backbone(sample_inputs)
                local_features = self.model.feature_head(local_features)
                
                global_features = global_model_copy.backbone(sample_inputs)
                global_features = global_model_copy.feature_head(global_features)
                
                # Compute cosine similarity
                cosine_similarity = self.compute_cosine_sim(local_features, global_features)
                
                print(f"Client {self.cid} Round {rnd}: Global-Local Cosine Similarity = {cosine_similarity:.4f}")
                
                # Log to CSV file
                result_dir = os.path.join(self.project_dir, 'model', self.experiment_name, f'client_{self.cid}')
                os.makedirs(result_dir, exist_ok=True)
                
                csv_file = os.path.join(result_dir, 'global_local_cosine_sim.csv')
                file_exists = os.path.isfile(csv_file)
                row_data = [self.cid, rnd, f"{cosine_similarity:.6f}"]
                
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        header = ['client', 'round', 'global_local_cosine_sim']
                        writer.writerow(header)
                    writer.writerow(row_data)
        
        # Store classifier and remove for federated aggregation
        self.classifier = self.model.classifier
        self.distance = self.optimization.cdw_feature_distance(federated_model, self.old_classifier, self.model)
        
        self.model.classifier = nn.Sequential()  # Remove for federated aggregation
        
        result_dir = os.path.join(self.project_dir, 'model',self.experiment_name, f'client_{self.cid}')
        os.makedirs(result_dir, exist_ok=True)

        csv_file = os.path.join(result_dir, 'loss.csv')
        file_exists = os.path.isfile(csv_file)
        row_data = [self.cid,rnd, round(self.y_loss[-1],3)]  # Store round and last loss value

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ['client','round','loss']  # Replace with your column names
                writer.writerow(header)
            
            writer.writerow(row_data)

        if rnd == 0 or (rnd+1)%10 == 0:
            print("Round 1: Client", self.cid, "local model trained, distance:", self.distance)
            self.test(use_cuda)

    def test(self, use_cuda=True):
        """
        Test the local model on shared query/gallery split and store results in local_result.csv.
        All clients now use the same query/gallery split for consistent evaluation.
        """
        # Check if test loaders exist - use shared split (dataset '0' contains shared query/gallery)
        if not hasattr(self.data, "test_loaders") or '0' not in self.data.test_loaders:
            print(f"No shared test data available for evaluation")
            return

        # Use shared query/gallery split (dataset '0') instead of client-specific splits
        test_loaders = self.data.test_loaders['0']
        if 'query' not in test_loaders or 'gallery' not in test_loaders:
            print(f"No shared query/gallery split available")
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
            'gallery_label': self.data.gallery_meta['0']['labels'],
            'query_f': query_feature.cpu().numpy(),
            'query_label': self.data.query_meta['0']['labels'],
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
            f"--dataset {self.cid} --output_file local_result.csv "
            # f"--enable_species_eval --species_a leopard --species_b hyena"
        )
        os.system(cmd)
        print(f"Client {self.cid} local test results saved to {output_file}")
        print(f"Client {self.cid} species-specific results saved to leopard_evaluation_results.csv and hyena_evaluation_results.csv")

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