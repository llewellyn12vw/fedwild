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
        self.epoch_cosine_distances = []  # Store cosine distances for each epoch

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
        """Compute FedGKD distillation loss using CSKD with feature-level temperature scaling"""
        if not self.fedgkd_enabled or self.fedgkd_ensemble_teacher is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        # Teacher model setup (unchanged)
        self.fedgkd_ensemble_teacher.eval()
        self.fedgkd_ensemble_teacher = self.fedgkd_ensemble_teacher.to(self.device)
        
        for param in self.fedgkd_ensemble_teacher.parameters():
            param.requires_grad = False
        
        with torch.no_grad():
            teacher_features = self.fedgkd_ensemble_teacher.backbone(inputs)
            teacher_features = self.fedgkd_ensemble_teacher.feature_head(teacher_features)
        
        if student_features.shape != teacher_features.shape:
            print(f"ERROR: Feature dimension mismatch - Student: {student_features.shape}, Teacher: {teacher_features.shape}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Normalize features for stable similarity computation
        student_features_norm = self.normalize_features(student_features)
        teacher_features_norm = self.normalize_features(teacher_features.detach())
        
        # Compute per-sample cosine similarity and distances
        cosine_sim = F.cosine_similarity(student_features_norm, teacher_features_norm, dim=1)  # [batch_size]
        cosine_distance = 1.0 - cosine_sim  # [batch_size]

        batch_mean_distance = torch.mean(cosine_distance).item()  # Convert to Python float
        self.epoch_cosine_distances.append(batch_mean_distance)
        # === TEMPERATURE SCALING BASED ON COSINE DISTANCE ===
        adaptive_temperatures = self.compute_adaptive_temperatures(cosine_distance)
        
        # Apply temperature scaling to features and compute losses
        temperature_scaled_loss = self.compute_temperature_scaled_losses(
            student_features, teacher_features,
            cosine_sim, cosine_distance, adaptive_temperatures
        )
        
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        return temperature_scaled_loss * self.fedgkd_distillation_coeff

    def compute_adaptive_temperatures(self, cosine_distance):
        """
        Compute adaptive temperatures based on cosine distance
        High distance (dissimilar) → High temperature → More aggressive learning
        Low distance (similar) → Low temperature → More conservative learning
        """
        # Temperature range (inspired by CSWT paper)
        T_min = 0.5   # Conservative learning for similar features
        T_max = 2.5   # Aggressive learning for dissimilar features
        
        # Batch-wise normalization for stability
        batch_min = cosine_distance.min()
        batch_max = cosine_distance.max()
        
        # Avoid division by zero
        if batch_max == batch_min:
            temperatures = torch.full_like(cosine_distance, (T_min + T_max) / 2)
        else:
            # Normalize distance to [0, 1] and map to temperature range
            normalized_distance = (cosine_distance - batch_min) / (batch_max - batch_min)
            temperatures = T_min + (T_max - T_min) * normalized_distance
        
        return temperatures  # [batch_size]

    def compute_temperature_scaled_losses(self, student_features, teacher_features, 
                                        cosine_sim, cosine_distance, temperatures):
        batch_size = student_features.shape[0]
        # Temperature constants
        T_min = 0.5
        T_max = 2.5

        base_guidance = (1.0 + cosine_sim).unsqueeze(1)  # [batch_size, 1]
        
        # Temperature modulation - higher temp = stronger guidance for dissimilar samples
        temp_modulation = temperatures.unsqueeze(1)  # [batch_size, 1]
        enhanced_guidance_map = base_guidance * temp_modulation
        
        # === 2. TEMPERATURE-SCALED MSE LOSS ===
        # Apply temperature scaling to feature differences
        feature_diff = student_features - teacher_features  # [batch_size, feature_dim]
        
        # Scale feature differences by temperature
        temperature_scaled_diff = feature_diff * temperatures.unsqueeze(1)  # [batch_size, feature_dim]
        
        # Compute MSE with temperature-scaled differences
        mse_per_sample = (temperature_scaled_diff ** 2).mean(dim=1)  # [batch_size]
        
        # Apply enhanced guidance map
        guided_temp_mse = (enhanced_guidance_map.squeeze(1) * mse_per_sample).mean()
        
        # === 3. TEMPERATURE-SCALED COSINE LOSS ===
        # Inverse temperature weighting for cosine loss
        # High temp (dissimilar) → Focus on reconstruction (MSE)
        # Low temp (similar) → Focus on alignment (cosine)
        inverse_temp_weights = (T_max + T_min) - temperatures  # Invert relationship
        
        temperature_scaled_cosine_loss = (inverse_temp_weights * cosine_distance).mean()
        
        # === 4. ADAPTIVE LAMBDA WEIGHTING ===
        # Adjust lambda based on average temperature in batch
        mean_temperature = temperatures.mean()
        
        # Higher avg temp → More MSE focus (reconstruction)
        # Lower avg temp → More cosine focus (alignment)  
        temp_factor = (mean_temperature - T_min) / (T_max - T_min)  # [0, 1]
        
        lambda_m = 0.5 + 0.3 * temp_factor  # Range: [0.5, 0.8]
        lambda_s = 0.5 - 0.3 * temp_factor  # Range: [0.2, 0.5]
        
        # === 5. COMBINED TEMPERATURE-SCALED LOSS ===
        total_loss = lambda_m * guided_temp_mse + lambda_s * temperature_scaled_cosine_loss
        
        return total_loss
    
    def train(self, federated_model, use_cuda,rnd):
        self.y_err = []
        self.y_loss = []
        self.epoch_cosine_distances = []
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
                    fedgkd_loss = self.compute_fedgkd_distillation_loss(student_features, inputs)
                    loss += fedgkd_loss
                
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
        
        # Store classifier and remove for federated aggregation
        self.classifier = self.model.classifier

        self.distance = self.optimization.cdw_feature_distance(federated_model, self.model)
        
        result_dir = os.path.join(self.project_dir, 'model', self.experiment_name, f'client_{self.cid}')
        os.makedirs(result_dir, exist_ok=True)

        csv_file = os.path.join(result_dir, 'cosine_sim.csv')
        file_exists = os.path.isfile(csv_file)
        row_data = [f"{rnd:.4f}",f"{self.distance:.4f}"]

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ['round', 'cosine_sim']
                writer.writerow(header)
            writer.writerow(row_data)
            
        if self.fedgkd_enabled:
            csv_file = os.path.join(result_dir, 'ensemble_cosine_sim.csv')
            file_exists = os.path.isfile(csv_file)
            epoch_avg_distance = sum(self.epoch_cosine_distances) / len(self.epoch_cosine_distances)
            row_data = [f"{rnd:.4f}",f"{epoch_avg_distance:.4f}"]
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = ['round', 'cosine_ensemble_sim']
                    writer.writerow(header)
                writer.writerow(row_data)

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
        Test the local model on its own client-specific query/gallery split.
        Each client evaluates on its own test data after local training.
        """


        # Use client-specific query/gallery split
        test_loaders = self.data.test_loaders[self.cid]

        model = self.model.eval()
        if use_cuda:
            model = model.cuda()
        else:
            model = model.cpu()

        from utils import extract_feature

        with torch.no_grad():
            gallery_feature = extract_feature(model, test_loaders['gallery'], [1.0], self.data.image_size)
            query_feature = extract_feature(model, test_loaders['query'], [1.0], self.data.image_size)

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