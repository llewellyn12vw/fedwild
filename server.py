import os
import math
import json
import matplotlib.pyplot as plt
from utils import get_model, extract_feature
import torch.nn as nn
import torch
import scipy.io
import copy
import random
import torch.optim as optim
from torchvision import datasets

def add_model(dst_model, src_model, dst_no_data, src_no_data):
    if dst_model is None:
        result = copy.deepcopy(src_model)
        return result
    
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data*src_no_data + dict_params2[name1].data*dst_no_data)
    

    return dst_model

def scale_model(model, scale):
    params = model.named_parameters()
    dict_params = dict(params)
    
    # DEBUG: Before scaling
    param_sum_before = sum(p.sum().item() for p in model.parameters())
    
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
        
    return model
#[2,3,1]
def aggregate_models(models, weights):
    """aggregate models based on weights
    params:
        models: model updates from clients
        weights: weights for each model, e.g. by data sizes or cosine distance of features
    """
    if models == []:
        return None
        
    model = add_model(None, models[0], 0, weights[0])
    total_no_data = weights[0]
        
    for i in range(1, len(models)):
        model = add_model(model, models[i], total_no_data, weights[i])
        model = scale_model(model, 1.0 / (total_no_data+weights[i]))
        total_no_data = total_no_data + weights[i]
            
    return model



class Server():
    def __init__(self, clients, data, device, project_dir, model_name, num_of_clients, lr, drop_rate, stride, multiple_scale,experiment_name,model, kd_lr_ratio=0.05):
        self.project_dir = project_dir
        self.data = data
        self.device = device
        self.model_name = model_name
        self.clients = clients
        self.client_list = self.data.client_list
        self.num_of_clients = num_of_clients
        self.lr = lr
        self.initial_kd_lr = lr * kd_lr_ratio  # Initial KD learning rate as ratio of client LR
        self.current_kd_lr = self.initial_kd_lr
        self.multiple_scale = multiple_scale
        self.drop_rate = drop_rate
        self.stride = stride
        self.experiment_name = experiment_name
        self.model = model
        
        # FedGKD-specific parameters
        self.fedgkd_enabled = False
        self.fedgkd_buffer_length = 5
        self.fedgkd_distillation_coeff = 0.1
        self.fedgkd_temperature = 1.0
        self.fedgkd_avg_param = True  # True for FedGKD, False for FedGKD-VOTE
        self.fedgkd_start_round = 0  # Round to start applying distillation loss
        self.fedgkd_models_buffer = []  # Store historical global models
        self.fedgkd_ensemble_teacher = None  # Ensemble teacher model
        self.fedgkd_model_weights = []  # Weights for FedGKD-VOTE
        self.historical_cdw_weights = []  # Track CDW weights per round for FedGKD-VOTE

        self.multiple_scale = []
        for s in multiple_scale.split(','):
            self.multiple_scale.append(math.sqrt(float(s)))

        self.full_model = get_model(750, drop_rate, stride, model).to(device)
        
        # Remove classifier for federated aggregation
        self.full_model.classifier = nn.Sequential()
            
        self.federated_model=self.full_model
        self.federated_model.eval()
        self.train_loss = []
    
    def configure_fedgkd(self, buffer_length=3, distillation_coeff=0.5, temperature=2.0, avg_param=False, start_round=0):
        """Configure FedGKD parameters with research-backed defaults"""
        self.fedgkd_enabled = True
        self.fedgkd_buffer_length = buffer_length  # Shorter buffer for more recent knowledge
        self.fedgkd_distillation_coeff = distillation_coeff  # Higher coefficient for meaningful impact
        self.fedgkd_temperature = temperature
        self.fedgkd_avg_param = avg_param  # Default to FedGKD-VOTE for performance weighting
        self.fedgkd_start_round = start_round  # Round to start applying distillation
        print(f"FedGKD configured with CSKD: buffer_length={buffer_length}, distillation_coeff={distillation_coeff}, temperature={temperature}, avg_param={avg_param}, start_round={start_round}")
        print(f"Using {'FedGKD (simple averaging)' if avg_param else 'FedGKD-VOTE (performance weighting)'}")
        if start_round > 0:
            print(f"FedGKD distillation will start from round {start_round} (buffer fills from round 0)")
    
    def ensemble_historical_models(self):
        """Create ensemble teacher by averaging historical models"""
        if not self.fedgkd_models_buffer:
            return None
            
        if self.fedgkd_avg_param:
            # FedGKD: Average all models in buffer
            ensemble_model = copy.deepcopy(self.fedgkd_models_buffer[0])
            
            if len(self.fedgkd_models_buffer) > 1:
                # Average parameters with proper type handling
                ensemble_state = ensemble_model.state_dict()
                
                for key in ensemble_state.keys():
                    if ensemble_state[key].dtype.is_floating_point:
                        ensemble_state[key] = torch.zeros_like(ensemble_state[key])
                    else:
                        # For integer tensors, convert to float for averaging
                        ensemble_state[key] = torch.zeros_like(ensemble_state[key], dtype=torch.float32)
                    
                for model in self.fedgkd_models_buffer:
                    model_state = model.state_dict()
                    for key in ensemble_state.keys():
                        if model_state[key].dtype.is_floating_point:
                            ensemble_state[key] += model_state[key]
                        else:
                            # Convert integer tensors to float before adding
                            ensemble_state[key] += model_state[key].float()
                
                # Average with proper type handling
                num_models = len(self.fedgkd_models_buffer)
                for key in ensemble_state.keys():
                    if ensemble_state[key].dtype.is_floating_point:
                        ensemble_state[key] /= num_models
                    else:
                        # For integer tensors (like batch norm stats), convert to float first
                        ensemble_state[key] = ensemble_state[key].float() / num_models
                    
                ensemble_model.load_state_dict(ensemble_state)
            
            return ensemble_model
        else:
            # FedGKD-VOTE: Return weighted ensemble based on CDW performance
            if not self.fedgkd_model_weights:
                # Fallback: equal weights if no CDW data available
                print("FedGKD-VOTE: No weights available, using equal weighting")
                return self.fedgkd_models_buffer[0] if len(self.fedgkd_models_buffer) == 1 else None
            
            # Create weighted ensemble
            ensemble_model = copy.deepcopy(self.fedgkd_models_buffer[0])
            
            ensemble_state = ensemble_model.state_dict()
            
            for key in ensemble_state.keys():
                if ensemble_state[key].dtype.is_floating_point:
                    ensemble_state[key] = torch.zeros_like(ensemble_state[key])
                else:
                    # For integer tensors, convert to float for weighted averaging
                    ensemble_state[key] = torch.zeros_like(ensemble_state[key], dtype=torch.float32)
                
            for i, model in enumerate(self.fedgkd_models_buffer):
                weight = self.fedgkd_model_weights[i] if i < len(self.fedgkd_model_weights) else (1.0 / len(self.fedgkd_models_buffer))
                model_state = model.state_dict()
                for key in ensemble_state.keys():
                    if model_state[key].dtype.is_floating_point:
                        ensemble_state[key] += weight * model_state[key]
                    else:
                        # Convert integer tensors to float before weighted addition
                        ensemble_state[key] += weight * model_state[key].float()
            
            ensemble_model.load_state_dict(ensemble_state)
            return ensemble_model
    
    def update_fedgkd_buffer(self, new_model):
        """Update FedGKD models buffer with new global model"""
        if not self.fedgkd_enabled:
            return
            
        model_copy = copy.deepcopy(new_model)
        self.fedgkd_models_buffer.append(model_copy)
        
        # Maintain buffer length
        if len(self.fedgkd_models_buffer) > self.fedgkd_buffer_length:
            self.fedgkd_models_buffer.pop(0)
        
        # Update ensemble teacher
        self.fedgkd_ensemble_teacher = self.ensemble_historical_models()
        
        print(f"FedGKD buffer updated. Buffer size: {len(self.fedgkd_models_buffer)}")
        
        # Debug: Validate ensemble teacher
        if self.fedgkd_ensemble_teacher is not None:
            param_count = sum(p.numel() for p in self.fedgkd_ensemble_teacher.parameters())
            print(f"FedGKD ensemble teacher created with {param_count} parameters")
        else:
            print("WARNING: FedGKD ensemble teacher is None!")
    
    def send_fedgkd_teacher_to_clients(self, selected_clients, current_round):
        """Send ensemble teacher to selected clients if distillation should start"""
        if not self.fedgkd_enabled or self.fedgkd_ensemble_teacher is None:
            return
        
        # Only send teacher if we're at or past the start round
        if current_round < self.fedgkd_start_round:
            if current_round == 0 and self.fedgkd_start_round > 0:
                print(f"FedGKD: Buffer filling phase - distillation will start at round {self.fedgkd_start_round}")
            return
        
        # Log when distillation starts
        if current_round == self.fedgkd_start_round:
            print(f"FedGKD: Starting distillation at round {current_round}")
            
        for client_id in selected_clients:
            if client_id in self.clients:
                self.clients[client_id].receive_fedgkd_teacher(
                    self.fedgkd_ensemble_teacher, 
                    self.fedgkd_distillation_coeff, 
                    self.fedgkd_temperature
                )
    
    def compute_fedgkd_vote_weights(self):
        """Compute performance-based weights using cosine distance weights for FedGKD-VOTE"""
        if self.fedgkd_avg_param or not self.historical_cdw_weights:
            return  # Not needed for standard FedGKD or if no CDW data
        
        weights = []
        
        # Use cosine distance weights from historical rounds
        # Higher CDW = better client alignment = higher weight for that model
        for i, model in enumerate(self.fedgkd_models_buffer):
            if i < len(self.historical_cdw_weights):
                # Use average CDW from when this model was the global model
                cdw_values = self.historical_cdw_weights[i]
                if cdw_values:  # Check if CDW data exists
                    avg_cdw = sum(cdw_values) / len(cdw_values)
                    weight = avg_cdw  # Higher CDW = higher weight
                else:
                    weight = 1.0  # Default weight if no CDW data
            else:
                weight = 1.0  # Default weight for models without CDW history
            
            weights.append(weight)
        
        # Apply exponential emphasis on better models and ensure positive weights
        weights = [max(0.01, w) for w in weights]  # Ensure minimum positive weight
        weights = [w ** 2 for w in weights]  # Square to amplify differences
        
        # Normalize weights
        total_weight = sum(weights)
        self.fedgkd_model_weights = [w / total_weight for w in weights]
        
        print(f"FedGKD-VOTE weights computed: {[f'{w:.3f}' for w in self.fedgkd_model_weights]}")


    def train(self, epoch, cdw, use_cuda,round):
        models = []
        loss = []
        cos_distance_weights = []
        data_sizes = []
        current_client_list = random.sample(self.client_list, self.num_of_clients)
        
        if self.fedgkd_enabled and self.fedgkd_ensemble_teacher is not None:
            self.send_fedgkd_teacher_to_clients(current_client_list, round)
        
        print(f"\n=== SERVER TRAINING ROUND {round} ===")
        
        for i in current_client_list:
            self.clients[i].train(self.federated_model, use_cuda,round)
            cos_distance_weights.append(self.clients[i].get_cos_distance_weight())
            loss.append(self.clients[i].get_train_loss())
            models.append(self.clients[i].get_model())
            data_sizes.append(self.clients[i].get_data_sizes())
            
        if epoch==0:
            self.L0 = torch.Tensor(loss) 

        avg_loss = sum(loss) / self.num_of_clients

        print("==============================")
        print("number of clients used:", len(models))
        print('Train Epoch: {}, AVG Train Loss among clients of lost epoch: {:.6f}'.format(epoch, avg_loss))
        
        self.train_loss.append(avg_loss)
        
        weights = data_sizes
        
        if cdw:
            weights = cos_distance_weights

        
        self.federated_model = aggregate_models(models, weights)
        
        # FedGKD: Store CDW weights for FedGKD-VOTE and update buffer
        if self.fedgkd_enabled:
            if cdw and cos_distance_weights:
                cdw_values = [w.cpu().item() if hasattr(w, 'cpu') else float(w) for w in cos_distance_weights]
                self.historical_cdw_weights.append(cdw_values)
                # same buffer length as models
                if len(self.historical_cdw_weights) > self.fedgkd_buffer_length:
                    self.historical_cdw_weights.pop(0)
            
            self.update_fedgkd_buffer(self.federated_model)
            
            # Compute new weights for FedGKD-VOTE
            if not self.fedgkd_avg_param:
                self.compute_fedgkd_vote_weights()
        
    def update_kd_learning_rate(self, round_num, decay_factor=0.98):
        """Update knowledge distillation learning rate with gentle exponential decay"""
        if round_num < 30:
            self.current_kd_lr = self.initial_kd_lr  # Keep initial rate early
        else:
            self.current_kd_lr = self.initial_kd_lr * (decay_factor ** (round_num - 30))
        if round_num % 10 == 0:  # Print every 10 rounds
            print(f"Server KD LR update - Round {round_num}: KD LR = {self.current_kd_lr:.8f}")
    
    def get_adaptive_temperature(self, round_num):
        """Get adaptive temperature for knowledge distillation"""
        initial_temp = 4.0
        min_temp = 2.0
        decay_factor = 0.95
        
        # Temperature annealing: start high, decay to minimum
        current_temp = max(min_temp, initial_temp * (decay_factor ** (round_num / 10)))
        return current_temp

    def draw_curve(self):
        plt.figure()
        x_epoch = list(range(len(self.train_loss)))
        plt.plot(x_epoch, self.train_loss, 'bo-', label='train')
        plt.legend()
        dir_name = os.path.join(self.project_dir, 'model', self.experiment_name)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        plt.savefig(os.path.join(dir_name, 'train.png'))
        plt.close('all')
        
    def test(self, use_cuda,):
        """
        Test the global federated model on each client's individual query/gallery split and global test set.
        This evaluates how well the global model generalizes to each client's data and the global test set.
        """
        print("="*10)
        print("Start Testing Global Model on Each Client Dataset!")
        print("="*10)
        print('We use the scale: %s'%self.multiple_scale)
        
        # Test global model on each client's dataset
        for dataset in self.data.datasets:
            if dataset == '-1': continue  # Skip global test set, focus on client-specific datasets
            
            print(f"Evaluating global model on client {dataset} dataset")
            self.federated_model = self.federated_model.eval()
            if use_cuda:
                self.federated_model = self.federated_model.cuda()
                
            test_loaders = self.data.test_loaders[dataset]
            with torch.no_grad():
                gallery_feature = extract_feature(self.federated_model, test_loaders['gallery'], self.multiple_scale, self.data.image_size)
                query_feature = extract_feature(self.federated_model, test_loaders['query'], self.multiple_scale, self.data.image_size)

            result = {
                'gallery_f': gallery_feature.numpy(),
                'gallery_label': self.data.gallery_meta[dataset]['labels'],
                'query_f': query_feature.numpy(),
                'query_label': self.data.query_meta[dataset]['labels'],
            }

            # Save results for each client in separate directories
            client_result_dir = os.path.join(self.project_dir, 'model', self.experiment_name)
            os.makedirs(client_result_dir, exist_ok=True)
            
            scipy.io.savemat(os.path.join(client_result_dir, 'pytorch_result.mat'), result)
            
            print(f"Global model evaluation on client {dataset}: {self.model_name}")

            output_file = f'global_result.csv'
            cmd = f'python evaluate.py --result_dir {client_result_dir} --dataset {dataset} --output_file {output_file}'
            os.system(cmd)
        

    def knowledge_distillation(self, regularization, round = 0):
        import torch.nn.functional as F
        
        # Adaptive temperature annealing
        temperature = self.get_adaptive_temperature(round)
        print(f"Round {round}: Using temperature = {temperature:.4f}")
        
        optimizer = optim.SGD(self.federated_model.parameters(), lr=self.current_kd_lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
        self.federated_model.train().to(self.device)

        # Get feature dimension dynamically from model using actual image size
        with torch.no_grad():
            self.federated_model.eval()
            dummy_input = torch.randn(1, 3, self.data.image_size, self.data.image_size).to(self.device)
            dummy_output = self.federated_model(dummy_input)
            feature_dim = dummy_output.shape[1]

        for _, (x, target) in enumerate(self.data.kd_loader): 
            x, target = x.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            soft_target = torch.zeros(len(x), feature_dim).to(self.device)
        
            # Generate soft labels from client models
            for i in self.client_list:
                self.clients[i].model = self.clients[i].model.to(self.device)
                
                # Keep heads removed for feature-level distillation
                
                i_label = (self.clients[i].generate_soft_label(x, regularization, temperature))
                soft_target += i_label
                    
            soft_target /= len(self.client_list)
        
            self.federated_model.train()

            student_features = self.federated_model(x)
            student_soft = F.log_softmax(student_features / temperature, dim=1)
            teacher_soft = F.softmax(soft_target / temperature, dim=1)

            loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
            loss.backward()
            optimizer.step()
            self.update_kd_learning_rate(round)
            print("train_loss_fine_tuning", loss.data)