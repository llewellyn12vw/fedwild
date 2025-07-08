import os
import math
import json
import matplotlib.pyplot as plt
from utils import get_model, extract_feature
import torch.nn as nn
import torch
import scipy.io
import copy
from data_utils import ImageDataset
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
    def __init__(self, clients, data, device, project_dir, model_name, num_of_clients, lr, drop_rate, stride, multiple_scale,experiment_name,model):
        self.project_dir = project_dir
        self.data = data
        self.device = device
        self.model_name = model_name
        self.clients = clients
        self.client_list = self.data.client_list
        self.num_of_clients = num_of_clients
        self.lr = lr
        self.multiple_scale = multiple_scale
        self.drop_rate = drop_rate
        self.stride = stride
        self.experiment_name = experiment_name
        self.model = model

        self.multiple_scale = []
        for s in multiple_scale.split(','):
            self.multiple_scale.append(math.sqrt(float(s)))

        self.full_model = get_model(750, drop_rate, stride, model).to(device)
        
        # Handle different model types
        if model == 'megadescriptor':
            self.full_model.arcface_head = nn.Sequential()
        else:  # resnet18_ft_net
            self.full_model.classifier = nn.Sequential()
            
        self.federated_model=self.full_model
        self.federated_model.eval()
        self.train_loss = []


    def train(self, epoch, cdw, use_cuda,round):
        models = []
        loss = []
        cos_distance_weights = []
        data_sizes = []
        current_client_list = random.sample(self.client_list, self.num_of_clients)
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
        print()
        
        self.train_loss.append(avg_loss)
        
        weights = data_sizes
        
        if cdw:
            print("cos distance weights:", cos_distance_weights)
            weights = cos_distance_weights

        self.federated_model = aggregate_models(models, weights)

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
        print("="*10)
        print("Start Tesing!")
        print("="*10)
        print('We use the scale: %s'%self.multiple_scale)
        
        for dataset in self.data.datasets:
            self.federated_model = self.federated_model.eval()
            if use_cuda:
                self.federated_model = self.federated_model.cuda()
            
            with torch.no_grad():
                gallery_feature = extract_feature(self.federated_model, self.data.test_loaders[dataset]['gallery'], self.multiple_scale)
                query_feature = extract_feature(self.federated_model, self.data.test_loaders[dataset]['query'], self.multiple_scale)

            result = {
                'gallery_f': gallery_feature.numpy(),
                'gallery_label': self.data.gallery_meta[dataset]['labels'],
                # 'gallery_cam': self.data.gallery_meta[dataset]['cameras'],
                'query_f': query_feature.numpy(),
                'query_label': self.data.query_meta[dataset]['labels'],
                # 'query_cam': self.data.query_meta[dataset]['cameras']
                }

            scipy.io.savemat(os.path.join(self.project_dir,
                        'model',
                        self.experiment_name,
                        'pytorch_result.mat'),
                        result)
                        
            print(self.model_name)
            print(dataset)

            os.system('python evaluate.py --result_dir {} --dataset {} --output_file {}'.format(os.path.join(self.project_dir, 'model', self.experiment_name), dataset, 'aggregated_result.csv'))


    def knowledge_distillation(self, regularization, temperature=3.0):
        import torch.nn.functional as F
        optimizer = optim.SGD(self.federated_model.parameters(), lr=self.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
        self.federated_model.train().to(self.device)

        # Get feature dimension dynamically from model
        with torch.no_grad():
            self.federated_model.eval()
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_output = self.federated_model(dummy_input)
            feature_dim = dummy_output.shape[1]

        for _, (x, target) in enumerate(self.data.kd_loader): 
            x, target = x.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            soft_target = torch.zeros(len(x), feature_dim).to(self.device)
        
            # Generate soft labels from client models
            for i in self.client_list:
                self.clients[i].model = self.clients[i].model.to(self.device)
                
                # Restore client-specific head for soft label generation
                if self.model == 'megadescriptor':
                    self.clients[i].model.arcface_head = self.clients[i].arcface_head
                else:  # resnet18_ft_net
                    self.clients[i].model.classifier = self.clients[i].classifier
                
                i_label = (self.clients[i].generate_soft_label(x, regularization, temperature))
                soft_target += i_label
                
                # Remove head after generation
                if self.model == 'megadescriptor':
                    self.clients[i].model.arcface_head = nn.Sequential()
                else:  # resnet18_ft_net
                    self.clients[i].model.classifier = nn.Sequential()
                    
            soft_target /= len(self.client_list)
        
            # Generate student output based on model type
            self.federated_model.train()
            if self.model == 'megadescriptor':
                # For MegaDescriptor, use feature-level distillation
                student_features = self.federated_model(x)
                # Apply temperature scaling to feature similarity
                student_soft = F.log_softmax(student_features / temperature, dim=1)
                teacher_soft = F.softmax(soft_target / temperature, dim=1)
            else:  # resnet18_ft_net
                # For ResNet, can use logit-level distillation if needed
                student_logits = self.federated_model(x)
                student_soft = F.log_softmax(student_logits / temperature, dim=1)
                teacher_soft = F.softmax(soft_target / temperature, dim=1)
            
            # Use KL divergence loss for proper knowledge distillation
            loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
            loss.backward()
            optimizer.step()
            print("train_loss_fine_tuning", loss.data)