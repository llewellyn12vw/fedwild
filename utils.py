import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import ft_net
from torch.autograd import Variable

from torchvision import datasets, transforms

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed**2)
    torch.manual_seed(seed**3)
    torch.cuda.manual_seed(seed**4)

def get_optimizer(model, lr):
    # Check if model has arcface_head (MegaDescriptor) or classifier (ResNet)
    if hasattr(model, 'arcface_head') and hasattr(model.arcface_head, 'parameters'):
        ignored_params = list(map(id, model.arcface_head.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.1*lr},
                {'params': model.arcface_head.parameters(), 'lr': lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    elif hasattr(model, 'classifier'):
        ignored_params = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.1*lr},
                {'params': model.classifier.parameters(), 'lr': lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        # Fallback: optimize all parameters with same learning rate
        optimizer_ft = optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
    
    return optimizer_ft

def save_network(network, cid, epoch_label, project_dir, name, gpu_ids):
    save_filename = 'net_%s.pth'% epoch_label
    dir_name = os.path.join(project_dir, 'model', name, cid)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    save_path = os.path.join(project_dir, 'model', name, cid, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])

def get_model(class_sizes, drop_rate, stride):
    model = ft_net(class_sizes, drop_rate, stride)
    return model

# functions for testing federated model
def fliplr(img):
    """flip horizontal
    """
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model, dataloaders, ms):
    features = torch.FloatTensor()
    
    # Get feature dimension dynamically from model
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 256, 128).cuda()
        dummy_output = model(dummy_input)
        feature_dim = dummy_output.shape[1]
    
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, feature_dim).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features



