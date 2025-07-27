import torch

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
# import pretrainedmodels

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

# Define the ResNet18-based Model (original ft_net)
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net, self).__init__()
       
        model_ft = models.resnet18(pretrained=True)
        # model_ft=torch.load('saved_res50.pkl')
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].conv1.stride = (1,1)  # ResNet-18 specific
            model_ft.layer4[0].downsample[0].stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(512, class_num, droprate)

    def backbone(self, x):
        """Extract backbone features (before classifier)"""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        return x

    def feature_head(self, x):
        """Identity function for compatibility - backbone already outputs features"""
        return x

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


# Define the ResNet50-based Model with ArcFace Loss
class resnet50_arcface(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, margin=0.5, scale=64):
        super(resnet50_arcface, self).__init__()
       
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].conv1.stride = (1,1)
            model_ft.layer4[0].downsample[0].stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        
        # Feature dimension for ResNet50 is 2048
        self.feature_dim = 2048
        
        # Add feature normalization layer
        self.bn = nn.BatchNorm1d(self.feature_dim)
        self.bn.apply(weights_init_kaiming)
        
        # ArcFace head
        self.arcface_head = ArcFaceHead(self.feature_dim, class_num, margin=margin, scale=scale)

    def backbone(self, x):
        """Extract backbone features (before classifier)"""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        return x

    def feature_head(self, x):
        """Apply feature normalization"""
        x = self.bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.feature_head(x)
        result = self.arcface_head(x, labels)
        return result


'''
# debug model structure
# Run this code with:
python model.py
'''

import timm
import sys
sys.path.append('/home/wellvw12/wildlife-tools')
from wildlife_tools.train.objective import ArcFaceLoss

class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, margin=0.5, scale=64):
        super(ArcFaceHead, self).__init__()
        self.arcface = ArcFaceLoss(
            num_classes=out_features, 
            embedding_size=in_features, 
            margin=margin, 
            scale=scale
        )
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, features, labels=None):
        if labels is not None:
            # Training mode: return both logits and loss
            logits = self.classifier(features)
            loss = self.arcface(features, labels)
            return logits, loss
        else:
            # Inference mode: return only logits
            return self.classifier(features)


# Define the DINOv2-based Model with ArcFace Loss
class dinov2_arcface(nn.Module):
    
    def __init__(self, class_num, droprate=0.5, margin=0.5, scale=64, model_size='small'):
        super(dinov2_arcface, self).__init__()
        
        # Load DINOv2 model
        if model_size == 'small':
            self.backbone_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.feature_dim = 384
        elif model_size == 'base':
            self.backbone_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.feature_dim = 768
        elif model_size == 'large':
            self.backbone_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.feature_dim = 1024
        elif model_size == 'giant':
            self.backbone_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            self.feature_dim = 1536
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        # Freeze backbone (optional - can be made configurable)
        for param in self.backbone_model.parameters():
            param.requires_grad = False
        
        # Add dropout if specified
        self.dropout = nn.Dropout(droprate) if droprate > 0 else nn.Identity()
        
        # Add feature normalization layer
        self.bn = nn.BatchNorm1d(self.feature_dim)
        self.bn.apply(weights_init_kaiming)
        
        # ArcFace head
        self.arcface_head = ArcFaceHead(self.feature_dim, class_num, margin=margin, scale=scale)
    
    def backbone(self, x):
        """Extract backbone features from DINOv2"""
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.backbone_model(x)
        return features
    
    def feature_head(self, x):
        """Apply dropout, normalization, and L2 normalization"""
        x = self.dropout(x)
        x = self.bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.feature_head(x)
        result = self.arcface_head(x, labels)
        return result


# Define the MegaDescriptor-based Model
class megadescriptor_ft_net(nn.Module):
    
    def __init__(self, class_num, droprate=0.5):
        super(megadescriptor_ft_net, self).__init__()
        
        # Load MegaDescriptor backbone
        self.backbone_model = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=False)
        self.feature_dim = self.backbone_model.num_features
        
        # Add classifier with feature extraction capability
        self.classifier = ClassBlock(self.feature_dim, class_num, droprate)
    
    def backbone(self, x):
        """Extract backbone features from MegaDescriptor"""
        return self.backbone_model(x)
    
    def feature_head(self, x):
        """Identity function for compatibility - backbone already outputs features"""
        return x
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def save_model(model, class_num, save_path, device='cuda'):
    """
    Compiles and saves the ft_net model with proper state dict handling.
    
    Args:
        model: The ft_net model instance
        class_num: Number of classes for the final layer
        save_path: Path to save the model (.pth or .pt)
        device: Target device ('cuda' or 'cpu')
    """
    # Ensure model is on right device
    model = model.to(device)
    
    # Create a complete state dict (including classifier)
    state_dict = {
        'model_state_dict': model.state_dict(),
        'class_num': class_num,
        'model_architecture': 'resnet50_ft_net'
    }
    
    # Save with proper serialization
    try:
        torch.save(state_dict, save_path)
        print(f"Model successfully saved to {save_path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = ft_net(751, stride=1)
    net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')

    save_model(net, 
            class_num=751, 
            save_path='resnet18_ft_net.pth',
            device='cuda')
    print(output.shape)
