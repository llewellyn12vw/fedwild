import torch 
import torch.nn.functional as F

class Optimization():
    def __init__(self, train_loader, device):
        self.train_loader = train_loader
        self.device = device

    def cdw_feature_distance(self, old_model, new_model):
        """cosine distance weight (cdw): calculate feature distance of 
           the features of a batch of data by cosine distance.
        """
        old_model=old_model.to(self.device)
        new_model=new_model.to(self.device)
        
        for data in self.train_loader:
            inputs, _ = data
            inputs=inputs.to(self.device)

            with torch.no_grad():
                # Get feature representations (before classifier) from both models
                old_backbone_features = old_model.backbone(inputs)
                old_features = old_model.feature_head(old_backbone_features)
                
                new_backbone_features = new_model.backbone(inputs)
                new_features = new_model.feature_head(new_backbone_features)

            distance = 1 - torch.cosine_similarity(old_features, new_features)
            return torch.mean(distance)

    def kd_generate_soft_label(self, model, data, regularization, temperature=4.0):
        """knowledge distillation (kd): generate soft labels.
        """
        model = model.to(data.device)
        result = model(data)
        if regularization:
            result = F.normalize(result, dim=1, p=2)
        # Apply temperature scaling for softer distributions
        result = result / temperature
        return result
