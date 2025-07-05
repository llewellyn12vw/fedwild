import torch 
import torch.nn.functional as F

class Optimization():
    def __init__(self, train_loader, device):
        self.train_loader = train_loader
        self.device = device

    def cdw_feature_distance(self, old_model, old_classifier, new_model):
        """cosine distance weight (cdw): calculate feature distance of 
           the features of a batch of data by cosine distance.
        """
        old_model=old_model.to(self.device)
        new_model=new_model.to(self.device)
        
        # Handle different model architectures
        if hasattr(old_classifier, 'parameters'):
            old_classifier = old_classifier.to(self.device)
        
        for data in self.train_loader:
            inputs, _ = data
            inputs=inputs.to(self.device)

            with torch.no_grad():
                # Get features from both models
                old_features = old_model(inputs)
                new_features = new_model(inputs)
                
                # For MegaDescriptor vs ResNet comparison, use feature-level similarity
                # Normalize features to same dimension space if needed
                if old_features.shape[1] != new_features.shape[1]:
                    # If dimensions don't match, use a simple distance metric
                    # based on feature magnitudes (L2 norms)
                    old_norm = torch.norm(old_features, dim=1)
                    new_norm = torch.norm(new_features, dim=1)
                    distance = torch.abs(old_norm - new_norm) / (old_norm + new_norm + 1e-8)
                    return torch.mean(distance)
                else:
                    # Same dimensions, use cosine similarity
                    distance = 1 - torch.cosine_similarity(old_features, new_features)
                    return torch.mean(distance)

    def kd_generate_soft_label(self, model, data, regularization):
        """knowledge distillation (kd): generate soft labels.
        """
        model = model.to(data.device)
        result = model(data)
        if regularization:
            result = F.normalize(result, dim=1, p=2)
        return result
