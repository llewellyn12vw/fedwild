import torch
import numpy as np
from typing import Dict, List, Optional

class ModelDebugger:
    """Utility class for debugging federated learning models"""
    
    def __init__(self):
        self.model_states = {}
        self.round_history = []
    
    def log_model_state(self, model, label: str, round_num: int = None):
        """Log model state with detailed statistics"""
        state = self.get_model_statistics(model)
        state['label'] = label
        state['round'] = round_num
        
        key = f"{label}_{round_num}" if round_num is not None else label
        self.model_states[key] = state
        
        print(f"Model State [{label}]: {state}")
        return state
    
    def get_model_statistics(self, model) -> Dict:
        """Get comprehensive model statistics"""
        stats = {}
        
        # Parameter statistics
        params = list(model.parameters())
        stats['param_count'] = sum(p.numel() for p in params)
        stats['param_sum'] = sum(p.sum().item() for p in params)
        stats['param_mean'] = stats['param_sum'] / stats['param_count']
        
        # Layer-wise statistics
        layer_stats = {}
        for name, param in model.named_parameters():
            layer_stats[name] = {
                'shape': list(param.shape),
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item(),
                'norm': param.norm().item()
            }
        stats['layer_stats'] = layer_stats
        
        # Gradient statistics (if available)
        grad_stats = {}
        has_grads = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grads = True
                grad_stats[name] = {
                    'grad_norm': param.grad.norm().item(),
                    'grad_mean': param.grad.mean().item(),
                    'grad_std': param.grad.std().item()
                }
        
        if has_grads:
            stats['gradient_stats'] = grad_stats
            stats['total_grad_norm'] = sum(stats['gradient_stats'][name]['grad_norm'] for name in grad_stats)
        
        return stats
    
    def compare_models(self, model1, model2, label1: str = "Model1", label2: str = "Model2"):
        """Compare two models and return differences"""
        stats1 = self.get_model_statistics(model1)
        stats2 = self.get_model_statistics(model2)
        
        comparison = {
            'param_count_diff': stats2['param_count'] - stats1['param_count'],
            'param_sum_diff': stats2['param_sum'] - stats1['param_sum'],
            'param_mean_diff': stats2['param_mean'] - stats1['param_mean']
        }
        
        # Layer-wise comparison
        layer_diffs = {}
        for name in stats1['layer_stats']:
            if name in stats2['layer_stats']:
                layer_diffs[name] = {
                    'mean_diff': stats2['layer_stats'][name]['mean'] - stats1['layer_stats'][name]['mean'],
                    'norm_diff': stats2['layer_stats'][name]['norm'] - stats1['layer_stats'][name]['norm']
                }
        
        comparison['layer_differences'] = layer_diffs
        
        print(f"Model Comparison ({label1} vs {label2}):")
        print(f"  Parameter sum difference: {comparison['param_sum_diff']:.6f}")
        print(f"  Parameter mean difference: {comparison['param_mean_diff']:.6f}")
        
        return comparison
    
    def verify_aggregation(self, client_models: List, weights: List, aggregated_model):
        """Verify if aggregation was computed correctly"""
        print(f"\n=== AGGREGATION VERIFICATION ===")
        
        # Manual aggregation check
        if not client_models:
            print("No client models to verify!")
            return False
        
        # Get parameter names from first model
        param_names = [name for name, _ in client_models[0].named_parameters()]
        
        verification_passed = True
        for param_name in param_names:
            # Get parameters from all models
            client_params = []
            for model in client_models:
                param_dict = dict(model.named_parameters())
                if param_name in param_dict:
                    client_params.append(param_dict[param_name])
            
            if not client_params:
                continue
            
            # Manual weighted average
            manual_avg = torch.zeros_like(client_params[0])
            total_weight = sum(weights)
            
            for param, weight in zip(client_params, weights):
                manual_avg += param * (weight / total_weight)
            
            # Compare with aggregated model
            agg_param_dict = dict(aggregated_model.named_parameters())
            if param_name in agg_param_dict:
                agg_param = agg_param_dict[param_name]
                diff = torch.norm(manual_avg - agg_param).item()
                
                if diff > 1e-6:  # Tolerance for numerical errors
                    print(f"  {param_name}: MISMATCH (diff={diff:.8f})")
                    verification_passed = False
                else:
                    print(f"  {param_name}: OK (diff={diff:.8f})")
        
        if verification_passed:
            print("✓ Aggregation verification PASSED")
        else:
            print("✗ Aggregation verification FAILED")
        
        return verification_passed
    
    def log_training_round(self, round_num: int, client_losses: List, client_weights: List, 
                          aggregated_loss: float, federated_model):
        """Log comprehensive training round information"""
        round_info = {
            'round': round_num,
            'client_losses': client_losses,
            'client_weights': client_weights,
            'aggregated_loss': aggregated_loss,
            'federated_model_state': self.get_model_statistics(federated_model)
        }
        
        self.round_history.append(round_info)
        
        print(f"\n=== ROUND {round_num} SUMMARY ===")
        print(f"Client losses: {[f'{loss:.4f}' for loss in client_losses]}")
        print(f"Client weights: {[f'{weight:.4f}' for weight in client_weights]}")
        print(f"Aggregated loss: {aggregated_loss:.4f}")
        print(f"Federated model param sum: {round_info['federated_model_state']['param_sum']:.6f}")
        
        return round_info
    
    def detect_issues(self, model, threshold_grad_norm: float = 10.0, 
                     threshold_param_change: float = 1000.0):
        """Detect potential training issues"""
        issues = []
        stats = self.get_model_statistics(model)
        
        # Check for exploding gradients
        if 'total_grad_norm' in stats and stats['total_grad_norm'] > threshold_grad_norm:
            issues.append(f"Exploding gradients detected: {stats['total_grad_norm']:.2f}")
        
        # Check for NaN or infinite values
        for name, layer_stat in stats['layer_stats'].items():
            if not np.isfinite(layer_stat['mean']) or not np.isfinite(layer_stat['std']):
                issues.append(f"NaN/Inf values in layer {name}")
        
        # Check for dead neurons (all zeros)
        for name, layer_stat in stats['layer_stats'].items():
            if layer_stat['std'] < 1e-8:
                issues.append(f"Potential dead neurons in layer {name} (std={layer_stat['std']:.2e})")
        
        if issues:
            print(f"⚠️  Issues detected:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✓ No issues detected")
        
        return issues
    
    def export_debug_log(self, filename: str):
        """Export debug information to file"""
        import json
        
        debug_data = {
            'model_states': self.model_states,
            'round_history': self.round_history
        }
        
        # Convert numpy/torch values to regular Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        debug_data = convert_for_json(debug_data)
        
        with open(filename, 'w') as f:
            json.dump(debug_data, f, indent=2)
        
        print(f"Debug log exported to {filename}")

# Global debugger instance
debugger = ModelDebugger()