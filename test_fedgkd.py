#!/usr/bin/env python3
"""
Test script for FedGKD implementation
This script validates the FedGKD functionality without running full training
"""

import torch
import torch.nn as nn
import sys
import os
import argparse
from utils import get_model
from server import Server
from client import Client
from data_utils import Data

def test_fedgkd_components():
    """Test FedGKD components individually"""
    print("Testing FedGKD Components")
    print("="*50)
    
    # Create dummy arguments
    class DummyArgs:
        def __init__(self):
            self.datasets = '0,1,2'
            self.data_dir = '/home/wellvw12/client_data_non_iid'
            self.batch_size = 32
            self.erasing_p = 0
            self.color_jitter = False
            self.train_all = False
            self.project_dir = '.'
            self.model_name = 'test_fedgkd'
            self.num_of_clients = 2
            self.lr = 0.001
            self.drop_rate = 0.5
            self.stride = 2
            self.multiple_scale = '1'
            self.ex_name = 'test_fedgkd'
            self.model = 'resnet18_ft_net'
            self.kd_lr_ratio = 0.05
            self.local_epoch = 1
            self.cosine_annealing = True
            self.total_rounds = 10
            self.eta_min = 1e-6
    
    args = DummyArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Test 1: Data loading
        print("Test 1: Data Loading")
        data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, args.train_all)
        data.preprocess()
        print(f"✓ Data loaded successfully. Client list: {data.client_list}")
        
        # Test 2: Client creation
        print("\nTest 2: Client Creation")
        clients = {}
        for cid in data.client_list[:2]:  # Test with first 2 clients
            clients[cid] = Client(
                cid, data, device, args.project_dir, args.model_name, 
                args.local_epoch, args.lr, args.batch_size, args.drop_rate, 
                args.stride, args.ex_name, args.model, args.cosine_annealing,
                args.total_rounds, args.eta_min
            )
        print(f"✓ Created {len(clients)} clients")
        
        # Test 3: Server creation
        print("\nTest 3: Server Creation")
        server = Server(
            clients, data, device, args.project_dir, args.model_name,
            args.num_of_clients, args.lr, args.drop_rate, args.stride,
            args.multiple_scale, args.ex_name, args.model, args.kd_lr_ratio
        )
        print("✓ Server created successfully")
        
        # Test 4: FedGKD configuration
        print("\nTest 4: FedGKD Configuration")
        server.configure_fedgkd(buffer_length=3, distillation_coeff=0.1, temperature=2.0, avg_param=True)
        print(f"✓ FedGKD configured: enabled={server.fedgkd_enabled}")
        
        # Test 4b: FedGKD-VOTE configuration
        print("\nTest 4b: FedGKD-VOTE Configuration")
        server.configure_fedgkd(buffer_length=3, distillation_coeff=0.1, temperature=2.0, avg_param=False)
        print(f"✓ FedGKD-VOTE configured: avg_param={server.fedgkd_avg_param}")
        
        # Test 5: Model buffer operations
        print("\nTest 5: Model Buffer Operations")
        dummy_model = get_model(100, 0.5, 2, 'resnet18_ft_net').to(device)
        dummy_model.classifier = nn.Sequential()  # Remove classifier like in federated training
        
        # Add models to buffer
        for i in range(4):
            # Create slightly different models
            model_copy = get_model(100, 0.5, 2, 'resnet18_ft_net').to(device)
            model_copy.classifier = nn.Sequential()
            # Add some randomness to distinguish models
            with torch.no_grad():
                for param in model_copy.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            
            server.update_fedgkd_buffer(model_copy)
            
        print(f"✓ Buffer operations successful. Buffer size: {len(server.fedgkd_models_buffer)}")
        
        # Test 6: Ensemble teacher creation
        print("\nTest 6: Ensemble Teacher Creation")
        teacher = server.ensemble_historical_models()
        print(f"✓ Ensemble teacher created: {teacher is not None}")
        
        # Test 7: Client-teacher communication
        print("\nTest 7: Client-Teacher Communication")
        test_client = list(clients.values())[0]
        test_client.receive_fedgkd_teacher(teacher, 0.1, 2.0)
        print(f"✓ Client received teacher: enabled={test_client.fedgkd_enabled}")
        
        # Test 8: Distillation loss computation
        print("\nTest 8: Distillation Loss Computation")
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        dummy_features = torch.randn(2, 512).to(device)  # Feature size to match model output
        
        distill_loss = test_client.compute_fedgkd_distillation_loss(dummy_features, dummy_input)
        print(f"✓ Distillation loss computed: {distill_loss}")
        
        # Test 9: FedGKD-VOTE weight computation
        print("\nTest 9: FedGKD-VOTE Weight Computation")
        # Simulate CDW weights for testing
        server.historical_cdw_weights = [[0.8, 0.7, 0.9], [0.85, 0.75, 0.95], [0.82, 0.78, 0.88]]
        server.fedgkd_avg_param = False  # Enable VOTE mode
        server.compute_fedgkd_vote_weights()
        print(f"✓ FedGKD-VOTE weights computed: {len(server.fedgkd_model_weights)} weights")
        
        print("\n" + "="*50)
        print("All FedGKD tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fedgkd_integration():
    """Test FedGKD integration with main training loop"""
    print("\nTesting FedGKD Integration")
    print("="*50)
    
    # Test command line arguments
    test_args = [
        '--fedgkd',
        '--fedgkd_buffer_length', '3',
        '--fedgkd_distillation_coeff', '0.2',
        '--fedgkd_temperature', '2.0',
        '--fedgkd_avg_param',
        '--datasets', '0,1',
        '--num_of_clients', '2',
        '--total_rounds', '5',
        '--local_epoch', '1'
    ]
    
    print("Example command to run FedGKD:")
    print(f"python main.py {' '.join(test_args)}")
    
    # Test argument parsing
    from main import parser
    try:
        args = parser.parse_args(test_args)
        print(f"✓ Arguments parsed successfully")
        print(f"  FedGKD enabled: {args.fedgkd}")
        print(f"  Buffer length: {args.fedgkd_buffer_length}")
        print(f"  Distillation coeff: {args.fedgkd_distillation_coeff}")
        print(f"  Temperature: {args.fedgkd_temperature}")
        print(f"  Average param: {args.fedgkd_avg_param}")
        
    except Exception as e:
        print(f"✗ Argument parsing failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("FedGKD Implementation Test Suite")
    print("="*50)
    
    # Test individual components
    component_test = test_fedgkd_components()
    
    # Test integration
    integration_test = test_fedgkd_integration()
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Component Tests: {'PASSED' if component_test else 'FAILED'}")
    print(f"Integration Tests: {'PASSED' if integration_test else 'FAILED'}")
    
    if component_test and integration_test:
        print("\n🎉 All tests passed! FedGKD implementation is ready.")
        print("\nUsage examples:")
        print("1. Basic FedGKD:")
        print("   python main.py --fedgkd --datasets 0,1,2")
        print("2. FedGKD with custom parameters:")
        print("   python main.py --fedgkd --fedgkd_buffer_length 7 --fedgkd_distillation_coeff 0.2")
        print("3. FedGKD-VOTE variant:")
        print("   python main.py --fedgkd --fedgkd_temperature 4.0")
    else:
        print("\n❌ Some tests failed. Check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main()