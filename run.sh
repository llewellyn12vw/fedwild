#!/bin/bash

# Test fixed FedGKD implementation with better parameters

python main.py --ex_name lepoes  --cdw \
    --local_epoch 2 \
    --num_of_clients 3 \
    --total_rounds 100 \
    --metadata_file /home/wellvw12/fedwild/federated_leopards/metadata.csv \
    --dataset_type leopard \

# python main.py --ex_name macaque/Ex1Reg/3  --cdw \
#     --local_epoch 3 \
#     --num_of_clients 3 \
#     --total_rounds 100 \
#     --metadata_file /home/wellvw12/fedwild/federated_clients_enhanced/metadata.csv \
#     --dataset_type macaque \
    
# python main.py --fedgkd --ex_name macaque/ExFEDGDK/5 --cdw \
#     --fedgkd_distillation_coeff 0.1 \
#     --fedgkd_temperature 0.5 \
#     --fedgkd_buffer_length 10 \
#     --local_epoch 2 \
#     --num_of_clients 3 \
#     --total_rounds 100 \
#     --metadata_file /home/wellvw12/fedwild/federated_clients_enhanced/metadata.csv \
#     --dataset_type macaque \
#     --fedgkd_avg_param \
#     --fedgkd_start_round 15

# python main.py --fedgkd --ex_name macaque/ExFEDGDK/6  --cdw \
#     --fedgkd_distillation_coeff 0.1 \
#     --fedgkd_temperature 0.5 \
#     --fedgkd_buffer_length 10 \
#     --local_epoch 3 \
#     --num_of_clients 3 \
#     --total_rounds 100 \
#     --metadata_file /home/wellvw12/fedwild/federated_clients_enhanced/metadata.csv \
#     --dataset_type macaque \
#     --fedgkd_avg_param \
#     --fedgkd_start_round 15


# python main.py --fedgkd --ex_name macaque/ExFEDGDK/7  --cdw \
#     --fedgkd_distillation_coeff 0.1 \
#     --fedgkd_temperature 1.0 \
#     --fedgkd_buffer_length 15 \
#     --local_epoch 2 \
#     --num_of_clients 3 \
#     --total_rounds 100 \
#     --metadata_file /home/wellvw12/fedwild/federated_clients_enhanced/metadata.csv \
#     --dataset_type macaque \
#     --fedgkd_avg_param \
#     --fedgkd_start_round 10




# python main.py --fedgkd --ex_name macaque/ExFEDGDK/8  --cdw \
#     --fedgkd_distillation_coeff 0.2 \
#     --fedgkd_temperature 2.0 \
#     --fedgkd_buffer_length 10 \
#     --local_epoch 2 \
#     --num_of_clients 3 \
#     --total_rounds 100 \
#     --metadata_file /home/wellvw12/fedwild/federated_clients_enhanced/metadata.csv \
#     --dataset_type macaque \
#     --fedgkd_avg_param \
#     --fedgkd_start_round 20

# python main.py --fedgkd --ex_name macaque/ExFEDGDK/9  --cdw \
#     --fedgkd_distillation_coeff 0.4 \
#     --fedgkd_temperature 2.0 \
#     --fedgkd_buffer_length 10 \
#     --local_epoch 2 \
#     --num_of_clients 3 \
#     --total_rounds 100 \
#     --metadata_file /home/wellvw12/fedwild/federated_clients_enhanced/metadata.csv \
#     --dataset_type macaque \
#     --fedgkd_avg_param \
#     --fedgkd_start_round 20
 

# python main.py --fedgkd --ex_name baseline800.30KD/2 --data_dir /home/wellvw12/baselines/baseline800.30  --cdw \
#     --fedgkd_distillation_coeff 0.9 \
#     --fedgkd_temperature 4.0 \
#     --fedgkd_buffer_length 5 \
#     --local_epoch 1 \
#     --num_of_clients 2 \
#     --total_rounds 50 \
#     --fedgkd_avg_param

# python main.py --fedgkd --ex_name baseline800.30KD/3 --data_dir /home/wellvw12/baselines/baseline800.30  --cdw \
#     --fedgkd_distillation_coeff 0.9 \
#     --fedgkd_temperature 2.0 \
#     --fedgkd_buffer_length 5 \
#     --local_epoch 1 \
#     --num_of_clients 2 \
#     --total_rounds 50 \
#     --fedgkd_avg_param

# python main.py --fedgkd --ex_name baseline800.30KD/4 --data_dir /home/wellvw12/baselines/baseline800.30  --cdw \
#     --fedgkd_distillation_coeff 0.5 \
#     --fedgkd_temperature 1.0 \
#     --fedgkd_buffer_length 5 \
#     --local_epoch 1 \
#     --num_of_clients 2 \
#     --total_rounds 50 \
#     --fedgkd_avg_param

# python main.py --fedgkd --ex_name lepKD2 \
#     --fedgkd_distillation_coeff 0.1 \
#     --fedgkd_temperature 2.0 \
#     --fedgkd_buffer_length 5 \
#     --local_epoch 1 \
#     --num_of_clients 3 \
#     --total_rounds 50 \

