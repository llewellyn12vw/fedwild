#!/bin/bash

# Run the commands sequentially
# python main.py --model_name ex2_110

# python main.py --ex_name baseline100.100_4perid/1 --data_dir /home/wellvw12/baselines/baseline100.100_4perid --cdw
python main.py --ex_name baseline800.30/2 --data_dir /home/wellvw12/baselines/baseline800.30  --cdw

# python main.py --kd --regularization --model_name exKD && \
# python main.py --cdw --kd --regularization --ex_name exKD_CDW_hyn4
# python main.py --kd --regularization --model_name ex2_110_kd && \
# python main.py --cdw --model_name ex2_110_cdw


# Test fixed FedGKD implementation with better parameters
# python main.py --fedgkd --ex_name baseline800.30KD/1 --data_dir /home/wellvw12/baselines/baseline800.30  --cdw \
#     --fedgkd_distillation_coeff 0.8 \
#     --fedgkd_temperature 3.0 \
#     --fedgkd_buffer_length 5 \
#     --local_epoch 1 \
#     --num_of_clients 2 \
#     --total_rounds 50 \
#     --fedgkd_avg_param

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

