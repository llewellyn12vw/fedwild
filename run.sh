#!/bin/bash

# Test fixed FedGKD implementation with better parameters



# python main.py --ex_name macaque_help/fedgdk/1 \
#     --local_epoch 1 \
#     --num_of_clients 5 \
#     --total_rounds 100 \
#     --metadata_file /home/wellvw12/fedwild/macaque_help/metadata.csv \
#     --dataset_type macaques \

python main.py --fedgkd --ex_name macaque_help/fedgdk/2 \
    --fedgkd_distillation_coeff 0.1 \
    --fedgkd_buffer_length 4 \
    --local_epoch 1 \
    --num_of_clients 3 \
    --total_rounds 100 \
    --metadata_file /home/wellvw12/fedwild/macaque_help/metadata.csv \
    --dataset_type macaque \
    --fedgkd_avg_param \
    --fedgkd_start_round 0 \


