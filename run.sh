#!/bin/bash

# Run the commands sequentially
# python main.py --model_name ex2_110

python main.py --ex_name baseline_3.3.3/2
# python main.py --kd --regularization --model_name exKD && \
# python main.py --cdw --kd --regularization --ex_name exKD_CDW_hyn4
# python main.py --kd --regularization --model_name ex2_110_kd && \
# python main.py --cdw --model_name ex2_110_cdw


# python main.py --fedgkd --cdw --ex_name client_rarity_data9ß \
#     --fedgkd_distillation_coeff 0.5 \
#     --fedgkd_temperature 2.0 \
#     --local_epoch 1 \
#     --num_of_clients 4 \
#     --total_rounds 50 \
#     --fedgkd_avg_param