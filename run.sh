#!/bin/bash

# Run the commands sequentially
# python main.py --model_name ex2_110

# python main.py --ex_name exReg && \
# python main.py --kd --regularization --model_name exKD && \
python main.py --cdw --kd --regularization --ex_name exKD_CDW
# python main.py --kd --regularization --model_name ex2_110_kd && \
# python main.py --cdw --model_name ex2_110_cdw



gsutil -m cp -r /home/wellvw12/leopard gs://leopard567wild/