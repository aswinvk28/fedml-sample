#!/bin/bash
#PBS -l nodes=2:xeon:ppn=2
#PBS -l walltime=24:00:00
#PBS -e error.txt
#PBS -o output.txt

cd /home/u93525/Documents/federated-learning-csv/FedML-master/fedml_experiments/standalone/fedavg

# wandb login e1fd0e2fbed65d1f83a7b27f681b6b0be9911ed7

/home/u93525/.conda/envs/fedml/bin/python main_fedavg.py --dataset nepal_earthquake --model tabnet --data_dir "../../../fedml_api/data_preprocessing/nepal_earthquake" --client_num_in_total 3 --client_num_per_round 2 --gpu 1

