#!/bin/bash
# Job name:
#SBATCH --job-name=train_1
#
# Account:
#SBATCH --account=fc_ferroic
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=jagar@berkeley.edu
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:K80:1
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
## Command(s) to run (example):
python Autoencoder_Savio.py
