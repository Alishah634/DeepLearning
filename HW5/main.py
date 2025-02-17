# Import my personal utility functions:
from UDF import ensure_directory_exists 

import argparse
import matplotlib.pyplot as plt
import os
from PIL import Image
from pycocotools.coco import COCO
import random
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt
from tqdm import tqdm

# DL Studio Import:
#DL_Studio_path = os.path.abspath("Masters/Spring2025/ECE20146/DLStudio-2.5.1.tar.gz")
DL_Studio_path = os.path.abspath("ECE60146/DLStudio-2.5.1.tar.gz")
print(f"{DL_Studio_path}")

# Ensure that data directory and output directory are absolute paths
data_dir = os.path.abspath("Dataset_COCO/")

# Track which images are the in the training and which are the testing set:
training_dataset = set()
testing_dataset = set()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Create directories if specified:
    parser.add_argument('--create_dirs', action='store_true', help="Flag to create data and output directories")
    parser.add_argument('--data_dir', default='./data', help="Directory containing data")
    parser.add_argument('--output_dir', default='./output', help="Directory to save output")
    
    parser.add_argument('--task', "-t", type=int, choices=[1, 2, 3], default=1, help="Pick a task to run")
    parser.add_argument('--train', type=bool, default=False, help="Load training dataset")
    parser.add_argument('--test', type=bool, default=False, help="Load testing dataset")
    
    args = parser.parse_args()
    
    # Logic for handling directory creation only if --create_dirs is provided
    if args.create_dirs:
        if args.data_dir:
            print(f"Creating directory: {args.data_dir}")
            ensure_directory_exists(args.data_dir)
        if args.output_dir:
            print(f"Creating directory: {args.output_dir}")
            ensure_directory_exists(args.output_dir)
        
    args = parser.parse_args()
    load_testing_dataset_setting = False
    load_training_dataset_setting = False
    categories = ["clock", "bird", "airplane", "train", "giraffe"]

    if args.task == 1:
        print(f"Custom Dataset_1/")
        # task1(dataset_num=1)
        
        print(f"Custom Dataset_2/")
        # task1(dataset_num=2)
        print("Task 1 completed")
        pass
    
    if args.task == 2:
        # task2()
        print("Task 2 completed")
        pass

    if args.task == 3:
        pass
    
    
