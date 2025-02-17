"""
Modified from DLStudio/Examples/playing_with_cifar10.py
"""
import traceback
import argparse
from UDF import *
import torch.nn as nn
import torch.nn.functional as F
from trace import start_tracing, stop_tracing

# USER ADDED: Add DLStudio to the path:
import os, sys
sys.path.append('/usr/local/lib/python3.10/dist-packages/DLStudio-2.5.1-py3.10.egg')

import random
import numpy
import torch
import os

"""
seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)
"""

## watch -d -n 0.5 nvidia-smi
from DLStudio import DLStudio
from DLStudio import *

dls = DLStudio(
    # dataroot = "/home/kak/ImageDatasets/CIFAR-10/",
    dataroot = "./data/CIFAR-10/",
    image_size = [32, 32],
    path_saved_model = "./saved_model",
    momentum = 0.9,
    learning_rate = 1e-3,
    epochs = 2,
    batch_size = 4,
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    # use_gpu = True,
)

# Inherit DLStudio.ExperimentsWithCIFAR to extend to NetGivenTest
class CustomExperimentsWithCIFAR(DLStudio.ExperimentsWithCIFAR):
    class NetGivenTest(nn.Module):  # Define a new model
        """ This class extends DLStudio.ExperimentsWithCIFAR to add a new model NetGivenTest.
        Args:
            DLStudio.ExperimentsWithCIFAR: 
        """
        def __init__(self):
            print(f"{CustomExperimentsWithCIFAR.__doc__}")
            super(CustomExperimentsWithCIFAR.NetGivenTest, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 4 * 4, 256)
            self.fc2 = nn.Linear(256, 10)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
        
    class Net3(nn.Module):  # Define a new model
        """ This class extends DLStudio.ExperimentsWithCIFAR to add a new model Net3.
        Args:
            DLStudio.ExperimentsWithCIFAR: 
        """
        def __init__(self):
            print(f"{CustomExperimentsWithCIFAR.__doc__}")
            super(CustomExperimentsWithCIFAR.NetGivenTest, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 4 * 4, 256)
            self.fc2 = nn.Linear(256, 10)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Create directories if specified:
    parser.add_argument('--create_dirs', action='store_true', help="Flag to create data and output directories")
    parser.add_argument('--data_dir', default='./data', help="Directory containing data")
    parser.add_argument('--output_dir', default='./output', help="Directory to save output")
    
    parser.add_argument('--task', "-t", type=int, choices=[1, 2, 3], default=1, help="Pick a task to run")
    parser.add_argument('--train', type=bool, default=False, help="Load training dataset")
    parser.add_argument('--test', type=bool, default=False, help="Load testing dataset")
    parser.add_argument('--given', '-g', action='store_true', help="Use the given code from playing_with_cifar10.py and the HW5 pdf")
    
    args = parser.parse_args()
    
    # Logic for handling directory creation only if --create_dirs is provided
    if args.create_dirs:
        if args.data_dir:
            print(f"Creating directory: {args.data_dir}")
            ensure_directory_exists(args.data_dir)
        if args.output_dir:
            print(f"Creating directory: {args.output_dir}")
            ensure_directory_exists(args.output_dir)

    load_testing_dataset_setting = False
    load_training_dataset_setting = False
    categories = ["clock", "bird", "airplane", "train", "giraffe"]

    # Execute the given code from playing_with_cifar10.py and the HW5 pdf:
    if args.given:
        # Instantiate the new custom experiments class
        exp_cifar = CustomExperimentsWithCIFAR(dl_studio=dls)
        # Load CIFAR-10 dataset
        exp_cifar.load_cifar_10_dataset()
        # Instantiate the new model
        model = exp_cifar.NetGivenTest()  # Use the new NetGivenTest model
        # Display network properties
        number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
        exp_cifar.run_code_for_training(model, display_images=False)
        exp_cifar.run_code_for_testing(model, display_images=False)

    if args.task == 1:
        print(f"Custom Dataset_1/")
        # Instantiate the new custom experiments class
        exp_cifar = CustomExperimentsWithCIFAR(dl_studio=dls)
        # Load CIFAR-10 dataset
        exp_cifar.load_cifar_10_dataset()
        # Instantiate the new model
        model = exp_cifar.Net3()  # Use the new NetGivenTest model
        # Display network properties
        number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
        exp_cifar.run_code_for_training(model, display_images=False)
        exp_cifar.run_code_for_testing(model, display_images=False)

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
