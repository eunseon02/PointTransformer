import os
import torch
import numpy as np
from plyfile import PlyData
import pandas as pd
import glob
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class GetTarget(Dataset):
    def __init__(self, root_dir):
        self.file_paths = file_pathsbatch_dirs = sorted(os.listdir(root_dir))
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                occupancy_grids = pickle.load(f)
            print("File loaded successfully.")
        else:
            print(f"File '{file_path}' does not exist.")
            
        return occupancy_grids, file_path