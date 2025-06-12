import math
import torch
from os.path import join
from torch.utils.tensorboard import SummaryWriter

class config:
    # maxiumum number of points per voxel
    # T = 3

    # # voxel size
    # vd = 0.05
    # vh = 0.05
    # vw = 0.05

    # # points cloud range
    # xrange = (-8, 8)
    # yrange = (-8, 8)
    # zrange = (-3, 3)

    # # voxel grid
    # W = math.ceil((xrange[1] - xrange[0]) / vw)
    # H = math.ceil((yrange[1] - yrange[0]) / vh)
    # D = math.ceil((zrange[1] - zrange[0]) / vd)

    # m = 16
    # nPlanes = [m, 2 * m, 3 * m, 4 * m]
    # dimension = 3

    # Voxelize
    vsize_xyz = [0.05, 0.05, 0.05]
    coors_range_xyz = [-10, -5, 0, 10, 5, 10]
    # input_shape = (50, 120, 120, 2)
    voxel_size = torch.tensor([0.05, 0.05, 0.05])

    # Point cloud features
    num_point_features = 4  
    max_num_points_per_voxel = 3  

    # Train configurations
    is_train = True
    batch_size = 16 if is_train else 1
    iter = 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BASE_LOGDIR = f"./train_logs{iter}"
    writer = SummaryWriter(join(BASE_LOGDIR, "occu"))
    file = "final_dataset_16.h5" if is_train else "sample_dataset_0.h5"
    weight = f"weight{iter}"
    log = f'train_log{iter}.txt'
    wandb_log_dir = 'logs'

    start_epoch =  270 if is_train else 0
    epochs = 300

    # Training and testing parameters
    debug_epoch = 10 if is_train else 1  
    occu_cutoff = 0.5 if is_train else 0.5
    teacher_forcing_ratio = 1.0 if is_train else 0.0 
    epochs = 300 if is_train else 1
    decay_rate= 0.1 if is_train else 0.0
    teacher_forcing_ratio = max(0.0,teacher_forcing_ratio - ((start_epoch // 20) * decay_rate))  if start_epoch != 0 else teacher_forcing_ratio

    
    
        
        