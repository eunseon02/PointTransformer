import math
import torch

class config:
    
    # maxiumum number of points per voxel
    T=3

    # voxel size
    vd = 0.05
    vh = 0.05
    vw = 0.05

    # points cloud range
    xrange = (-8, 8)
    yrange = (-8, 8)
    zrange = (-3,3)

    # voxel grid
    W = math.ceil((xrange[1] - xrange[0]) / vw)
    H = math.ceil((yrange[1] - yrange[0]) / vh)
    D = math.ceil((zrange[1] - zrange[0]) / vd)



    m = 16
    nPlanes = [m, 2*m, 3*m, 4*m] 
    dimension = 3

    ## train
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



