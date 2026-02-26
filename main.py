from utils import *
from pdb import set_trace as st
from matplotlib import pyplot as plt
import numpy as np
import argparse
from torch.utils.data import DataLoader
from sklearn import metrics
import pandas as pd
import csv

def pinhole_range_est_exp(dataset,max_dist=60,inc=10,predicted=False):
    """
    ## Pinhole Range Estimation RMSE. Total average, and per distance bin.
    """
    err_abs = []
    RMSE_per_bin = {key+inc: [] for key in range(0,max_dist,inc)}
    for sample in dataset:
        img, distance, ann, full_ann, scene, name = sample
        bb = ann['bbox']   
        bin = int((distance//inc)*inc+inc)

        pinhole_dist = pinhole_distance_estimate(bb, scene)
        err_abs.append(abs(pinhole_dist-distance))

        RMSE_per_bin[bin].append(abs(pinhole_dist-distance))
        
        
    print("Total RMSE: ",np.sqrt(np.mean(err_abs)))
    for key, value in RMSE_per_bin.items():
        if len(value) == 0: continue
        print(f"RMSE for bin {str(key)} is {np.mean(value)}\n")