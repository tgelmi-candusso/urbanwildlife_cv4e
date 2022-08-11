from dataset import CTDataset 
import os
import argparse
import yaml
import glob
from tqdm import trange
import pandas as pd

    # hard code config
cfg = yaml.safe_load(open('/home/azureuser/snow-Dayz/configs/exp_resnet50_2classes.yaml'))
labels = 'trainLabels.csv'
folder = 'train'
split = 'train'

dataset_instance = CTDataset(labels, cfg, folder, split)

dataset_instance.__len__()