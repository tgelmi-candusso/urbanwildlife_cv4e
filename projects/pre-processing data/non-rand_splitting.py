# %%
import os
import random
from tkinter import W
from datetime import datetime
from datetime import timedelta
import numpy as np
import json
from functools import reduce
#from sklearn.model_selection import train_test_split

# %%
output_folder = r'D:\animals_training_dataset'
training_folder = r'D:\animals_training_dataset\crops'
sample_percent = [0.6, 0.1, 0.3]
num_images_max = 1000
coco_path = 'C:/Users/tizge/Documents/GitHub/urbanwildlife_cv4e/datasets/TUW/training_dataset.json'


# %%
def split(training_folder, output_folder, sample_percent, num_images_max=None):
    ###random splitting, not recommended for setup, correct splitting later on
    ## create random subset to move on

    ## load coco file with information per image
    with open(coco_path) as f:
        coco_data = json.load(f)

    ##create dictionary for station and datetime per path
    im_dic = {}
    for u in coco_data['images']:
        key = u['file_path']
        value = [u['station'], u['datetime']]
        im_dic[key] = value

    map = {}
    for u in coco_data['images']:
        key = u['file_path']
        value = u['image_id']
        map[key] = value
    
    ## extract whether time is day or night
    for t in im_dic:
        time = datetime.strptime(im_dic[t][1], " %Y-%m-%d %H:%M:%S")
        hr, mi = (time.hour, time.minute)
        if hr>=7 and hr<18: 
            daytime = "day"
        else: 
            daytime = "night"
        im_dic[t].append(daytime)

    ### reduce redundant images:

    names = np.array(list(im_dic.keys()))
    times = np.array([im_dic[i][1] for i in im_dic])
    times = times.astype(datetime)

    ordered_indices = np.argsort(times)

    sorted_times = times[ordered_indices]
    sorted_names = names[ordered_indices]

    sorted_name_time_pairs = [(t,n) for t, n in zip(sorted_names, sorted_times)]
    reduced_files = [t0[0] for (t0,t1) in zip(sorted_name_time_pairs, sorted_name_time_pairs[1:]) if datetime.strptime(t1[1], " %Y-%m-%d %H:%M:%S") - datetime.strptime(t0[1], " %Y-%m-%d %H:%M:%S") > timedelta(seconds=5)]
 
    non_red_dic = {k: im_dic[k] for k in reduced_files}


    names2 = list(non_red_dic.keys())
    non_red_map = {k: map[k] for k in names2}
    ids = [non_red_map[i] for i in non_red_map]



## extract files from coco file
    images_out = []
    annotations_out = []
    categories_out = coco_data['categories']

    for i in coco_data['images']:
        if i['image_id'] in ids:
            images_out.append(i)

    for i in coco_data['annotations']:
        if i['image_id'] in ids:
            annotations_out.append(i)

    train_data = {
        "images": images_out,
        "annotations": annotations_out,
        "categories": categories_out
    }

#main issue now is mapping the json file with the dictionary



split(training_folder, output_folder, sample_percent, num_images_max)

# %%
