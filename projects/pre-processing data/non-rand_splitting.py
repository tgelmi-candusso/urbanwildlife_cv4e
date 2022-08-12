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
 
    #reduce dictionary containing file paths as keys and [location, datetime, daytime] as values
    non_red_dic = {k: im_dic[k] for k in reduced_files}

    #create dictionary of file paths with image_id
    names2 = list(non_red_dic.keys())
    non_red_map = {k: map[k] for k in names2}
    #list of ids (dkw \o/)
    ids = [non_red_map[i] for i in non_red_map]

    im_dic_cat = {}
    for item in non_red_dic:
        split_category = non_red_dic[item][0] + "." + non_red_dic[item][2]
        if split_category not in im_dic_cat:
            im_dic_cat[split_category] = []
        im_dic_cat[split_category].append(item)
    
    for sp in os.listdir(training_folder):
        directory = os.path.join(training_folder, sp)
    if not os.path.isdir(directory):
        continue
    
    ## im here
    files_inside_s = os.listdir(directory)
    for files in files_inside_s:
        if file not in im_dic_cat
    
    
    if num_images_max is not None or num_images_max > len(files_inside_s):
    random.shuffle(files_inside_s)

    
        # perform random subsampling
        files_inside_s = files_inside_s[0:num_images_max]

    for f in range(len(files_inside_s)):
        files_inside_s[f] = os.path.join(sp, files_inside_s[f])

    train = int(len(files_inside_s)*sample_percent[0])
    val = int(len(files_inside_s)*sample_percent[1])
    # test = int(len(files_inside_s)*sample_percent[2])

    train_samples = files_inside_s[0:train]
    val_samples = files_inside_s[train+1:train+val]
    test_samples = files_inside_s[train+val+1:]     # all the rest for test


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
