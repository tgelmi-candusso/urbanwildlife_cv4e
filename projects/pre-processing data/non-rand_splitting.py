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
def split(training_folder, output_folder, sample_percent, num_images_max=None, split_by_location=True):
    ###random splitting, not recommended for setup, correct splitting later on
    ## create random subset to move on

    ## load coco file with information per image
    with open(coco_path) as f:
        coco_data = json.load(f)

    # writers for train/val/test txt files
    if split_by_location = True:
        output_subfolder = os.join.path(output_folder, 'split_by_loc')
        os.makedirs(output_subfolder, exist_ok=True)
        write_train = open(os.path.join(output_subfolder, 'train.txt'), mode = 'w')
        write_val = open(os.path.join(output_subfolder, 'val.txt'), mode = 'w')
        write_test = open(os.path.join(output_subfolder, 'test.txt'), mode = 'w')
    else:
        output_subfolder = os.join.path(output_folder, 'split_across_loc')
        os.makedirs(output_subfolder, exist_ok=True)
        write_train = open(os.path.join(output_subfolder, 'train.txt'), mode = 'w')
        write_val = open(os.path.join(output_subfolder, 'val.txt'), mode = 'w')
        write_test = open(os.path.join(output_subfolder, 'test.txt'), mode = 'w')

    ##create dictionary for station and datetime per path
    species_folders = os.listdir(training_folder)
    for idx, sp in enumerate(species_folders):
        print(f'[{idx+1}/{len(species_folders)}] {sp}')
        directory = os.path.join(training_folder, sp)
        if not os.path.isdir(directory):
            continue
        files_inside_s = os.listdir(directory)

        im_dic = {}
        for u in coco_data['images']:
            key = u['file_name']
            value = [u['station'], u['datetime']]
            if key not in files_inside_s:
                continue
            im_dic[key] = value


        map = {}
        for u in coco_data['images']:
            key = u['file_name']
            value = u['image_id']
            if key not in files_inside_s:
                continue
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

        #create dictionary of categories loc+daytime
        im_dic_cat = {}
        for item in non_red_dic:
            split_category = non_red_dic[item][0] + "." + non_red_dic[item][2]
            if split_category not in im_dic_cat:
                im_dic_cat[split_category] = []
            im_dic_cat[split_category].append(item)
    
        # split into train/val/test
        if split_by_location:
            keys = list(im_dic_cat.keys())
            random.shuffle(keys)
            train = int(len(keys)*sample_percent[0])
            val = int(len(keys)*sample_percent[1])
            #test = int(len(keys)*sample_percent[2])

            train_samples = keys[0:train]
            val_samples = keys[train+1:train+val]
            test_samples = keys[train+val+1:]     # all the rest for test
         
            for sample in train_samples:
                images = im_dic_cat[sample]
                write_train.write('\n'.join([os.path.join(sp, i) for i in images]))
                write_train.write('\n')
                # write_train.write(sample + '\n')
            for sample in val_samples:
                images = im_dic_cat[sample]
                write_val.write('\n'.join([os.path.join(sp, i) for i in images]))
                write_val.write('\n')
            for sample in test_samples:
                images = im_dic_cat[sample]
                write_test.write('\n'.join([os.path.join(sp, i) for i in images]))
                write_test.write('\n')

        else:
            for key in im_dic_cat.keys():
                imgs = im_dic_cat[key]
                train = int(len(imgs)*sample_percent[0])
                val = int(len(imgs)*sample_percent[1])

                train_samples = imgs[0:train]
                val_samples = imgs[train+1:train+val]
                test_samples = imgs[train+val+1:]   
                
                write_train.write('\n'.join([os.path.join(sp, i) for i in train_samples]))
                write_train.write('\n')

                write_val.write('\n'.join([os.path.join(sp, i) for i in val_samples]))
                write_val.write('\n')

                write_test.write('\n'.join([os.path.join(sp, i) for i in test_samples]))
                write_test.write('\n')

    write_train.close()
    write_val.close()
    write_test.close()

    
    
## extract files from coco file if training a detector which we wont
#    images_out = []
#    annotations_out = []
#    categories_out = coco_data['categories']

#    for i in coco_data['images']:
#        if i['image_id'] in ids:
#            images_out.append(i)

#    for i in coco_data['annotations']:
#        if i['image_id'] in ids:
#            annotations_out.append(i)

#    train_data = {
#        "images": images_out,
#        "annotations": annotations_out,
#        "categories": categories_out
#    }

#main issue now is mapping the json file with the dictionary



split(training_folder, output_folder, sample_percent, num_images_max, split_by_location=True)
split(training_folder, output_folder, sample_percent, num_images_max, split_by_location=False)

# %%
