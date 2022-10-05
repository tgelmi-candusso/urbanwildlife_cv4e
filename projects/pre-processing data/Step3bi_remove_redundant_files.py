import os
import json
from datetime import datetime
from datetime import timedelta
import numpy as np
from tqdm import tqdm

path = r'/datadrive/animals_training_dataset/crops/'
coco_path = r'/home/azureuser/urbanwildlife_cv4e/projects/pre-processing data/training_dataset.json'

with open(coco_path) as f:
        coco_data = json.load(f)

species_folders = os.listdir(path)
for idx, sp in enumerate(species_folders):
        #print(f'[{idx+1}/{len(species_folders)}] {sp}')
        directory = os.path.join(path, sp)
        if not os.path.isdir(directory):
            pass
        if sp == 'Cropped-Multicity':
            pass
        files_inside_s = os.listdir(directory)
        print(sp, ':', len(files_inside_s))
        #### trying to make a list of redundant files

        im_dic = {}
        for u in coco_data['images']:
            key = u['file_name']
            value = [u['station'], u['datetime']]
            if key not in files_inside_s:
                continue
                
            im_dic[key] = value

        ##dont really need this map for this
        # map = {}
        # for u in coco_data['images']:
        #     key = u['file_name']
        #     value = u['image_id']
        #     if key not in files_inside_s:
        #         continue
        #     map[key] = value

        # ##dont realy need this day or night
        # ## extract whether time is day or night
        # for t in im_dic:
        #     time = datetime.strptime(im_dic[t][1], " %Y-%m-%d %H:%M:%S")
        #     hr, mi = (time.hour, time.minute)
        #     if hr>=7 and hr<18: 
        #         daytime = "day"
        #     else: 
        #         daytime = "night"
        #     im_dic[t].append(daytime)

        ### reduce redundant images:

        names = np.array(list(im_dic.keys()))
        times = np.array([im_dic[i][1] for i in im_dic])
        times = times.astype(datetime)

        ordered_indices = np.argsort(times)

        sorted_times = times[ordered_indices]
        sorted_names = names[ordered_indices]

        sorted_name_time_pairs = [(t,n) for t, n in zip(sorted_names, sorted_times)]
        redundant_files = [t0[0] for (t0,t1) in zip(sorted_name_time_pairs, sorted_name_time_pairs[1:]) if datetime.strptime(t1[1], " %Y-%m-%d %H:%M:%S") - datetime.strptime(t0[1], " %Y-%m-%d %H:%M:%S") < timedelta(seconds=5)]
    
        #reduce dictionary containing file paths as keys and [location, datetime, daytime] as values
        #redundant_dic = {k: im_dic[k] for k in redundant_files1} ##these are REDUNDANT files, ie all files that had less the 5seconds with the previous file

        #create dictionary of file paths with image_id
        #redundant_files = list(redundant_dic.keys()) ##this is the list I need to remove redundant files
        
        # ##dont really need these two lines next:
        # red_map = {k: map[k] for k in redundant_files}
        # #list of ids (dkw \o/)
        # ids = [red_map[i] for i in red_map]

        ###just to double check, check the ones that remain, yup it worked
        non_redundant_files = [t0[0] for (t0,t1) in zip(sorted_name_time_pairs, sorted_name_time_pairs[1:]) if datetime.strptime(t1[1], " %Y-%m-%d %H:%M:%S") - datetime.strptime(t0[1], " %Y-%m-%d %H:%M:%S") > timedelta(seconds=5)]
        # non_redundant_dic = {k: im_dic[k] for k in non_redundant_files} ##these are REDUNDANT files, ie all files that had less the 5seconds with the previous file
        # non_redundant_files = list(non_redundant_dic.keys())

        ###end of list of redundant files

        to_delete = []
        to_keep = []
        for file in tqdm(files_inside_s):
            #print(f'[{idx+1}/{len(files_inside_s)}]')
            # construct full file path
            file1 = os.path.join(directory,file)
            if file not in directory:
                pass
            if file not in redundant_files:
                #print(file, 'NOT')
                to_keep.append(file)
                pass
            else:
                #print(file, 'YES')
                to_delete.append(file)
                #os.remove(file1)  ##to remove files uncomment
        print('keeping:', len(to_keep))
        print('deleting:', len(to_delete))

        #update object
        files_inside_s1 = os.listdir(directory)
        print('filenumber_before:', len(files_inside_s), 'filenumber_now:', len(files_inside_s1))
