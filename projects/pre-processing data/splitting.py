# 
import os
import random
from tkinter import W
import datetime
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

output_folder = '/datadrive/animals_training_dataset/random_split3'
training_folder = '/datadrive/animals_training_dataset/crops'
sample_percent = [0.6, 0.1, 0.3]
num_images_max = 100000


def split(training_folder, output_folder, sample_percent, num_images_max=None):
    ###random splitting, not recommended for setup, correct splitting later on
    ## create random subset to move on
    
    output_subfolder = os.path.join(output_folder, 'split_random')
    os.makedirs(output_subfolder, exist_ok=True)
    write_train = open(os.path.join(output_subfolder, 'train.txt'), mode = 'w')
    write_val = open(os.path.join(output_subfolder, 'val.txt'), mode = 'w')
    write_test = open(os.path.join(output_subfolder, 'test.txt'), mode = 'w')

    species_folders = os.listdir(training_folder)
    for idx, sp in enumerate(species_folders):
        print(f'[{idx+1}/{len(species_folders)}] {sp}')
        directory = os.path.join(training_folder, sp)
        if not os.path.isdir(directory):
            continue

        files_inside_s = os.listdir(directory)

        filename = []
        filetime = []
        for i in files_inside_s:
            filename.append(i)
            filepath = os.path.join(directory, i)
            filetime_ts = os.path.getmtime(filepath) #not giving me the original date
            #filetime_ts = os.stat(filepath).st_birthtime ##this didnt work
            #image = Image.open(filepath) ##tried exif but didnt manage to get to the datetime
            #exif = {}
            #for tag, value in image.getexif().items():
            #    if tag in TAGS:
            #        exif[TAGS[tag]] = value
            #print(exif)
            filetime.append(filetime_ts)

        times = np.array(filetime)
        names = np.array(filename)
        #times = datetime.datetime.strptime(filetime, "%Y-%m-%d %H:%M:%S")
        #times = filetime.astype(datetime)
        ordered_indices = np.argsort(times)
        sorted_times = times[ordered_indices]
        sorted_names = names[ordered_indices]
        sorted_name_time_pairs = [(t,n) for t, n in zip(sorted_names, sorted_times)]
        reduced_files = [t0[0] for (t0,t1) in zip(sorted_name_time_pairs, sorted_name_time_pairs[1:]) if (t1[1] - t0[1]) > 5]
        non_red_dic = [files_inside_s[k] for k in reduced_files]

        files_inside_s1 = random.shuffle(non_red_dic)

        if num_images_max is not None or num_images_max > len(files_inside_s1):
            # perform random subsampling
            files_inside_s1 = files_inside_s1[0:num_images_max]

        for f in range(len(files_inside_s1)):
            files_inside_s1[f] = os.path.join(sp, files_inside_s1[f])

        train = int(len(files_inside_s1)*sample_percent[0])
        val = int(len(files_inside_s1)*sample_percent[1])
        # test = int(len(files_inside_s1)*sample_percent[2])

        train_samples = files_inside_s1[0:train]
        val_samples = files_inside_s1[train+1:train+val]
        test_samples = files_inside_s1[train+val+1:]     # all the rest for test

        for sample in train_samples:
            write_train.write(sample + '\n')
        for sample in val_samples:
            write_val.write(sample + '\n')
        for sample in test_samples:
            write_test.write(sample + '\n')

    write_train.close()
    write_val.close()
    write_test.close()


split(training_folder, output_folder, sample_percent, num_images_max)
