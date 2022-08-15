# 
import os
import random
from tkinter import W

output_folder = r'D:\animals_training_dataset'
training_folder = r'D:\animals_training_dataset\crops'
sample_percent = [0.6, 0.1, 0.3]
num_images_max = 1000


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
        random.shuffle(files_inside_s)

        if num_images_max is not None or num_images_max > len(files_inside_s):
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
