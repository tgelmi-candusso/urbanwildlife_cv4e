'''
    2022 Benjamin Kellenberger, Tiziana Gelmi-Candusso
'''

import os
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, RandomRotation, RandomCrop, GaussianBlur, ToTensor, RandomApply, Normalize
from PIL import Image
import pandas as pd



class CTDataset(Dataset):

    def __init__(self, cfg, split='train', split_type = 'split_by_loc', max_num=-1):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        self.split_type = split_type
        self.max_num = max_num
        self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            Resize((cfg['image_size'])),  # For now, we just resize the images to the same dimensions...
            #RandomRotation(degrees=cfg['image_rotation']), #random rotation with a rango of angles between -45 and 45 with a 10 angle interval
            #RandomApply(transforms = [RandomCrop(224, 50)], p=0.15),
            #RandomApply(transforms = [GaussianBlur(kernel_size= (51), sigma = (1,2))], p=0.05),
            #nop-RandomGrayscale(), #some pictures on grayscale #this was good for birds not small mammals
            #iaa.Sometimes(0.25, )
            #functional.adjust_hue(image,hue_factor=0.3)
            #GaussianBlur(kernel_size=(51, 91), sigma=2), #blur some images with a sigma of 1 and 3
            ToTensor(),
            #Normalize(mean = [0.485, 0.456, 0.406], std =  [0.229, 0.224, 0.225])  #normalize to speed up computations
                          # ...and convert them to torch.Tensor.
        ]) 
        
        # index data into dict
        data_dict = {}
        #dict categories
        ##the following will need to be transformed into a in script mapping dictionay when adding more classes for UWIN
        cat_csv = pd.read_csv(os.path.join(self.data_root, 'categories.csv')) #this could go into the cfg file
        species_idx = cat_csv['class'].to_list()
        species = cat_csv['description'].to_list()
        self.species_to_index_mapping = dict(zip(species, species_idx))

        #load the train file
        f = open(os.path.join(self.data_root, self.split_type.lower(), self.split.lower()+'.txt'), 'r') 
        lines = f.readlines() # load all lines

        for line in lines: # loop over lines
            file_name = line.strip().replace("\\", "/")
            sp = os.path.split(file_name)[0]
            # print(file_name)
            # print(os.path.split(file_name))
            
            # if not, add it and assign an index
            species_idx = self.species_to_index_mapping[sp]
            if species_idx not in data_dict:
                data_dict[species_idx] = []
            data_dict[species_idx].append(file_name)
        
        # subsample if needed
        self.data = []
        for species in data_dict.keys():
            species_list = [[i, species] for i in data_dict[species]]
            if max_num > 0:
                random.shuffle(species_list)
                species_list = species_list[:min(len(species_list), max_num)]
            self.data.extend(species_list)


    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_path, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list

        # load image
        image_path = os.path.join(self.data_root, 'crops', image_path)
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)

        return img_tensor, label, image_path

# %%
