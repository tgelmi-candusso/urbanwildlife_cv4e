'''
    2022 Benjamin Kellenberger, Tiziana Gelmi-Candusso
'''

import os
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, RandomRotation, RandomCrop, ToTensor, RandomApply, Normalize
from PIL import Image
import pandas as pd



class CTDataset(Dataset):

    def __init__(self, cfg, split, split_type='random_split2' , max_num=-1):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        self.split_type = cfg['split_type']
        self.max_num = max_num
        self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            Resize((cfg['image_size'])),  # For now, we just resize the images to the same dimensions...
            #RandomRotation(degrees=cfg['image_rotation']), #random rotation with a rango of angles between -45 and 45 with a 10 angle interval
            #RandomApply(transforms = [RandomCrop(224, 50)], p=0.15),
            #RandomApply(transforms = [GaussianBlur(kernel_size= (51), sigma = (1,2))], p=0.01),
            #nop-RandomGrayscale(), #some pictures on grayscale #this was good for birds not small mammals
            #iaa.Sometimes(0.25, )
            #functional.adjust_hue(image,hue_factor=0.3)
            #GaussianBlur(kernel_size=(51, 91), sigma=2), #blur some images with a sigma of 1 and 3
            ToTensor(),
            #Normalize(mean = [0.485, 0.456, 0.406], std =  [0.229, 0.224, 0.225])  #normalize to speed up computations
                          # ...and convert them to torch.Tensor.
        ]) 
       
	
        # populate data_dict without using read_csv
        # should directly update data_dict from crops folder
        # get the name of the crops and the index
        # empty: 0, human: 1, vehicle: 2
        # then, for each crop in crops, get name : index
        #data_dict = {(0: 'empty'),

	# loop through crops
	# for each crop,
	# put name : index into datadict
	# print at end	
 
        # index data into dict
        species_to_index = {}
        
        species_to_index['empty'] = 0
        species_to_index['human'] = 1
        species_to_index['vehicle'] = 2
        
        # get crops information
        folder_dir = "/home/ykarandikar/crop-test-pipeline/crops"        
        #l = [x[0] for x in os.walk("~/crop-test-pipeline/crops/")]
        print("subfolders: ")
        subfolders = [ f.name for f in os.scandir(folder_dir) if f.is_dir() ]
        print(subfolders)




        # put subfolder names into the crops
        count = 3
        for name in subfolders:
                species_to_index[name] = count
                count += 1



        #dict categories
        ##the following will need to be transformed into a in script mapping dictionay when adding more classes for UWIN
        #cat_csv = pd.read_csv(os.path.join(self.data_root, 'categories.csv')) #this could go into the cfg file
        #species_idx = cat_csv['class'].to_list()
        #species = cat_csv['description'].to_list()
        self.species_to_index_mapping = species_to_index 
        print('species-index-mapping')
        print(self.species_to_index_mapping)


        data_dict = {}

        #load the train file
        print('split_type file path')
        print(os.path.join(self.data_root, self.split_type.lower(), self.split.lower()+'.txt'))
        f = open(os.path.join(self.data_root, self.split_type.lower(), self.split.lower()+'.txt'), 'r')
        lines = f.readlines() # load all lines
        for line in lines: # loop over lines
            # 'Coyote/19473_IMG_2079.JPG\n'
            sp = line.split('/')[0]
            file_name = line.strip().replace("\\", "/")
            #sp = os.path.split(file_name)[0]
            #print('sp')
            #print(sp)
            # print(os.path.split(file_name))
            #print(sp)
            # if not, add it and assign an index
            #print(data_dict)
            species_idx = self.species_to_index_mapping[sp]
            

            print("species_idx")

            print(species_idx)
            print(self.species_to_index_mapping[sp])
            if species_idx not in data_dict:
                data_dict[species_idx] = []
            data_dict[species_idx].append(file_name)
        #print('data_dict')
        #print(data_dict)
        # subsample if needed
        self.data = []
        for species in data_dict.keys():
            species_list = [[i, species] for i in data_dict[species]]
            if max_num > 0:
                random.shuffle(species_list)
                species_list = species_list[:min(len(species_list), max_num)]
            self.data.extend(species_list)
        
        print(data_dict)
        print(data_dict.keys())     

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
        #adding this to skip
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)

        return img_tensor, label, image_path

# %%
