'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html
    We just hack our way through the COCO JSON files here for demonstration
    purposes.
    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data
    2022 Benjamin Kellenberger
'''

import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


class CTDataset(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            Resize((cfg['image_size'])),        # For now, we just resize the images to the same dimensions...
            ToTensor()                          # ...and convert them to torch.Tensor.
        ])
        
        # index data into list
        self.data = []
        #dict categories
        self.species_to_index_mapping = dict()

        #load the train file
        f = open(os.path.join(self.data_root, self.split.lower()+'.txt'), 'r') 
        lines = f.readlines() # load all lines
        for line in lines: # loop over lines
            file_name = line.strip()
            species, _ = os.path.split(file_name)
            
           #TODO: check if species is in self.species_to_index_mapping
            if species not in self.species_to_index_mapping[species]:
                continue
           # if not, add it and assign an index
            species_idx = self.species_to_index_mapping[species]
            self.data.append([file_name, species_idx])


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

        return img_tensor, label
