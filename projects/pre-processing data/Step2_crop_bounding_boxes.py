 # %%

import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from tqdm import tqdm

# %%

### save paths

coco_path = 'C:/Users/tizge/Documents/GitHub/urbanwildlife_cv4e/datasets/TUW/training_dataset.json'
output_folder = r'D:\animals_training_dataset'
#output_folder = 'C:/Users/tizge/Documents/GitHub/urbanwildlife_cv4e/datasets/TUW/'
animals_folder = r'D:\animals'
#animals_folder = 'C:/Users/tizge/Documents/GitHub/urbanwildlife_cv4e/datasets/TUW/animals'

# %%
print(output_folder)
print(coco_path)
print(animals_folder)
# %%

def crop_images(coco_path, animals_folder, output_folder):
  
  #### create dictionary for images from json file derived from the megadetector/megaclassification
  with open(coco_path) as f:
    coco_data = json.load(f)

  im_dic = {}
  for u in tqdm(coco_data['images']):
    key = u['image_id']
    value = u['file_name']
    im_dic[key] = value

  #full_folder = os.path.join(output_folder, "full_bbox")
  #os.makedirs(full_folder, exist_ok=True)

  crops_folder = os.path.join(output_folder, "crops")
  os.makedirs(crops_folder, exist_ok=True)  
  
  ###loop through the target species list
  for sp in tqdm(os.listdir(animals_folder)):
    #test = 0
    print('cropping files of species ' + sp)
    #if sp != "coyote":
    #  break

    #path_spfull = os.path.join(full_folder, str(sp))
    #os.makedirs(path_spfull, exist_ok=True)

    path_spcrops = os.path.join(crops_folder, sp)
    os.makedirs(path_spcrops, exist_ok=True)

    #present_files = list_files(path_spfull)
    #coco_subset = {key: value for key, value in coco_data.items['images']() if value in present_files}
    
    for u_ann in tqdm(coco_data['annotations']):
      path = os.path.join(animals_folder, sp)
      name = os.path.join(path, im_dic[u_ann['image_id']])
      if not os.path.exists(name):
        #print(f'Image "{name}" does not exist.')
        continue
      name2 = os.path.join(path_spcrops, im_dic[u_ann['image_id']])
      if not os.path.exists(name2):
        #crop(f'Image "{name2}" does not exist.')
        continue
      print('cropping files of species ' + str(name2))
      img = Image.open(name)
      width, height = img.size

      # plt.figure()
      # plt.imshow(img)
      # ax = plt.gca()
      # ax.add_patch(Rectangle(
      #           (u_ann['bbox'][0], u_ann['bbox'][1],),
      #           u_ann['bbox'][2], u_ann['bbox'][3],
      #           ec='r', fill=False
      #         ))
      # name1 = os.path.join(path_spfull, im_dic[u_ann['image_id']])
      # plt.savefig(name1)

      crop_img = img.crop((u_ann['bbox'][0], u_ann['bbox'][1], u_ann['bbox'][0] + u_ann['bbox'][2], u_ann['bbox'][1] + u_ann['bbox'][3]))
      crop_img.save(name2)
      
# %%

crop_images(coco_path, animals_folder, output_folder) 
# %%
