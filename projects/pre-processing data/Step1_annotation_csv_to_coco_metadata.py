# %%
import os
import numpy as np
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm, trange
##this can be done with pip install -r "requirements.txt"

# %%
path_to_labels = r'C:\Users\tizge\Documents\GitHub\urbanwildlife_cv4e\datasets\TUW\annotations.csv'
images_folder = r'D:\animals'
#images_folder = r'C:/Users/tizge/Documents/GitHub/urbanwildlife_cv4e/datasets/TUW/coyotes'
megadetector_output = r'datasets/TUW/megaclassifier_slash.json' # the path to the CSV file
save_json_path = r'datasets/TUW/training_dataset.json'

# %%

def preprocess (path_to_labels, megadetector_output, images_folder, save_json_path='training_dataset.json'):
    #0. preparing, name output jason file
    
    data = pd.read_csv(path_to_labels)

    #creating new columns in the csv where:
    #convert filenames to codes
    data['image_id'] = data['Filename_full'].astype('category').cat.codes
    #get index of every image
    data['id'] = data.index
     
    #setting up the code that will run for each row
    #define function for creating the dictionary within list images,
    # %%
    def image(row):
        image = {}
        #image["height"] = row.height
        #image["width"] = row.width
        image["image_id"] = row.image_id
        image["image_id_full"] = row.Filename_full
        image["image_id_alt"] = row.Filename_alt
        image["station"] = row.location
        image["datetime"] = row.DateTime
        #image["seq_id"] = row.id ## this would be used if there were are more pictures per trigger

        image["category_id"] = row.Annotations_c
        image["category_name"] = row.Annotations

        image["file_path"] = os.path.join(row.Annotations, row.Filename_alt)

        return image
    # %% 
    # create a dict of all images
    print('Creating image dict...')
    imagedf = data.drop_duplicates(subset=['image_id']).sort_values(by='image_id')
    images_dict = {}
    progressBar = trange(len(imagedf))

    for row in imagedf.itertuples():

        progressBar.update(1)
        img = image(row)
        key = img["image_id_full"]

        filePath = os.path.join(images_folder, img['file_path'])
        if not os.path.exists(filePath):
        # don't care about image
            continue

        # add image size
        imgObj = Image.open(filePath)
        width, height = imgObj.size
        img['width'] = width
        img['height'] = height

        images_dict[key] = img

    progressBar.close()

    # %%
    # add bounding boxes from MegaDetector
    with open(megadetector_output) as f:
        megadetector = json.load(f)

    print('Merging with MegaDetector predictions...')
    for img in tqdm(megadetector['images']):
        if 'detections' not in img or img['detections'] is None:
            continue
        key = img["file"]
        if key not in images_dict:
            # MegaDetector image not found in Timelapse dataset
            continue
        
        width, height = images_dict[key]['width'], images_dict[key]['height']

        images_dict[key]["annotations"] = []
         
        for det in img['detections']:
            if det['conf'] <= 0.80:
                continue
            
            #convert relative XY/XY to absolute XYWH coordinates
            bbox = det["bbox"]
            bbox[0] *= width      # equivalent: bbox[0] = bbox[0] * width
            bbox[1] *= height
            bbox[2] *= width
            bbox[3] *= height

            images_dict[key]["annotations"].append(bbox)


    # convert to COCO
    print('Convert to COCO format...')
    images_out = []
    annotations_out = []
    categories_out = []
    categories_dict = {}   # mapping from category name to category COCO id

    for entry in images_dict.values():
        if "width" not in entry or not len(entry["annotations"]):
            continue
        image_id = len(images_out) + 1
        images_out.append({
            "image_id": image_id,
            "file_name": entry["image_id_alt"],
            "file_path": entry["file_path"],
            "station": entry["station"],
            "datetime": entry["datetime"],
            "width": entry["width"],
            "height": entry["height"]
            # add more metadata here if you want for the image
        })

        catName = entry["category_name"]
        if catName not in categories_dict:
            catID = len(categories_dict)+1
            categories_dict[catName] = catID
            categories_out.append({
                "id": catID,
                "name": catName
            })
        catID = categories_dict[catName]
        bboxes = entry["annotations"]
        for bbox in bboxes:
            annotations_out.append({
                "id": len(annotations_out)+1,
                "image_id": image_id,
                "category_id": catID,
                "bbox": bbox
            })

    # save as COCO file
    data_coco = {
        "images": images_out,
        "annotations": annotations_out,
        "categories": categories_out
    }
    json.dump(data_coco, open(save_json_path, "w"), indent=4)


# %%
##running function
preprocess(path_to_labels, megadetector_output, images_folder, save_json_path)



# %% checking how many within each annotation

#list = []
#for u in coco_data['annotations']:
#    if u['category_id'] == 13:
#        list.append(u['bbox'])
#print(len(list))

# %%

#coco_data['images'][1]['file_name'].split(".")[0]