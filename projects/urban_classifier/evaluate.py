import json
import yaml
import torch
import scipy
import numpy as np
import argparse
import os
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, plot_confusion_matrix, average_precision_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from train import create_dataloader, load_model 
from tqdm import trange
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18
from torch.utils.tensorboard import SummaryWriter 

import csv
def predict(cfg, dataLoader, model, device):

    model.to(device)
    model.eval()

    with torch.no_grad(): # no gradients needed for prediction
        predictions = [] ## predictions as tensor probabilites
        true_labels = [] ## labels as 0, 1 .. (classes)
        predicted_labels = [] ## labels as 0, 1 .. (classes)
        confidences = [] ## soft max of probabilities 
        ##### may need to adjust this in the dataloader for the sequence:
        ### this will evaluate on each batch of data (usually 64)
        #IPython.embed()
        #print(len(dataLoader)) ## number of total divisions n/batchsize
        for idx, (data, label, image_path) in enumerate(tqdm(dataLoader)): 
                #print(idx)
            print(label)
            true_label = label.numpy()
            true_labels.extend(true_label)

            data = data.to(device)

            prediction = model(data) ## the full probabilty
            predictions.append(prediction.cpu())
            #print(prediction.shape) ## it is going to be [batch size #num_classes]
            
            ## predictions
            predict_label = torch.argmax(prediction.cpu(), dim=1).numpy() ## the label
            predicted_labels.extend(predict_label)
            #print(predict_label)

            confidence = torch.nn.Softmax(dim=1)(prediction).cpu().numpy()
            confidences.append(confidence)


    true_labels = np.array(true_labels)
    #print(true_labels)
   # print(len(true_labels))
    predicted_labels = np.array(predicted_labels)
    #print(predicted_labels)
    #print(len(predicted_labels))
    #### this should be full dataset as a dataframe
    #results = pd.DataFrame({"true_labels": true_labels, "predict_label":predicted_labels}) #"confidence":confidence

    return true_labels, predicted_labels, confidences

def save_confusion_matrix(true_labels, predicted_labels, cfg, args, epoch='200', split='train', split_type='random', log_dir = "latest_exp", labels=None):
    # make figures folder if not there

    os.makedirs('figs', exist_ok=True)

    confmatrix = confusion_matrix(true_labels, predicted_labels, normalize = 'true')
    print(true_labels)
    print(json.dumps(predicted_labels.tolist()))
    print(labels)
    # array to reflect 63 classes
    # labels = ['empty', 'human', 'vehicle', 'Coyote', 'Raccoon', 'Empty', 'Fox squirrel', 'Rabbit (cannot ID)', 'Unknown', 'Human', 'Desert cottontail rabbit', 'Bobcat', 'Western gray squirrel', 'Squirrel (cannot ID)', 'Bird', 'Striped Skunk', 'small mammal (cannot ID)', 'Mule deer', 'Domestic dog', 'Quail', 'Rat spp.', 'Butterfly', 'Turkey Vulture', 'Canada Goose', 'Greylag Goose', 'American robin', 'Mourning Dove', 'Virginia opossum', 'Killdeer', 'California Ground Squirrel', 'Lizard', 'Side-blotched Lizard', 'Common raven', 'California Towhee', 'Domestic cat', 'owl', 'California Scrub Jay', 'House Finch', 'California Thrasher', 'Western Fence Lizard', 'wren', 'Spotted Towhee', 'Deer (cannot ID)', 'White-crowned Sparrow', 'Western Toad', 'American Wigeon', 'Mallard duck', 'Gray fox', 'American Crow', 'Bird - Duck', 'Tiger Whiptail Lizard', 'Northern Mockingbird', 'Acorn Woodpecker', 'Hermit Thrush', 'Hummingbird', 'Horse', 'Brown Rat', 'Cormorant', 'Red-tailed hawk', 'Mountain lion', 'Kangaroo rat', 'Bird - Blue jay']
    disp = ConfusionMatrixDisplay(confmatrix, display_labels=labels)
    disp.plot(values_format = '.1f')
    #plt.show()
    plt.savefig(f'figs/{log_dir}/{split_type}/{split}/confusion_matrix_epoch_{epoch}.png', facecolor="white")
    
       ## took out epoch)
    return confmatrix

def generate_results(data_loader, split, cfg, model, epoch, device, args):

    split_type = cfg['split_type']
    log_dir = cfg['log_dir']

    #predict
    true_labels, predicted_labels, confidence = predict(cfg, data_loader, model, device)

    #generate function for running results with true_labels, predicted_labels, confidence as input variables
    # legend (species names in order)
    species_available = np.unique(true_labels).tolist() # change back to true_labels
    species_available.sort()
    mapping_inv = dict([v,k] for k,v in data_loader.dataset.species_to_index_mapping.items()) # maps all species, 62 elements
    legend = np.array([mapping_inv[s] for s in species_available]) # creates array of available species names, 52
    
    
    # reverse the map, get all values from mapping_inx (62)
    keys = list(mapping_inv.keys())
    values = list(mapping_inv.values())
    # creates map of all species
    mapping_species_avail = {} # want to create new map, only maps index to species of availble species (shorter version of mapping_inv)
    for name in legend:
        # get index ftom mapping_inv
        index = values.index(name)
        mapping_species_avail[index] = name  # maps available species names to value

    # create new array with only available speces, should be 52
    keys = list(mapping_species_avail.keys())
    legend_avail = []
    for ind in keys: # for index in mapping_speces_vail keys
        # print(ind, mapping_species_avail[ind])
        legend_avail.append(mapping_species_avail[ind])

    print(len(legend_avail))
    print('legend_avail', legend_avail)
    print(len(legend))
    print(legend)
    print(len(mapping_inv))
    print(mapping_inv)
    
    # get accuracy score
    ### this is just a way to get two decimal places 
    acc = accuracy_score(true_labels, predicted_labels)
    print("Accuracy of model is {:0.2f}".format(acc))

    # confusion matrix
    os.makedirs(f'figs/{log_dir}/{split_type}/{split}/prec_rec', exist_ok=True)
    confmatrix = save_confusion_matrix(true_labels, predicted_labels, cfg, args, epoch = epoch, split = split, split_type=split_type, log_dir = log_dir, labels=legend)
    print("confusion matrix saved")

            ###evaluation metrics:
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    confidence = np.concatenate(confidence, 0)
    for i in range(confidence.shape[1]):
        y_true = true_labels == i
        precision[i], recall[i], _ = precision_recall_curve(y_true, confidence[:, i])
        average_precision[i] = average_precision_score(y_true, confidence[:, i])

        display = PrecisionRecallDisplay(recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
        )
        display.plot()
        _ = display.ax_.set_title(f"Prec-Rec ep. {epoch}, species {mapping_species_avail.get(i, i)}")
        plt.savefig(f'figs/{log_dir}/{split_type}/{split}/prec_rec/epoch_{epoch}_{mapping_species_avail.get(i, i)}.png', facecolor="white")
    
    # for i in range(confidence.shape[0]): #for every image
    #     missclass = true_labels[i] != predicted_labels[i]
    #     if missclass == True:
    #         plt.imshow(img)
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

    #macro_average = sklearn.metrics.average_precision_score(true_labels, predicted_labels)
        #print("Macro average of model is {:0.2f}".format(macro_average))


def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/cfg.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='projects/urban_classifier/configs/cfg.yaml')

    # parser.add_argument('--config', help='Path to config file', default='/home/ykarandikar/cv4e/csvless/urbanwildlife_cv4e/projects/urban_classifier/configs/cfg.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize Tensorboard
    tbWriter = SummaryWriter(log_dir=cfg['log_dir'])
    os.makedirs(cfg['log_dir'], exist_ok=True)

    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')
    dl_test = create_dataloader(cfg, split='test')

    # initialize model
    model, epoch = load_model(cfg)

    generate_results(data_loader=dl_train, split='train', cfg = cfg, model=model, epoch=epoch, device =device, args = args)
    generate_results(data_loader=dl_val, split='val', cfg = cfg, model=model, epoch=epoch, device = device, args = args)
    generate_results(data_loader=dl_test, split='test', cfg = cfg, model=model, epoch=epoch, device=device, args = args)

    # train(52), val (28), test (48)

    ######################### put this all in a function ###############
    # #this must categorical
    # get precision score
    ### this is just a way to get two decimal places 
    #precision = precision_score(true_labels, predicted_labels)
    #print("Precision of model is {:0.2f}".format(precision))

    # get recall score
    ### this is just a way to get two decimal places 
    #recall = recall_score(true_labels, predicted_labels)
    #print("Recall of model is {:0.2f}".format(recall))

    # get recall score
    ### this is just a way to get two decimal places 
    #F1score = f1_score(true_labels, predicted_labels)
    #print("F1score of model is {:0.2f}".format(F1score))

#top-k accuracy
# class average accuracy - average precision
#sklearn.classifier.metrics
#classifier report

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
    
    



