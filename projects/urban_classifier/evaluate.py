import yaml
import torch
import scipy
import numpy as np
import argparse
import os
from glob import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from model import CustomResNet18
from train import create_dataloader, load_model 
import pandas as pd
import random
import IPython
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from train import DataLoader

# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18
from torch.utils.tensorboard import SummaryWriter 

def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob('model_states/*.pt')
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace('model_states/','').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch

def predict(cfg, dataLoader, model):
    with torch.no_grad(): # no gradients needed for prediction
        predictions = [] ## predictions as tensor probabilites
        true_labels = [] ## labels as 0, 1 .. (classes)
        predicted_labels = [] ## labels as 0, 1 .. (classes)
        confidences = [] ## soft max of probabilities 
        ##### may need to adjust this in the dataloader for the sequence:
        ### this will evaluate on each batch of data (usually 64)
        #IPython.embed()
        #print(len(dataLoader)) ## number of total divisions n/batchsize
        for idx, (data, label) in enumerate(dataLoader): 
                #print(idx)
            true_label = label.numpy()
            true_labels.extend(true_label)

            prediction = model(data) ## the full probabilty
            predictions.append(prediction)
            #print(prediction.shape) ## it is going to be [batch size #num_classes]
            
            ## predictions
            predict_label = torch.argmax(prediction, dim=1).numpy() ## the label
            predicted_labels.extend(predict_label)
            #print(predict_label)

            confidence = torch.nn.Softmax(dim=1)(prediction).numpy()
            confidence = confidence[:,1]
            confidences.extend(confidence)

    true_labels = np.array(true_labels)
    #print(true_labels)
   # print(len(true_labels))
    predicted_labels = np.array(predicted_labels)
    #print(predicted_labels)
    #print(len(predicted_labels))
    #### this should be full dataset as a dataframe
    #results = pd.DataFrame({"true_labels": true_labels, "predict_label":predicted_labels}) #"confidence":confidence

    return true_labels, predicted_labels, confidences

def save_confusion_matrix(true_labels, predicted_labels, cfg, args, epoch='200', split='train'):
    # make figures folder if not there

    matrix_path = cfg['data_root']+'/figs'
    #### make the path if it doesn't exist
    if not os.path.exists(matrix_path):  
        os.makedirs(matrix_path, exist_ok=True)

    confmatrix = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confmatrix)
    disp.savefig(cfg['data_root'] +'/figs/confusion_matrix_epoch'+'_'+ str(split) +'.png', facecolor="white")
    
       ## took out epoch)
    return confmatrix


def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/cfg.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='projects/urban_classifier/configs/cfg.yaml')
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

    # initialize model
    model, current_epoch = load_model(cfg)

    #predict
    true_labels, predicted_labels, confidence = predict(cfg, dl_val, model)

     # get accuracy score
    ### this is just a way to get two decimal places 
    acc = accuracy_score(true_labels, predicted_labels)
    print("Accuracy of model is {:0.2f}".format(acc))

    # confusion matrix
    confmatrix = save_confusion_matrix(true_labels, predicted_labels, cfg, args, epoch = 200, split = 'train')
    print("confusion matrix saved")

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


if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
    
    



