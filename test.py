import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import os
import time

from model import model,device
from dataloader import dataloader
from train import data_transforms
from config import positive_threshold,negative_threshold

data_dir = 'data/2021MCM_ProblemC_Files'

# classify a new sample into its label
def classifier(features):
    centers = np.load('output/centers.npy')
    center_labels = np.load('output/center_labels.npy')
    pred_center_labels = []
    # PCA
    pca = PCA(n_components=3)
    pca.fit(features)
    features = pca.transform(features)
    # find labels
    for i in range(features.shape[0]):
        feature = features[i,:][np.newaxis, :]
        closest_center = 0
        for c in range(centers.shape[0]):
            center_vector = centers[c,:][np.newaxis, :]
            if euclidean_distances(feature,center_vector) \
                < euclidean_distances(feature,centers[closest_center,:][np.newaxis, :]):
                closest_center = c
        final_label = center_labels[closest_center]
        # Set a threshold that only gets close enough to the center to be classified directly, 
        # otherwise you need to continue investigating
        threshold = positive_threshold if final_label == 1 else negative_threshold
        # print(euclidean_distances(feature,centers[closest_center,:][np.newaxis, :]))
        final_label = final_label if \
            euclidean_distances(feature,centers[closest_center,:][np.newaxis, :]) < threshold else 2
        pred_center_labels.append(final_label)
    return pred_center_labels


def test():
    start_time = time.time()
    count = 0
    wrong_number = 0
    pred_labels = []
    for notes,comments,latitudes,longitudes,imgs,labels in dataloader:
        try:
            latitudes = latitudes.to(device)
            longitudes =  longitudes.to(device)
            latitudes = torch.unsqueeze(latitudes, -1)
            longitudes =  torch.unsqueeze(longitudes, -1)
            imgs = torch.tensor([data_transforms['val'](Image.open(os.path.join(data_dir,img)).convert('RGB') if img!='' and img!=None 
                else Image.fromarray(np.zeros((224,224,3),dtype='uint8'))).numpy().tolist() for img in imgs]).to(device)

            features = model(notes, comments, imgs)
            features = torch.cat((latitudes,longitudes,features),-1)
            features = features.detach().numpy()
            
            count += 1
            labels_ = classifier(features)
            print(f"Batch number:{count}, Running time:{time.time()-start_time:.1f}s",end=" ")
            print(f"Predicted labels are:\n",labels_)
            pred_labels.append(labels_)
        except:
            print(f"Batch number:{count} has something wrong!")
            wrong_number += 1
    pred_labels = np.concatenate(pred_labels,axis=0)
    with open('output/labels.txt','w',encoding='utf-8') as f1:
        for l in range(pred_labels.shape[0]):
            f1.write(str(pred_labels[l]))

    print(f"The number of batches is:{count}, and wrong number is {wrong_number}")

test()
