import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import time

from model import model,device
from dataloader import dataloader


data_dir = 'data/2021MCM_ProblemC_Files'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def train():
    start_time = time.time()
    count = 0
    wrong_number = 0
    for notes,comments,latitudes,longitudes,imgs,labels in dataloader:
        try:
            latitudes = latitudes.to(device)
            longitudes =  longitudes.to(device)
            latitudes = torch.unsqueeze(latitudes, -1)
            longitudes =  torch.unsqueeze(longitudes, -1)
            imgs = torch.tensor([data_transforms['val'](Image.open(os.path.join(data_dir,img)).convert('RGB') if img!='' and img!=None 
                else Image.fromarray(np.zeros((224,224,3),dtype='uint8'))).numpy().tolist() for img in imgs]).to(device)

            features = model(notes, comments, imgs)
            # print(features.shape)
            features = torch.cat((latitudes,longitudes,features),-1)
            # print(features.shape)
            # break
            features = features.detach().numpy()
            count += 1
            print(f"Batch number:{count}, Running time:{time.time()-start_time:.1f}s")
            np.save(f"output/X{count}.npy",features)
            np.save(f"output/Y{count}.npy",labels.numpy())
        except:
            print(f"Batch number:{count} has something wrong!")
            wrong_number += 1
    print(f"The number of batches is:{count}, and wrong number is {wrong_number}")

if __name__ == "__main__":
    train()

