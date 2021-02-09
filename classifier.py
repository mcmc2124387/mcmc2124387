import torch
from torch import nn
from sklearn.model_selection import train_test_split
import time
import copy
import numpy as np

from config import learning_rate,weight_decay,num_epochs
from focalloss import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define models
classifier_n = nn.Linear(768,2)

classifier_c = nn.Linear(768,2)

classifier_f = nn.Linear(1000,2)

classifier_joint = nn.Sequential(
    nn.Linear(2538,128),
    nn.BatchNorm1d(128),
    nn.Linear(128,2)
)

classifier_n.to(device)
classifier_c.to(device)
classifier_f.to(device)
classifier_joint.to(device)

criterion = FocalLoss(gamma=2,alpha=0.25)
# criterion = nn.CrossEntropyLoss()

optimizer_n = torch.optim.Adam(classifier_n.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_c = torch.optim.Adam(classifier_c.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_f = torch.optim.Adam(classifier_f.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_joint = torch.optim.Adam(classifier_joint.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Load dataset
data_set = []
label_set = []
unverified_set = []
for i in range(1,260):
    X_item = np.load(f'output/X{i}.npy')
    Y_item = np.load(f'output/Y{i}.npy')
    for i in range(X_item.shape[0]):
        if Y_item[i] == 2:
            unverified_set.append(X_item[i,:])
        elif Y_item[i] == 1:
            for _ in range(100):
                data_set.append(X_item[i,:])
                label_set.append(Y_item[i])
        else:
            data_set.append(X_item[i,:])
            label_set.append(Y_item[i])  

X_train,X_test,y_train,y_test = train_test_split(data_set,label_set,test_size=0.2,stratify=label_set,random_state=42)
print(len(y_train),len(y_test))
train_set = [i for i in zip(X_train,y_train)]
test_set = [i for i in zip(X_test,y_test)]
data_total = [i for i in zip(data_set,label_set)]
dataloaders = {'train':torch.utils.data.DataLoader(data_total, batch_size=16, shuffle=False),
                'val':torch.utils.data.DataLoader(data_total, batch_size=16, shuffle=False)}

dataset_sizes = {'train':4140,'val':4140}


since = time.time()

best_n_wts = copy.deepcopy(classifier_n.state_dict())
best_c_wts = copy.deepcopy(classifier_c.state_dict())
best_f_wts = copy.deepcopy(classifier_f.state_dict())
best_joint_wts = copy.deepcopy(classifier_joint.state_dict())
best_acc = 0.0
best_precison = 0.0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            classifier_n.train()  # Set model to training mode
            classifier_c.train()
            classifier_f.train()
            classifier_joint.train()
        else:
            classifier_n.eval()   # Set model to evaluate mode
            classifier_c.eval()
            classifier_f.eval()
            classifier_joint.eval()

        running_loss = 0.0
        running_corrects = 0
        tp = 0
        count = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs_n, inputs_c, inputs_f, _ = inputs.split([768,768,1000,2],dim=1)
            inputs_n = inputs_n.float().to(device)
            inputs_c = inputs_c.float().to(device)
            inputs_f = inputs_f.float().to(device)
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer_n.zero_grad()
            optimizer_c.zero_grad()
            optimizer_f.zero_grad()
            optimizer_joint.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs_n = classifier_n(inputs_n)
                loss_n = criterion(outputs_n, labels)

                outputs_c = classifier_c(inputs_c)
                loss_c = criterion(outputs_c, labels)

                outputs_f = classifier_f(inputs_f)
                loss_f = criterion(outputs_f, labels)

                outputs_joint = classifier_joint(inputs)
                _, preds = torch.max(outputs_joint, 1)
                loss_joint = criterion(outputs_joint, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss_n.backward()
                    loss_c.backward()
                    loss_f.backward()
                    loss_joint.backward()
                    optimizer_n.step()
                    optimizer_c.step()
                    optimizer_f.step()
                    optimizer_joint.step()

            # statistics
            running_loss += loss_joint.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            tp += torch.sum(preds[preds == labels.data] == 1)
            count += torch.sum(preds == 1)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        precision = tp.double() / count

        print('{} Loss: {:.4f} Acc: {:.4f} Pre: {:.4f}'.format(
            phase, epoch_loss, epoch_acc, precision))

        if phase == 'val' and precision > best_precison:
            best_precison = precision

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_n_wts = copy.deepcopy(classifier_n.state_dict())
            best_c_wts = copy.deepcopy(classifier_c.state_dict())
            best_f_wts = copy.deepcopy(classifier_f.state_dict())
            best_joint_wts = copy.deepcopy(classifier_joint.state_dict())

    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}  Best precision: {:4f}'.format(best_acc,best_precison))

# load best model weights
classifier_n.load_state_dict(best_n_wts)
classifier_c.load_state_dict(best_c_wts)
classifier_f.load_state_dict(best_f_wts)
classifier_joint.load_state_dict(best_joint_wts)