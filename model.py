import torch
import torch.nn as nn
import torchvision
from transformers import BertTokenizer, BertModel

from config import device

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
   param.requires_grad = False

model_conv = model_conv.to(device)
num_ftrs = model_conv.fc.in_features


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
for param in bert.parameters():
   param.requires_grad = False

bert = bert.to(device)


class Net(nn.Module):
   def __init__(self):
      super(Net, self).__init__()
   def forward(self,notes,comments,imgs):
      notes_input = tokenizer(notes, return_tensors='pt', padding=True) # "pt"表示"pytorch"
      notes_tensor = bert(**notes_input).pooler_output.to(device)

      comments_input = tokenizer(comments, return_tensors='pt', padding=True) # "pt"表示"pytorch"
      comments_tensor = bert(**comments_input).pooler_output.to(device)

      imgs_tensor = model_conv(imgs)
      # print(notes_tensor.shape,comments_tensor.shape,imgs_tensor.shape)
      features = torch.cat((notes_tensor,comments_tensor,imgs_tensor),dim=-1)
      return features

model = Net()