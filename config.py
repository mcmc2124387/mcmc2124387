import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

learning_rate = 0.0004
weight_decay = 1e-5
num_epochs = 25

positive_threshold = 16 # Distance threshold for classification as a positive class
negative_threshold = 60 # Distance threshold for classification as a negative class