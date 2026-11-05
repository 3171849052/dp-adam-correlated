import torch
from torch import nn
class CNN(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(1,8,5,padding=2),nn.ReLU(),nn.AvgPool2d(2),nn.Conv2d(8,16,5,padding=2),nn.ReLU(),nn.AvgPool2d(2),nn.Flatten(),nn.Linear(16*7*7,10))
def tensors(official=False):
    from exp18 import BASE
    data=torch.load(BASE/'data'/('test.pt' if official else 'train.pt'),weights_only=True)
    return data['images'],data['labels']
