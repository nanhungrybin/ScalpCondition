from torchvision import models
import torch.nn as nn



def RESNET50(num_classes):

    resnet18_pretrained = models.resnet18(weights="DEFAULT")
    num_features = resnet18_pretrained.fc.in_features
    resnet18_pretrained.fc = nn.Linear(num_features, num_classes)
    
    return resnet18_pretrained