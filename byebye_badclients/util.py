import scipy.spatial as spatial
import torch
import torchvision

from byebye_badclients.task import NetMNIST, Net, freeze_model

def mahalanobis_distance(update, mean, inv_covariance):
    return spatial.distance.mahalanobis(update, mean, inv_covariance)

def load_model(dataset: str):
    num_classes = 100 if dataset == 'cifar100' else 10
    if dataset == 'mnist':
        net = NetMNIST()
    elif dataset == 'svhn':
        net = Net()
    else:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=weights)
        freeze_model(net)
        net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=num_classes)
    return net