import scipy.spatial as spatial
import torch
import torchvision

from byebye_badclients.task import NetMNIST, Net, freeze_model


def mahalanobis_distance(update, mean, inv_covariance):
    return spatial.distance.mahalanobis(update, mean, inv_covariance)

def load_model(dataset: str):
    if dataset == 'mnist':
        net = NetMNIST()
    elif dataset == 'cifar10':
        net = Net()
    else:
        net = torchvision.models.resnet18(pretrained=True)
        freeze_model(net)
        net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=10)
    return net