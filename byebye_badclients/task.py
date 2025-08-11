"""ByeBye-BadClients: A Flower / PyTorch app."""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class NetMNIST(nn.Module):
    def __init__(self):
        super(NetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 13 * 13, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6 * 13 * 13)
        return self.fc1(x)

fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, dataset: str, total_max_samples: int, non_iid: bool = False, dirichlet_alpha: float = 0.5):
    """Load partition CIFAR10 data."""
    dataset_name = dataset.split('/')[-1]

    subset = None
    if dataset_name == 'svhn':
        subset = "cropped_digits"

    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by='label', alpha=dirichlet_alpha) if non_iid else IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset=dataset_name,
            subset=subset,
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    dataset = dataset.split('/')[-1]
    if dataset == "mnist":
        pytorch_transforms = Compose([ToTensor(), Normalize(mean=0.5, std=0.5)])
        img_col_name = 'image'

    elif dataset == "cifar10":
        pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img_col_name = 'img'
    else:
        pytorch_transforms = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        img_col_name = 'image'


    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[img_col_name] = [pytorch_transforms(img) for img in batch[img_col_name]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)

    if total_max_samples != -1:
        max_samples = min(total_max_samples, len(partition_train_test["train"]))
        partition_train_test["train"] = torch.utils.data.Subset(partition_train_test["train"], list(range(max_samples)))

        max_samples = min(max_samples, len(partition_train_test["test"]))
        partition_train_test["test"] = torch.utils.data.Subset(partition_train_test["test"], list(range(max_samples)))


    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device, img_col_name="image"):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    net.train()
    correct, running_loss = 0, 0.0
    for _ in range(epochs):
        for i, batch in enumerate(trainloader):
            images = batch[img_col_name].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(trainloader.dataset)
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss, accuracy


def test(net, testloader, device, img_col_name="image"):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    predictions = []
    prediction_labels = []
    with torch.no_grad():
        for batch in testloader:
            images = batch[img_col_name].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            prediction = torch.max(outputs.data, 1)[1]
            predictions.append(prediction.cpu().numpy().tolist())
            prediction_labels.append(labels.cpu().numpy().tolist())
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy, {"predictions": np.concatenate(predictions), "labels": np.concatenate(prediction_labels)}


def get_weights(net: nn.Module):
    return [param.detach().cpu().numpy() for _, param in net.named_parameters() if param.requires_grad]

def get_update(local_net, global_params):
    local_weights = get_weights(local_net)
    return [local - global_w for local, global_w in zip(local_weights, global_params)]

def set_weights(net, parameters):
    trainable_keys = [k for k, param in net.named_parameters() if param.requires_grad]
    params_dict = zip(trainable_keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=False)

def freeze_model(net):
    for param in net.parameters():
        param.requires_grad = False