"""ByeBye-BadClients: A Flower / PyTorch app."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Parameters
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(128*4*4, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.adaptive_pool(x)
        x = x.view(-1, 128*4*4)
        return self.fc1(x)


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


def create_dirichlet_partition(labels: np.ndarray, num_partitions: int, alpha: float = 0.5) -> dict:
    """Create Dirichlet non-IID partitions based on labels."""
    # Set seed for reproducible partitioning
    np.random.seed(num_partitions)

    num_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * num_classes, num_partitions)

    # Get indices for each class
    class_indices = {}
    for class_id in range(num_classes):
        class_indices[class_id] = np.where(labels == class_id)[0]

    partitions = {i: [] for i in range(num_partitions)}

    for class_id in range(num_classes):
        class_idx = class_indices[class_id].copy()
        np.random.shuffle(class_idx)

        # Split class indices according to Dirichlet distribution
        splits = np.cumsum(label_distribution[:, class_id] * len(class_idx)).astype(int)
        splits = np.concatenate([[0], splits])

        for partition_id in range(num_partitions):
            start_idx = splits[partition_id]
            end_idx = splits[partition_id + 1]
            partitions[partition_id].extend(class_idx[start_idx:end_idx].tolist())

    for partition_id in range(num_partitions):
        np.random.seed(partition_id)
        np.random.shuffle(partitions[partition_id])

    return partitions


def create_iid_partition(dataset_size: int, num_partitions: int) -> dict:
    """Create IID partitions."""
    # Set seed for reproducible partitioning
    np.random.seed(num_partitions)

    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    partition_size = dataset_size // num_partitions
    partitions = {}

    for i in range(num_partitions):
        start_idx = i * partition_size
        if i == num_partitions - 1:  # Last partition gets remaining samples
            end_idx = dataset_size
        else:
            end_idx = (i + 1) * partition_size
        partitions[i] = indices[start_idx:end_idx]

    return partitions


class PartitionDataset:
    """Mock FederatedDataset partition to maintain similar API."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def train_test_split(self, test_size=0.2, seed=42):
        """Split partition into train and test, mimicking HF dataset API."""
        dataset_size = len(self.indices)
        test_size_actual = int(test_size * dataset_size)
        train_size_actual = dataset_size - test_size_actual

        # Use torch random split for consistency
        train_indices, test_indices = random_split(
            self.indices,
            [train_size_actual, test_size_actual],
            generator=torch.Generator().manual_seed(seed)
        )

        train_subset = TransformableSubset(self.dataset, list(train_indices))
        test_subset = TransformableSubset(self.dataset, list(test_indices))

        return {
            "train": train_subset,
            "test": test_subset
        }


class TransformableSubset(Subset):
    """Subset that can have transforms applied, mimicking HF dataset with_transform."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.pytorch_transforms = None

    def with_transform(self, pytorch_transforms):
        """Apply PyTorch transforms directly to images."""
        self.pytorch_transforms = pytorch_transforms
        return self

    def __getitem__(self, idx):
        # Get the actual dataset index
        dataset_idx = self.indices[idx]

        # Get image and label from original torchvision dataset
        image, label = self.dataset[dataset_idx]

        # Apply transforms directly to the image
        if self.pytorch_transforms:
            image = self.pytorch_transforms(image)

        # Return as dictionary to match HF dataset format expected by your training code
        return {"image": image, "label": label}


class MockFederatedDataset:
    """Mock FederatedDataset to maintain similar API structure."""

    def __init__(self, dataset_name, subset, partitions, torchvision_dataset):
        self.dataset_name = dataset_name
        self.subset = subset
        self.partitions = partitions
        self.torchvision_dataset = torchvision_dataset

    def load_partition(self, partition_id):
        """Load a specific partition, mimicking FederatedDataset API."""
        partition_indices = self.partitions[partition_id]
        return PartitionDataset(self.torchvision_dataset, partition_indices)

partitioned_datasets = {}
def load_data(
    partition_id: int,
    num_partitions: int,
    dataset: str,
    total_max_samples: int,
    non_iid: bool = False,
    dirichlet_alpha: float = 0.5,
):
    """Load partition data using torchvision datasets (no HuggingFace)."""

    dataset_name = dataset.split("/")[-1]

    # Pick cache dir
    cache_dir = os.environ.get("TORCHVISION_DATA_DIR", f"./data/")
    os.makedirs(cache_dir, exist_ok=True)

    # Partition cache key
    partition_key = f"{dataset_name}_{num_partitions}_{non_iid}_{dirichlet_alpha}"

    if partition_key not in partitioned_datasets:
        print(f"Initializing partitioned dataset for {dataset_name}...")

        # Choose transforms
        if dataset_name == "mnist":
            transform = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])
            torchvision_dataset = torchvision.datasets.MNIST(
                root=cache_dir, train=True, download=True, transform=transform
            )
        elif dataset_name == "cifar10":
            transform = Compose(
                [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
            torchvision_dataset = torchvision.datasets.CIFAR10(
                root=cache_dir, train=True, download=True, transform=transform
            )
        elif dataset_name == "cifar100":
            transform = Compose(
                [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
            torchvision_dataset = torchvision.datasets.CIFAR100(
                root=cache_dir, train=True, download=True, transform=transform
            )
        elif dataset_name == "svhn":
            transform = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])
            torchvision_dataset = torchvision.datasets.SVHN(
                root=cache_dir, split="train", download=True, transform=transform
            )
        else:
            print(f"Warning: Unknown dataset {dataset_name}, defaulting to CIFAR10")
            transform = Compose(
                [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
            torchvision_dataset = torchvision.datasets.CIFAR10(
                root=cache_dir, train=True, download=True, transform=transform
            )

        # Extract labels
        if hasattr(torchvision_dataset, "targets"):
            labels = np.array(torchvision_dataset.targets)
        elif hasattr(torchvision_dataset, "labels"):
            labels = np.array(torchvision_dataset.labels)
        else:
            labels = np.array([lbl for _, lbl in torchvision_dataset])

        # Partition indices
        if non_iid:
            partitions = create_dirichlet_partition(
                labels, num_partitions, dirichlet_alpha
            )
        else:
            partitions = create_iid_partition(
                len(torchvision_dataset), num_partitions
            )

        partitioned_datasets[partition_key] = (torchvision_dataset, partitions)

    torchvision_dataset, partitions = partitioned_datasets[partition_key]

    # Pick this client’s partition
    indices = partitions[partition_id]
    partition = Subset(torchvision_dataset, indices)

    # Train/test split (80/20)
    n_total = len(partition)
    n_test = n_total // 5
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_set = Subset(torchvision_dataset, train_indices)
    test_set = Subset(torchvision_dataset, test_indices)

    # Cap max samples if requested
    if total_max_samples != -1:
        max_train = min(total_max_samples, len(train_set))
        max_test = min(total_max_samples, len(test_set))
        train_set = Subset(train_set, list(range(max_train)))
        test_set = Subset(test_set, list(range(max_test)))

    # DataLoaders
    trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
    testloader = DataLoader(test_set, batch_size=32)

    return trainloader, testloader


def flip_labels_fn(labels, num_classes=10):
    return (labels + 1) % num_classes


def random_update_fn(net: nn.modules.Module):
    params = net.parameters()
    for param in params:
        param.data = torch.randn_like(param.data)  # Normal distribution with µ = 0 and std = 1

def update_scaling_fn(net: nn.modules.Module, factor: float):
    params = net.parameters()
    for param in params:
        param.data *= factor

def mean_and_sigma_flat(parameters_list):
    flat_tensors = []
    for p in parameters_list:
        ndarrays = parameters_to_ndarrays(p)
        # Flatten all arrays and concatenate
        flat = torch.cat([torch.from_numpy(arr).float().view(-1) for arr in ndarrays])
        flat_tensors.append(flat)

    stacked = torch.stack(flat_tensors)
    mean_flat = torch.mean(stacked, dim=0)
    sigma_flat = torch.std(stacked, dim=0, unbiased=False)
    return mean_flat, sigma_flat

def flatten_parameters(parameters):
    ndarrays = parameters_to_ndarrays(parameters)
    return torch.cat([torch.from_numpy(x).float().view(-1) for x in ndarrays])

def unflatten_parameters(flat_tensor, template_parameters):
    ndarrays = parameters_to_ndarrays(template_parameters)
    new_nd = []
    idx = 0
    for arr in ndarrays:
        numel = arr.size
        new_arr = flat_tensor[idx:idx + numel].reshape(arr.shape).numpy()
        new_nd.append(new_arr)
        idx += numel
    return ndarrays_to_parameters(new_nd)

STATE_DIR = "/tmp/flwr_client_state"
os.makedirs(STATE_DIR, exist_ok=True)

def statistical_mimicry_fn(net: torch.nn.Module, cid, alpha=0.5):
    # Convert current net to Flower Parameters
    current_params = ndarrays_to_parameters([p.detach().cpu().numpy() for p in net.parameters()])

    state_path = f"{STATE_DIR}/{cid}.pt"

    if os.path.exists(state_path):
        # allowlist numpy.ndarray for unpickling
        with torch.serialization.safe_globals([Parameters, np.ndarray]):
            loaded_state = torch.load(state_path, weights_only=False)

        # loaded_state["parameters"] is list of ndarrays, convert to Flower Parameters
        state = {
            "parameters": [ndarrays_to_parameters(p) for p in loaded_state["parameters"]],
            "direction": loaded_state["direction"],
            "direction_accumulation": loaded_state["direction_accumulation"],
        }
        state["parameters"].append(current_params)
    else:
        # Initialize state
        state = {
            "parameters": [current_params],
            "direction": torch.randn_like(flatten_parameters(current_params)),
            "direction_accumulation": 0.1,
        }

    # Compute mean and sigma
    mu, sigma = mean_and_sigma_flat(state["parameters"])

    # Sample epsilon ~ N(0, sigma)
    epsilon = torch.randn_like(sigma) * sigma

    current_flat = flatten_parameters(current_params)
    g_flat = (
        (1 - alpha) * current_flat
        + alpha * (mu + epsilon)
        + state["direction_accumulation"] * state["direction"]
    )

    state["direction_accumulation"] *= 1.2 if state["direction_accumulation"] < 2 else state["direction_accumulation"]
    state["direction"] = state["direction"] / (torch.norm(state["direction"]) + 1e-12)

    # Save state (ndarrays are safe)
    torch.save(
        {
            "parameters": [parameters_to_ndarrays(p) for p in state["parameters"]],
            "direction": state["direction"],
            "direction_accumulation": state["direction_accumulation"],
        },
        state_path,
    )

    g_parameters = unflatten_parameters(g_flat, current_params)
    return g_parameters


def train(net, trainloader, epochs, device, flip_labels=False, random_update=False, update_scaling=False, alie=False, statistical_mimicry=False, factor=2, num_classes=10, cid=None):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for _ in range(epochs):
        for i, (images, labels) in enumerate(trainloader):   # <-- unpack tuple
            images, labels = images.to(device), labels.to(device)
            if flip_labels:
                labels = flip_labels_fn(labels, num_classes)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_trainloss = running_loss / (len(trainloader) * epochs)

    if random_update:
        random_update_fn(net)
    elif update_scaling:
        update_scaling_fn(net, factor=factor)
    elif statistical_mimicry:
        statistical_mimicry_fn(net, round, cid)
    return avg_trainloss, accuracy


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    correct, loss = 0, 0.0
    predictions = []
    prediction_labels = []

    total_loss = 0.0
    total_examples = 0

    net.eval()
    with torch.no_grad():
        for images, labels in testloader:   # <-- unpack tuple
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            prediction = torch.max(outputs.data, 1)[1]

            predictions.append(prediction.cpu().numpy().tolist())
            prediction_labels.append(labels.cpu().numpy().tolist())

            batch_loss = criterion(outputs, labels)
            batch_size = images.size(0)

            total_loss += batch_loss.item() * batch_size
            total_examples += batch_size
            correct += (prediction == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = total_loss / total_examples

    return loss, accuracy, {
        "predictions": np.concatenate(predictions),
        "labels": np.concatenate(prediction_labels),
    }

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

def parameters_to_tensor(parameters) -> torch.Tensor:
    """Convert flwr.common.Parameters to a single flattened tensor."""
    ndarrays = parameters_to_ndarrays(parameters)  # list of numpy arrays
    flat = torch.tensor([], dtype=torch.float32)
    for arr in ndarrays:
        flat = torch.cat([flat, torch.from_numpy(arr).float()])
    return flat

def parameters_to_tensorlist(parameters):
    ndarrays = parameters_to_ndarrays(parameters)
    return [torch.from_numpy(arr).float() for arr in ndarrays]

def parameters_multiply_scalar(parameters, scalar):
    for p in parameters():
        p.data = p.data * scalar
    return parameters

def freeze_model(net):
    for name, param in net.named_parameters():
        param.requires_grad = False