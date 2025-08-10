"""ByeBye-BadClients: A Flower / PyTorch app."""
import time
from collections import Counter

import torch
import torchvision

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from byebye_badclients.task import Net, get_weights, load_data, set_weights, test, train, NetMNIST, get_update
from sklearn import metrics

def get_class_distribution(dataloader):
    def add_label(counts, label):
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    class_counts = {}
    for batch in dataloader:
        labels = batch["label"]
        for label in labels:
            add_label(class_counts, label.item())
    return dict(sorted(class_counts.items()))

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, dataset_name, cid):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.dataset_name = dataset_name
        self.cid = cid

    def fit(self, parameters, config):
        receive_time = time.time()
        server_send_time = config["send_time"]

        set_weights(self.net, parameters)

        train_start = time.time()
        if self.dataset_name == "cifar10":
            train_loss = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.device,
                img_col_name="img"
            )
        else:
            train_loss = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.device,
                img_col_name="image"
            )
        train_end = time.time()
        return (
            get_update(self.net, parameters),
            len(self.trainloader.dataset),
            {"cid": self.cid,
             "loss": train_loss,
             "train_time": train_end-train_start,
             "receive_time": receive_time - server_send_time},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        if self.dataset_name == "cifar10":
            loss, accuracy, m = test(self.net, self.valloader, self.device, img_col_name="img")
        else:
            loss, accuracy, m = test(self.net, self.valloader, self.device, img_col_name="image")
        y_pred = m["predictions"]
        labels = m["labels"]

        recall_score = metrics.recall_score(y_pred=y_pred, y_true=labels, average='macro')
        precision_score = metrics.precision_score(y_pred=y_pred, y_true=labels, average='macro')
        f1_score = metrics.f1_score(y_pred=y_pred, y_true=labels, average='macro')

        return loss, len(self.valloader.dataset), {"cid": self.cid,
                                                   "accuracy": accuracy,
                                                   "loss": loss,
                                                   "recall_score": recall_score,
                                                   "precision_score": precision_score,
                                                   "f1_score": f1_score}

def client_fn(context: Context):
    hf_dataset = context.run_config["dataset"]
    total_max_samples = context.run_config["total-max-samples"]
    non_iid = context.run_config["non-iid"]
    dirichlet_alpha = context.run_config["dirichlet-alpha"]

    # Load model and data
    dataset = hf_dataset.split('/')[-1]
    if dataset == 'mnist':
        net = NetMNIST()
    elif dataset == 'cifar10':
        net = Net()
    else:
        net = torchvision.models.resnet18(pretrained=True)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader = load_data(partition_id, num_partitions, hf_dataset, total_max_samples=total_max_samples,
                                       non_iid=non_iid, dirichlet_alpha=dirichlet_alpha)
    local_epochs = context.run_config["local-epochs"]

    # print("DEVICE: Cuda" if torch.cuda.is_available() else "Device: CPU")
    # print(f"NODE_CONFIG: {context.node_config}\n"
    #       f"PARTITION ID: {partition_id}\n"
    #       f"CLASS_DISTRIBUTION: {get_class_distribution(trainloader)}")

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, dataset, cid=partition_id).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
