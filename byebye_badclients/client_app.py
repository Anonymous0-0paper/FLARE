"""ByeBye-BadClients: A Flower / PyTorch app."""
import time
import random

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.exceptions import UndefinedMetricWarning

from byebye_badclients.task import load_data, set_weights, test, train, get_update, get_weights
from sklearn import metrics
from enum import Enum

from byebye_badclients.util import load_model

import warnings
# Ignore deprecation warnings from datasets/dill
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Completely ignore sklearn undefined metric warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def get_class_distribution(dataloader):
    def add_label(counts, lbl):
        if lbl not in counts:
            counts[lbl] = 0
        counts[lbl] += 1

    class_counts = {}
    for batch in dataloader:
        labels = batch["label"]
        for label in labels:
            add_label(class_counts, label.item())
    return dict(sorted(class_counts.items()))

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, dataset_name, cid, role, attack_pattern, existing_attack_patterns, update_scaling_factor, no_defense_fedavg):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.dataset_name = dataset_name
        self.cid = cid
        self.role = role
        self.attack_pattern = attack_pattern
        self.existing_attack_patterns = existing_attack_patterns
        self.update_scaling_factor = update_scaling_factor
        self.no_defense_fedavg = no_defense_fedavg

    def fit(self, parameters, config):
        receive_time = time.time()

        server_send_time = config["send_time"]

        set_weights(self.net, parameters)

        num_classes = 100 if config["dataset"] == "cifar100" else 10

        attack_patterns = {"flip_labels": False,
                           "random_update": False,
                           "update_scaling": False,
                           "alie": False,
                           "statistical_mimicry": False,}
        if self.role == Role.MALICIOUS:
            if self.attack_pattern == 'label-flipping':
                attack_patterns["flip_labels"] = True

            elif self.attack_pattern == 'random-update':
                attack_patterns["random_update"] = True

            elif self.attack_pattern == 'update-scaling':
                attack_patterns["update_scaling"] = True

            elif self.attack_pattern == 'alie':
                attack_patterns["alie"] = True

            elif self.attack_pattern == 'statistical_mimicry':
                attack_patterns["statistical_mimicry"] = True

            else:
                rng = random.Random(time.time())
                benign_malicious = rng.choices([Role.MALICIOUS, Role.BENIGN], weights=[0.7, 0.3], k=1)[0]
                if benign_malicious == Role.MALICIOUS:
                    choice = rng.choices(list(attack_patterns.keys()))[0]
                    attack_patterns[choice] = True

        train_start = time.time()
        train_loss, train_acc = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            flip_labels=attack_patterns["flip_labels"],
            random_update=attack_patterns["random_update"],
            update_scaling=attack_patterns["update_scaling"],
            alie=attack_patterns["alie"],
            statistical_mimicry=attack_patterns["statistical_mimicry"],
            factor=self.update_scaling_factor,
            num_classes=num_classes,
            cid=self.cid
        )
        train_end = time.time()

        parameters = get_weights(self.net) if self.no_defense_fedavg else get_update(self.net, parameters)

        return (
            parameters,
            len(self.trainloader.dataset),
            {"cid": self.cid,
             "loss": train_loss,
             "accuracy": train_acc,
             "train_time": (train_end - train_start),
             "receive_time": (receive_time - server_send_time),
             "role": str(self.role),
             "attack_pattern": self.attack_pattern},
        )
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)

        loss, accuracy, m = test(self.net, self.valloader, self.device)
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
                                                   "f1_score": f1_score,
                                                   "role": str(self.role)}

class Role(Enum):
    BENIGN = 0
    MALICIOUS = 1

def get_role(node_id, malicious_probability, attack_patterns):
    if attack_patterns:
        rng = random.Random(node_id)
        choices = [Role.MALICIOUS, Role.BENIGN]
        probabilities = [malicious_probability, 1-malicious_probability]
        role = rng.choices(choices, weights=probabilities, k=1)[0]
        attack_pattern = rng.choices(list(attack_patterns), k=1)[0]
    else:
        role = Role.BENIGN
        attack_pattern = ''
    return role, attack_pattern

def client_fn(context: Context):
    hf_dataset = context.run_config["dataset"]
    total_max_samples = context.run_config["total-max-samples"]
    non_iid = context.run_config["non-iid"]
    dirichlet_alpha = context.run_config["dirichlet-alpha"]
    malicious_probability = context.run_config["malicious-probability"]
    attack_patterns = context.run_config["attack-patterns"]
    attack_patterns = set(attack_patterns.split(","))
    possible_attack_patterns = {"update-scaling", "label-flipping", "random-update", "adaptive-attack", "alie", "statistical-mimicry"}
    if len(possible_attack_patterns - attack_patterns) == 6:
        attack_patterns = []
    update_scaling_factor = context.run_config["update-scaling-factor"]
    # Load model and data
    dataset = hf_dataset.split('/')[-1]
    net = load_model(dataset)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    no_defense_fedavg = context.run_config["no-defense-fedavg"]

    trainloader, valloader = load_data(partition_id, num_partitions, hf_dataset, total_max_samples=total_max_samples,
                                       non_iid=non_iid, dirichlet_alpha=dirichlet_alpha)
    local_epochs = context.run_config["local-epochs"]

    role, attack_pattern = get_role(partition_id, malicious_probability, attack_patterns)
    print("Cuda Available: ", torch.cuda.is_available())
    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, dataset, cid=partition_id,
                        role=role, attack_pattern=attack_pattern, existing_attack_patterns=attack_patterns,
                        update_scaling_factor=update_scaling_factor, no_defense_fedavg=no_defense_fedavg).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
