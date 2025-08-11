"""ByeBye-BadClients: A Flower / PyTorch app."""
import os
import time
from typing import Optional

import numpy as np
import torch
import torchvision

from flwr.common import Context, ndarrays_to_parameters, Parameters, FitIns, parameters_to_ndarrays, \
    MetricsAggregationFn
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common.logger import log
from logging import WARNING

from byebye_badclients.result_processing import list_to_csv
from byebye_badclients.task import Net, NetMNIST, get_weights, freeze_model
from byebye_badclients.client_reputation import ClientReputation, Classification
from sklearn.random_projection import GaussianRandomProjection

def process_evaluate_results(results: dict[str, list[float]], dataset: str):
    dataset_name = dataset.split('/')[-1]
    base_path = os.path.abspath(f"plots/{dataset_name}")
    os.makedirs(base_path, exist_ok=True)

    '''Model Performance Results'''

    filepath = os.path.join(base_path, "weighted_avg_evaluation_loss.csv")
    list_to_csv(l=results["weighted_avg_loss"], filepath=filepath)

    filepath = os.path.join(base_path, "accuracy.csv")
    list_to_csv(l=results["weighted_avg_accuracy"], filepath=filepath)

    filepath = os.path.join(base_path, "recall_scores.csv")
    list_to_csv(l=results["weighted_avg_recall_score"], filepath=filepath)

    filepath = os.path.join(base_path, "precision_scores.csv")
    list_to_csv(l=results["weighted_avg_precision_score"], filepath=filepath)

    filepath = os.path.join(base_path, "f1-scores.csv")
    list_to_csv(l=results["weighted_avg_f1_score"], filepath=filepath)


evaluate_results = None
collect_evaluate_results_call_tracker = None
def collect_evaluate_results(metrics_dict: dict[str, float], context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    hf_dataset = context.run_config["dataset"]

    global collect_evaluate_results_call_tracker
    if collect_evaluate_results_call_tracker is None:
        collect_evaluate_results_call_tracker = 0
    collect_evaluate_results_call_tracker += 1

    global evaluate_results
    if evaluate_results is None:
        evaluate_results = {}

    for k, v in metrics_dict.items():
        if k not in evaluate_results:
            evaluate_results[k] = []
        evaluate_results[k].append(v)

    if collect_evaluate_results_call_tracker >= num_rounds:
        process_evaluate_results(results=evaluate_results, dataset=hf_dataset)

def process_fit_results(results: dict[str, list[float]], dataset: str):
    dataset_name = dataset.split('/')[-1]
    base_path = os.path.abspath(f"plots/{dataset_name}")
    os.makedirs(base_path, exist_ok=True)

    '''Global Results'''
    filepath = os.path.join(base_path, "weighted_avg_loss.csv")
    list_to_csv(l=results["weighted_avg_loss"], filepath=filepath)

fit_results = None
collect_fit_results_call_tracker = None
def collect_fit_results(metrics_dict: dict[str, float], context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    hf_dataset = context.run_config["dataset"]

    global collect_fit_results_call_tracker
    if collect_fit_results_call_tracker is None:
        collect_fit_results_call_tracker = 0
    collect_fit_results_call_tracker += 1

    global fit_results
    if fit_results is None:
        fit_results = {}

    for k, v in metrics_dict.items():
        if k not in fit_results:
            fit_results[k] = []
        fit_results[k].append(v)

    if collect_fit_results_call_tracker >= num_rounds:
        process_fit_results(results=fit_results, dataset=hf_dataset)

def get_fit_metrics_aggregation_fn(context: Context):
    def fit_metrics_aggregation_fn(metrics: list[tuple[int, dict[str, bool | bytes | float | int | str]]]):
        ret_dict = {'weighted_avg_loss': 0.0}
        total_examples = 0
        for num_examples, m in metrics:

            loss = m['loss']
            ret_dict['weighted_avg_loss'] += loss * num_examples
            total_examples += num_examples

        ret_dict['weighted_avg_loss'] = ret_dict['weighted_avg_loss'] / total_examples

        collect_fit_results(ret_dict, context)
        return ret_dict
    return fit_metrics_aggregation_fn


def get_evaluate_metrics_aggregation_fn(context: Context):
    def evaluate_metrics_aggregation_fn(metrics: list[tuple[int, dict[str, bool | bytes | float | int | str]]]):
        ret_dict = {'weighted_avg_accuracy': 0.0,
                    'weighted_avg_loss': 0.0,
                    'weighted_avg_recall_score': 0.0,
                    'weighted_avg_precision_score': 0.0,
                    'weighted_avg_f1_score': 0.0}
        total_examples = 0
        for num_examples, m in metrics:

            ret_dict['weighted_avg_accuracy'] += m['accuracy'] * num_examples

            ret_dict['weighted_avg_loss'] += m['loss'] * num_examples

            ret_dict['weighted_avg_precision_score'] += m["precision_score"] * num_examples

            ret_dict['weighted_avg_recall_score'] += m["recall_score"] * num_examples

            ret_dict['weighted_avg_f1_score'] += m["f1_score"] * num_examples

            total_examples += num_examples

        ret_dict['weighted_avg_accuracy'] /= total_examples
        ret_dict['weighted_avg_loss'] /= total_examples
        ret_dict['weighted_avg_precision_score'] /= total_examples
        ret_dict['weighted_avg_recall_score'] /= total_examples
        ret_dict['weighted_avg_f1_score'] /= total_examples

        collect_evaluate_results(ret_dict, context)
        return ret_dict
    return evaluate_metrics_aggregation_fn



class WeightedFedAvg(FedAvg):
    def __init__(self, fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None, **kwargs):
        super().__init__(**kwargs)
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.current_parameters = parameters_to_ndarrays(self.initial_parameters)
        self.clients = {}
        self.role_distribution = {}
        self.anomaly_rate = None
        self.conv = 0
        self.last_two_updates = (None, None)
        self.reputation_weights = (1, 1, 1)
        # Hyperparameters
        self.reliability_threshold = 0.5
        self.alpha = 0.7
        self.beta = 0.6
        self.anomaly_threshold = 5.99
        self.penalty_severity = 2
        self.gamma = 0.3
        self.delta = 0.4
        self.recovery = 0.05
        self.decay = 0.15

    def aggregate_fit(self, server_round, results, failures):
        now = time.time()
        updates = {}

        # Client Registration
        for client, fit_res in results:
            if client.cid not in self.clients.keys():
                client_reputation = ClientReputation(cid=client.cid, num_examples=fit_res.num_examples)
                self.clients[client.cid] = client_reputation
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            update_vector = np.concatenate([arr.ravel() for arr in ndarrays])
            updates[client.cid] = update_vector
            self.clients[client.cid].participations += 1
            self.clients[client.cid].participation_rate = self.clients[client.cid].participations / server_round
        print(f"Registered {len(results)} clients!")

        # Update Scores
        rp = GaussianRandomProjection(n_components=50)
        cids = list(updates.keys())
        update_matrix = np.stack([updates[cid] for cid in cids])
        reduced_update_matrix = rp.fit_transform(update_matrix)
        mean = np.mean(reduced_update_matrix, axis=0)

        cov = np.cov(reduced_update_matrix.T)
        epsilon = 1e-6
        cov_reg = cov + epsilon * np.eye(cov.shape[0])
        inv_covariance = np.linalg.inv(cov_reg)

        anomaly_count = 0
        for client, fit_res in results:
            client_idx = cids.index(client.cid)
            reduced_update_vector = reduced_update_matrix[client_idx, :]
            self.clients[client.cid].update_scores(fit_res=fit_res, reduced_update_vector=reduced_update_vector, now=now, mean=mean, inv_covariance=inv_covariance,
                                                   reliability_threshold=self.reliability_threshold,
                                                   reputation_weights=self.reputation_weights,
                                                   alpha=self.alpha, beta=self.beta,
                                                   anomaly_threshold=self.anomaly_threshold,
                                                   penalty_severity=self.penalty_severity)
            if self.clients[client.cid].reputation_score < self.reliability_threshold / 2:
                anomaly_count += 1

        print("Updated Scores")
        # set anomaly rate
        self.anomaly_rate = anomaly_count / len(results)
        print(f"set anomaly rate: {self.anomaly_rate}")

        # set conv
        total_samples = np.sum([result.num_examples for _, result in results])
        self.conv = np.sum([result.metrics["accuracy"] * result.num_examples for _, result in results]) / total_samples
        print(f"conv: {self.conv}")
        # update reliability_threshold based on conv and anomaly rate
        self.reliability_threshold += self.gamma * self.conv - self.delta * self.anomaly_rate

        print(f"set reliability threshold: {self.reliability_threshold}")
        # Gradient Clipping
        norms = np.array([np.linalg.norm(update) for update in updates.values()])
        c = np.median(norms)
        for cid, update in updates.items():
            norm = np.linalg.norm(update)
            scale = min(1, c / norm)
            updates[cid] = update * scale

        print(f"Gradient Clipping done.")

        # Aggregation
        trusted_suspicious_clients = [self.clients[client.cid] for client, _ in results if self.clients[client.cid].classification != Classification.UNTRUSTED]
        if trusted_suspicious_clients:
            aggregated_params = np.sum([client.reputation_score * client.num_examples * updates[client.cid] for client in trusted_suspicious_clients], axis=0)
            aggregated_params /= np.sum([client.reputation_score * client.num_examples for client in trusted_suspicious_clients])
            sizes = [arr.size for arr in self.current_parameters]
            splits = np.cumsum(sizes)[:-1]
            agg_split = np.split(aggregated_params, splits)
            new_params = [curr + agg.reshape(curr.shape) for curr, agg in zip(self.current_parameters, agg_split)]
            self.current_parameters = new_params
        else:
            new_params = self.current_parameters
        # Metrics Aggregation
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Reputation update for next round (decay, recovery)
        for client, _ in results:
            self.clients[client.cid].reputation_decay_recovery(self.reliability_threshold, self.recovery, self.decay)


        id_changed = 0
        for client, fit_res in results:
            if client.cid not in self.role_distribution:
                self.role_distribution[client.cid] = fit_res.metrics["role"]
            if self.role_distribution[client.cid] != fit_res.metrics["role"]:
                id_changed += 1
                print("Client ID changed.")
        print(f"Malicious clients: {len([v for v in self.role_distribution.values() if v == "Role.MALICIOUS"])}/{len(self.role_distribution)}")
        print(f"Ids changed: {id_changed}")
        return ndarrays_to_parameters(new_params) , metrics_aggregated

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        send_time = time.time()
        """Configure the next round of training."""
        config = {"send_time": send_time}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    hf_dataset = context.run_config["dataset"]

    # Initialize model parameters
    dataset = hf_dataset.split('/')[-1]
    if dataset == 'mnist':
        net = NetMNIST()
    elif dataset == 'cifar10':
        net = Net()
    else:
        net = torchvision.models.resnet18(pretrained=True)
        freeze_model(net)
        net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=10)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = WeightedFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(context),
        evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation_fn(context),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
