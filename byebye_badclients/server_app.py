"""ByeBye-BadClients: A Flower / PyTorch app."""
import os
import time
from typing import Optional

import numpy as np

from flwr.common import Context, ndarrays_to_parameters, Parameters, FitIns, parameters_to_ndarrays, \
    MetricsAggregationFn
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common.logger import log
from logging import WARNING

from sklearn.exceptions import UndefinedMetricWarning

from byebye_badclients.client_app import Role
from byebye_badclients.result_processing import list_to_csv, robustness_metric, hard_rate_metric, dict_to_csv, \
    soft_target_exclusion_rate_metric, soft_target_inclusion_rate_metric
from byebye_badclients.task import get_weights
from byebye_badclients.client_reputation import ClientReputation, Classification
from sklearn.decomposition import PCA

from byebye_badclients.util import load_model
from sklearn.covariance import LedoitWolf

import warnings
# Ignore deprecation warnings from datasets/dill
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Completely ignore sklearn undefined metric warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def process_evaluate_results(results: dict[str, list[float]]):
    base_path = os.path.abspath(f"plots/performance/")
    os.makedirs(base_path, exist_ok=True)

    '''Model Performance Results'''

    filepath = os.path.join(base_path, "weighted_avg_evaluation_loss.csv")
    list_to_csv(l=results["weighted_avg_loss"], filepath=filepath, column_name="weighted_avg_loss")

    filepath = os.path.join(base_path, "accuracy.csv")
    list_to_csv(l=results["weighted_avg_accuracy"], filepath=filepath, column_name="weighted_avg_accuracy")

    filepath = os.path.join(base_path, "recall_scores.csv")
    list_to_csv(l=results["weighted_avg_recall_score"], filepath=filepath, column_name="weighted_avg_recall_score")

    filepath = os.path.join(base_path, "precision_scores.csv")
    list_to_csv(l=results["weighted_avg_precision_score"], filepath=filepath, column_name="weighted_avg_precision_score")

    filepath = os.path.join(base_path, "f1-scores.csv")
    list_to_csv(l=results["weighted_avg_f1_score"], filepath=filepath, column_name="weighted_avg_f1_score")


evaluate_results = None
collect_evaluate_results_call_tracker = None
def collect_evaluate_results(metrics_dict: dict[str, float], context: Context):
    num_rounds = context.run_config["num-server-rounds"]

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
        process_evaluate_results(results=evaluate_results)

def process_fit_results(results: dict[str, list[float]]):
    base_path = os.path.abspath(f"plots/performance")
    os.makedirs(base_path, exist_ok=True)

    '''Global Results'''
    filepath = os.path.join(base_path, "weighted_avg_loss.csv")
    list_to_csv(l=results["weighted_avg_loss"], filepath=filepath, column_name="weighted_avg_loss")

fit_results = None
collect_fit_results_call_tracker = None
def collect_fit_results(metrics_dict: dict[str, float], context: Context):
    num_rounds = context.run_config["num-server-rounds"]

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
        process_fit_results(results=fit_results)

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
    def __init__(self, server_rounds, fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
                 base_reliability_threshold=0.5, alpha=0.7, beta=0.6, anomaly_threshold=5.99, penalty_severity=5,
                 gamma=0.3, delta=0.4, recovery=0.05, decay=0.15, **kwargs):
        super().__init__(**kwargs)
        self.server_rounds = server_rounds
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.current_parameters = parameters_to_ndarrays(self.initial_parameters)
        self.clients = {}
        self.role_distribution = {}
        self.anomaly_rate = None
        self.conv = 0
        self.last_two_updates = (None, None)
        self.reputation_weights = (0.3, 0.3, 0.3)
        self.reliability_threshold = base_reliability_threshold
        # Hyperparameters
        self.base_reliability_threshold = base_reliability_threshold
        self.alpha = alpha
        self.beta = beta
        self.anomaly_threshold = anomaly_threshold
        self.penalty_severity = penalty_severity
        self.gamma = gamma
        self.delta = delta
        self.recovery = recovery
        self.decay = decay
        # Metrics
        self.robustness_score = {}
        self.num_trusted_clients = {}
        self.num_untrusted_clients = {}
        self.num_suspicious_clients = {}
        self.soft_malicious_exclusion_rate = {}
        self.hard_malicious_exclusion_rate = {}
        self.soft_malicious_inclusion_rate = {}
        self.hard_malicious_inclusion_rate = {}
        self.soft_benign_inclusion_rate = {}
        self.hard_benign_inclusion_rate = {}
        self.soft_benign_exclusion_rate = {}
        self.hard_benign_exclusion_rate = {}
    def aggregate_fit(self, server_round, results, failures):
        now = time.time()

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        self.num_trusted_clients[server_round] = 0
        self.num_untrusted_clients[server_round] = 0
        self.num_suspicious_clients[server_round] = 0

        # Client Registration
        updates = {}
        for client, fit_res in results:
            if client.cid not in self.clients.keys():
                client_reputation = ClientReputation(cid=client.cid, num_examples=fit_res.num_examples, role=fit_res.metrics["role"], attack_pattern=fit_res.metrics["attack_pattern"])
                self.clients[client.cid] = client_reputation
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            update_vector = np.concatenate([arr.ravel() for arr in ndarrays])
            updates[client.cid] = update_vector
            self.clients[client.cid].participations += 1
            self.clients[client.cid].participation_rate = self.clients[client.cid].participations / server_round

        # Update Scores

        anomaly_count = 0
        pca = PCA(n_components=8)
        cids = list(updates.keys())
        update_matrix = np.stack([updates[cid] for cid in cids])
        reduced_update_matrix = pca.fit_transform(update_matrix)

        for client, fit_res in results:
            print(f"Behavior: {fit_res.metrics["role"]}" + (f" Attack Pattern: {self.clients[client.cid].attack_pattern}" if
                  self.clients[client.cid].role == str(Role.MALICIOUS) else ""))
            others_reduced_update_matrix = np.delete(reduced_update_matrix, cids.index(client.cid), axis=0)
            mean = np.mean(others_reduced_update_matrix, axis=0)

            lw = LedoitWolf()
            lw.fit(others_reduced_update_matrix)
            cov = lw.covariance_
            inv_covariance = lw.precision_

            client_idx = cids.index(client.cid)
            reduced_update_vector = reduced_update_matrix[client_idx, :]
            self.clients[client.cid].update_scores(fit_res=fit_res, reduced_update_vector=reduced_update_vector, now=now, mean=mean, inv_covariance=inv_covariance,
                                                   reliability_threshold=self.reliability_threshold,
                                                   reputation_weights=self.reputation_weights,
                                                   alpha=self.alpha, beta=self.beta,
                                                   anomaly_threshold=self.anomaly_threshold,
                                                   penalty_severity=self.penalty_severity)
            if self.clients[client.cid].classification == Classification.TRUSTED:
                self.num_trusted_clients[server_round] += 1
            elif self.clients[client.cid].classification == Classification.SUSPICIOUS:
                self.num_suspicious_clients[server_round] += 1
            else:
                self.num_untrusted_clients[server_round] += 1
                anomaly_count += 1
            print("------------------------------------------------" + "\n"
                  "------------------------------------------------")
        # Success Rate related Metrics
        client_ids = list(updates.keys())
        self.handle_robustness_metrics(server_round, client_ids)

        print("Reliability Threshold: ", self.reliability_threshold)

        # set anomaly rate
        self.anomaly_rate = anomaly_count / len(results)

        # set conv
        total_samples = np.sum([result.num_examples for _, result in results])
        self.conv = np.sum([result.metrics["accuracy"] * result.num_examples for _, result in results]) / total_samples

        # update reliability_threshold based on conv and anomaly rate
        self.reliability_threshold = self.base_reliability_threshold + self.gamma * self.conv - self.delta * self.anomaly_rate
        print("Updated Reliability Threshold: ", self.reliability_threshold)
        # Gradient Clipping
        norms = np.array([np.linalg.norm(update) for update in updates.values()])
        c = np.median(norms)
        for cid, update in updates.items():
            norm = np.linalg.norm(update)
            scale = min(1, c / norm)
            updates[cid] = update * scale

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
            self.clients[client.cid].reputation_decay_recovery(recovery=self.recovery, decay=self.decay)

        aggregation_time = time.time()

        return ndarrays_to_parameters(new_params) , metrics_aggregated

    def handle_robustness_metrics(self, server_round, client_ids):
        # Calculate Robustness Score
        self.robustness_score[server_round] = robustness_metric(
            {cid: client for cid, client in self.clients.items() if client.cid in client_ids}, self.reliability_threshold)

        # Calculate Soft Malicious Exclusion Rate (including Suspicious ones partially)
        malicious_clients = {cid: client for cid, client in self.clients.items() if
                             client.cid in client_ids and client.role == str(Role.MALICIOUS)}
        self.soft_malicious_exclusion_rate[server_round] = soft_target_exclusion_rate_metric(malicious_clients, self.reliability_threshold)

        len_malicious_clients = len(malicious_clients)

        # Calculate Hard Malicious Exclusion Rate
        num_untrusted_malicious_clients = len(
            {client for _, client in malicious_clients.items() if client.classification == Classification.UNTRUSTED})
        self.hard_malicious_exclusion_rate[server_round] = hard_rate_metric(len_malicious_clients,
                                                                        num_untrusted_malicious_clients)

        # Calculate Hard Malicious Inclusion Rate
        num_trusted_malicious_clients = len(
            {client for _, client in malicious_clients.items() if client.classification == Classification.TRUSTED})
        self.hard_malicious_inclusion_rate[server_round] = hard_rate_metric(len_malicious_clients, num_trusted_malicious_clients)

        # Calculate Soft Malicious Inclusion Rate
        self.soft_malicious_inclusion_rate[server_round] = soft_target_inclusion_rate_metric(malicious_clients, self.reliability_threshold)

        # Calculate Soft Benign Inclusion Rate (including Suspicious ones partially)
        benign_clients = {cid: client for cid, client in self.clients.items() if
                          client.cid in client_ids and client.role == str(Role.BENIGN)}
        self.soft_benign_inclusion_rate[server_round] = soft_target_inclusion_rate_metric(benign_clients,
                                                                                          self.reliability_threshold)
        len_benign_clients = len(benign_clients)
        # Calculate Hard Benign Inclusion Rate
        num_trusted_benign_clients = len(
            {client for _, client in benign_clients.items() if client.classification == Classification.TRUSTED})
        self.hard_benign_inclusion_rate[server_round] = hard_rate_metric(len_benign_clients, num_trusted_benign_clients)

        # Calculate Soft Benign Inclusion Rate (including Suspicious ones partially)
        self.soft_benign_exclusion_rate[server_round] = soft_target_exclusion_rate_metric(benign_clients, self.reliability_threshold)

        # Calculate Hard Benign Exclusion Rate
        num_untrusted_benign_clients = len(
            {client for _, client in benign_clients.items() if client.classification == Classification.UNTRUSTED}
        )
        self.hard_benign_exclusion_rate[server_round] = hard_rate_metric(len_benign_clients, num_untrusted_benign_clients)

        if server_round == self.server_rounds:
            self.robustness_metrics_to_csvs()

    def robustness_metrics_to_csvs(self):
        base_path = os.path.abspath(f"plots/robustness/")
        os.makedirs(base_path, exist_ok=True)

        filepath = os.path.join(base_path, "robustness_score.csv")
        dict_to_csv(d=self.robustness_score, filepath=filepath, column_name="robustness_score")

        filepath = os.path.join(base_path, "hard_malicious_detection_rate.csv")
        dict_to_csv(d=self.hard_malicious_exclusion_rate, filepath=filepath, column_name="hard_malicious_detection_rate")

        filepath = os.path.join(base_path, "soft_malicious_detection_rate.csv")
        dict_to_csv(d=self.soft_malicious_exclusion_rate, filepath=filepath, column_name="soft_malicious_detection_rate")

        filepath = os.path.join(base_path, "soft_malicious_inclusion_rate.csv")
        dict_to_csv(d=self.soft_malicious_inclusion_rate, filepath=filepath, column_name="soft_malicious_inclusion_rate")

        filepath = os.path.join(base_path, "hard_malicious_inclusion_rate.csv")
        dict_to_csv(d=self.hard_malicious_inclusion_rate, filepath=filepath, column_name="hard_malicious_inclusion_rate")

        filepath = os.path.join(base_path, "hard_benign_inclusion_rate.csv")
        dict_to_csv(d=self.hard_benign_inclusion_rate, filepath=filepath, column_name="hard_benign_inclusion_rate")

        filepath = os.path.join(base_path, "soft_benign_inclusion_rate.csv")
        dict_to_csv(d=self.soft_benign_inclusion_rate, filepath=filepath, column_name="soft_benign_inclusion_rate")

        filepath = os.path.join(base_path, "soft_benign_exclusion_rate.csv")
        dict_to_csv(d=self.soft_benign_exclusion_rate, filepath=filepath, column_name="soft_benign_exclusion_rate")

        filepath = os.path.join(base_path, "hard_benign_exclusion_rate.csv")
        dict_to_csv(d=self.hard_benign_exclusion_rate, filepath=filepath, column_name="hard_benign_exclusion_rate")

        filepath = os.path.join(base_path, "num_untrusted_clients.csv")
        dict_to_csv(d=self.num_untrusted_clients, filepath=filepath, column_name="num_untrusted_clients")

        filepath = os.path.join(base_path, "num_trusted_clients.csv")
        dict_to_csv(d=self.num_trusted_clients, filepath=filepath, column_name="num_trusted_clients")

        filepath = os.path.join(base_path, "num_suspicious_clients.csv")
        dict_to_csv(d=self.num_suspicious_clients, filepath=filepath, column_name="num_suspicious_clients")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        send_time = time.time()
        config = {"send_time": send_time}

        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.all()
        client_map = {client.cid: client for _, client in clients.items()}
        client_reputation = {}
        for cid, client in client_map.items():
            if cid not in self.clients:
                client_reputation[cid] = 0.5
            else:
                client_reputation[cid] = self.clients[cid].reputation_score
                if client_reputation[cid] == 0:
                    client_reputation[cid] += 0.01

        def reputations_to_weights(reputations):
            return [reputation / np.sum(reputations) for reputation in reputations]

        choices = list(client_reputation.keys())
        weights = reputations_to_weights(list(client_reputation.values()))
        sampled_client_keys = np.random.choice(a=choices, size=sample_size, p=weights, replace=False)
        sampled_clients = [client for str, client in clients.items() if client.cid in sampled_client_keys]
        return [(client, fit_ins) for client in sampled_clients]

def server_fn(context: Context):
    # Read from config
    no_defense_fedavg = context.run_config["no-defense-fedavg"]
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    hf_dataset = context.run_config["dataset"]
    base_reliability_threshold = context.run_config["base-reliability-threshold"]
    alpha = context.run_config["alpha"]
    beta = context.run_config["beta"]
    anomaly_threshold = context.run_config["anomaly_threshold"]
    penalty_severity = context.run_config["penalty_severity"]
    gamma = context.run_config["gamma"]
    delta = context.run_config["delta"]
    recovery = context.run_config["recovery"]
    decay = context.run_config["decay"]

    # Initialize model parameters
    dataset = hf_dataset.split('/')[-1]
    net = load_model(dataset)

    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    if no_defense_fedavg:
        strategy = FedAvg(fraction_fit=fraction_fit,
                          initial_parameters=parameters,
                          fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(context),
                          evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation_fn(context),
                          on_fit_config_fn=lambda server_round: {"send_time": time.time()})
    else:
        strategy = WeightedFedAvg(
            server_rounds=num_rounds,
            base_reliability_threshold=base_reliability_threshold,
            alpha=alpha, beta=beta, anomaly_threshold=anomaly_threshold,
            penalty_severity=penalty_severity,
            gamma=gamma, delta=delta, recovery=recovery, decay=decay,
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
