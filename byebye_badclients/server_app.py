"""ByeBye-BadClients: A Flower / PyTorch app."""
import json
import os
import time
from typing import Optional, Union

import numpy as np

from flwr.common import Context, ndarrays_to_parameters, Parameters, FitIns, parameters_to_ndarrays, \
    MetricsAggregationFn, FitRes, Scalar
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
from sklearn.covariance import MinCovDet
import warnings
# Ignore deprecation warnings from datasets/dill
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Completely ignore sklearn undefined metric warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def process_evaluate_results(results: dict[str, list[float]]):
    base_path = os.path.abspath(f"results/performance/")
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
    base_path = os.path.abspath(f"results/performance")
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


def analyze_pattern(rep_var):
    highest_variance_score = np.argmax(rep_var)
    if highest_variance_score == 0:
        return 'label-flipping'
    elif highest_variance_score == 1:
        return 'update-manipulation'
    else:
        return 'adaptive-attack'

class FedAvgWrapper(FedAvg):
    def __init__(self, server_rounds, **kwargs):
        super().__init__(**kwargs)
        self.aggregation_times = {}
        self.server_rounds = server_rounds
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        aggregate_start = time.time()
        res = super().aggregate_fit(server_round=server_round, results=results, failures=failures)
        aggregate_end = time.time()

        self.aggregation_times[server_round] = aggregate_end - aggregate_start
        if server_round == self.server_rounds:
            base_path = "results/latency/"
            os.makedirs(base_path, exist_ok=True)

            filepath = os.path.join(base_path, "aggregation_time.csv")
            dict_to_csv(self.aggregation_times, filepath=filepath, column_name="Aggregation Time (s)")
        return res

class WeightedFedAvg(FedAvg):
    def __init__(self, server_rounds, fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
                 base_reliability_threshold=0.5, alpha=0.7, beta=0.6, anomaly_threshold=5.99, penalty_severity=5,
                 gamma=0.3, delta=0.4, recovery=0.05, decay=0.15, w1=0.3, w2=0.4, w3=0.3, late_training_threshold=0.6, **kwargs):
        super().__init__(**kwargs)
        self.server_rounds = server_rounds
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.current_parameters = parameters_to_ndarrays(self.initial_parameters)
        self.clients = {}
        self.role_distribution = {}
        self.anomaly_rate = None
        self.conv = 0
        self.last_two_updates = (None, None)
        self.reputation_weights = (w1, w2, w3)
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
        self.late_training_threshold = late_training_threshold
        # Metrics Round-scope
        self.robustness_score_round = {}
        self.num_trusted_clients_round = {}
        self.num_untrusted_clients_round = {}
        self.num_suspicious_clients_round = {}
        self.soft_malicious_exclusion_rate_round = {}
        self.hard_malicious_exclusion_rate_round = {}
        self.soft_malicious_inclusion_rate_round = {}
        self.hard_malicious_inclusion_rate_round = {}
        self.soft_benign_inclusion_rate_round = {}
        self.hard_benign_inclusion_rate_round = {}
        self.soft_benign_exclusion_rate_round = {}
        self.hard_benign_exclusion_rate_round = {}
        # Metrics Global-scope
        self.robustness_score_global = {}
        self.num_trusted_clients_global = {}
        self.num_untrusted_clients_global = {}
        self.num_suspicious_clients_global = {}
        self.soft_malicious_exclusion_rate_global = {}
        self.hard_malicious_exclusion_rate_global = {}
        self.soft_malicious_inclusion_rate_global = {}
        self.hard_malicious_inclusion_rate_global = {}
        self.soft_benign_inclusion_rate_global = {}
        self.hard_benign_inclusion_rate_global = {}
        self.soft_benign_exclusion_rate_global = {}
        self.hard_benign_exclusion_rate_global = {}
        self.aggregation_times = {}

    def aggregate_fit(self, server_round, results, failures):
        aggregation_start = time.time()

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        self.num_trusted_clients_round[server_round] = 0
        self.num_untrusted_clients_round[server_round] = 0
        self.num_suspicious_clients_round[server_round] = 0
        self.num_trusted_clients_global[server_round] = 0
        self.num_untrusted_clients_global[server_round] = 0
        self.num_suspicious_clients_global[server_round] = 0

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
        pca = PCA(n_components=max(int(len(results)/10), 1))
        cids = list(updates.keys())
        update_matrix = np.stack([updates[cid] for cid in cids])
        reduced_update_matrix = pca.fit_transform(update_matrix)

        mcd = MinCovDet(random_state=server_round)
        mcd_matrix = mcd.fit(reduced_update_matrix)
        mcd_mean = mcd_matrix.location_
        mcd_inv_cov = mcd_matrix.precision_
        for client, fit_res in results:
            print(f"Behavior: {fit_res.metrics["role"]}" + (f" Attack Pattern: {self.clients[client.cid].attack_pattern}" if
                  self.clients[client.cid].role == str(Role.MALICIOUS) else ""))
            # others_reduced_update_matrix = np.delete(reduced_update_matrix, cids.index(client.cid), axis=0) if len(results) >= 2 else reduced_update_matrix
            # mean = np.mean(others_reduced_update_matrix, axis=0)

            # lw = LedoitWolf()
            # lw.fit(others_reduced_update_matrix)
            # inv_covariance = lw.precision_

            client_idx = cids.index(client.cid)
            reduced_update_vector = reduced_update_matrix[client_idx, :]
            self.clients[client.cid].update_scores(fit_res=fit_res, reduced_update_vector=reduced_update_vector, now=aggregation_start, mean=mcd_mean, inv_covariance=mcd_inv_cov,
                                                   reliability_threshold=self.reliability_threshold,
                                                   reputation_weights=self.reputation_weights,
                                                   alpha=self.alpha, beta=self.beta,
                                                   anomaly_threshold=self.anomaly_threshold,
                                                   penalty_severity=self.penalty_severity)
            if self.clients[client.cid].classification == Classification.TRUSTED:
                self.num_trusted_clients_round[server_round] += 1
            elif self.clients[client.cid].classification == Classification.SUSPICIOUS:
                self.num_suspicious_clients_round[server_round] += 1
            else:
                self.num_untrusted_clients_round[server_round] += 1
                anomaly_count += 1

            print("------------------------------------------------\n" +
                  "\n" +
                  "------------------------------------------------")

        for cid, client in self.clients.items():
            if client.classification == Classification.TRUSTED:
                self.num_trusted_clients_global[server_round] += 1
            elif client.classification == Classification.SUSPICIOUS:
                self.num_suspicious_clients_global[server_round] += 1
            else:
                self.num_untrusted_clients_global[server_round] += 1

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

       # Dynamic adjustment of reputation mixing coefficients
        self.dynamic_reputation_weight_computation()

        # Gradient Clipping
        norms = np.array([np.linalg.norm(update) for update in updates.values()])
        c = np.median(norms)
        for cid, update in updates.items():
            norm = np.linalg.norm(update)
            scale = min(1, c / norm)
            updates[cid] = update * scale

        # Aggregation
        trusted_suspicious_clients = []
        for client, fit_res in results:
            if self.clients[client.cid].classification != Classification.UNTRUSTED:
                trusted_suspicious_clients.append(self.clients[client.cid])
                if self.clients[client.cid].classification == Classification.SUSPICIOUS:
                    ndarrays = parameters_to_ndarrays(fit_res.parameters)
                    factor = self.clients[client.cid].reputation_score * self.reliability_threshold
                    ndarrays = [arr * factor for arr in ndarrays]
                    fit_res.parameters = ndarrays_to_parameters(ndarrays)

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


        aggregation_end = time.time()
        self.aggregation_times[server_round] = aggregation_end - aggregation_start
        if server_round == self.server_rounds:
            base_path = "results/latency/"
            os.makedirs(base_path, exist_ok=True)

            filepath = os.path.join(base_path, "aggregation_time.csv")
            dict_to_csv(self.aggregation_times, filepath=filepath, column_name="Aggregation Time (s)")
        return ndarrays_to_parameters(new_params) , metrics_aggregated

    def dynamic_reputation_weight_computation(self):

        all_reps = []
        trusted_reps = []
        untrusted_reps = []

        # Single pass to build all necessary lists
        for _, client in self.clients.items():
            scores = (client.performance_consistency_score, client.statistical_anomaly_score,
                      client.temporal_behavior_score)
            all_reps.append(scores)
            if client.classification == Classification.UNTRUSTED:
                untrusted_reps.append(scores)
            else:
                trusted_reps.append(scores)

        rep_var = np.var(all_reps, axis=0)

        mean_sus = np.mean(untrusted_reps, axis=0) if untrusted_reps else np.zeros(3)
        mean_trusted = np.mean(trusted_reps, axis=0) if trusted_reps else np.zeros(3)

        sep = np.abs(mean_trusted - mean_sus)

        priorities = sep * rep_var

        if self.conv > self.late_training_threshold:
            priorities[0] *= 1.5
            priorities[2] *= 1.2
        else:
            priorities[1] *= 1.3
            priorities[0] *= 0.8

        attack_pattern = analyze_pattern(rep_var)

        if attack_pattern == 'update-manipulation': # includes 'update-scaling', 'random-update'
            priorities[1] *= 2.0
        elif attack_pattern == 'adaptive-attack':
            priorities[2] *= 2.0
        else:
            priorities[0] *= 1.8

        exp_priorities = np.exp(priorities)
        weights = exp_priorities / np.sum(exp_priorities)

        self.reputation_weights = 0.7 * weights + 0.3 * np.array(self.reputation_weights)
        print(f"Reputation Weights: {self.reputation_weights}")

    def handle_robustness_metrics(self, server_round, client_ids):
        # Calculate Robustness Score
        self.robustness_score_round[server_round] = robustness_metric(
            {cid: client for cid, client in self.clients.items() if client.cid in client_ids}, self.reliability_threshold)

        # Calculate Soft Malicious Exclusion Rate (including Suspicious ones partially)
        malicious_clients = {cid: client for cid, client in self.clients.items() if
                             client.cid in client_ids and client.role == str(Role.MALICIOUS)}
        self.soft_malicious_exclusion_rate_round[server_round] = soft_target_exclusion_rate_metric(malicious_clients, self.reliability_threshold)

        len_malicious_clients = len(malicious_clients)

        # Calculate Hard Malicious Exclusion Rate
        num_untrusted_malicious_clients = len(
            {client for _, client in malicious_clients.items() if client.classification == Classification.UNTRUSTED})
        self.hard_malicious_exclusion_rate_round[server_round] = hard_rate_metric(len_malicious_clients,
                                                                        num_untrusted_malicious_clients)

        # Calculate Hard Malicious Inclusion Rate
        num_trusted_malicious_clients = len(
            {client for _, client in malicious_clients.items() if client.classification == Classification.TRUSTED})
        self.hard_malicious_inclusion_rate_round[server_round] = hard_rate_metric(len_malicious_clients, num_trusted_malicious_clients)

        # Calculate Soft Malicious Inclusion Rate
        self.soft_malicious_inclusion_rate_round[server_round] = soft_target_inclusion_rate_metric(malicious_clients, self.reliability_threshold)

        # Calculate Soft Benign Inclusion Rate (including Suspicious ones partially)
        benign_clients = {cid: client for cid, client in self.clients.items() if
                          client.cid in client_ids and client.role == str(Role.BENIGN)}
        self.soft_benign_inclusion_rate_round[server_round] = soft_target_inclusion_rate_metric(benign_clients,
                                                                                          self.reliability_threshold)
        len_benign_clients = len(benign_clients)
        # Calculate Hard Benign Inclusion Rate
        num_trusted_benign_clients = len(
            {client for _, client in benign_clients.items() if client.classification == Classification.TRUSTED})
        self.hard_benign_inclusion_rate_round[server_round] = hard_rate_metric(len_benign_clients, num_trusted_benign_clients)

        # Calculate Soft Benign Inclusion Rate (including Suspicious ones partially)
        self.soft_benign_exclusion_rate_round[server_round] = soft_target_exclusion_rate_metric(benign_clients, self.reliability_threshold)

        # Calculate Hard Benign Exclusion Rate
        num_untrusted_benign_clients = len(
            {client for _, client in benign_clients.items() if client.classification == Classification.UNTRUSTED}
        )
        self.hard_benign_exclusion_rate_round[server_round] = hard_rate_metric(len_benign_clients, num_untrusted_benign_clients)

        # Calculate Global Robustness Score
        self.robustness_score_global[server_round] = robustness_metric(clients=self.clients, reliability_threshold=self.reliability_threshold)

        # Calculate Global Soft Malicious Exclusion Rate (including Suspicious ones partially)
        global_malicious_clients = {cid: client for cid, client in self.clients.items() if client.role == str(Role.MALICIOUS)}
        self.soft_malicious_exclusion_rate_global[server_round] = soft_target_exclusion_rate_metric(global_malicious_clients, self.reliability_threshold)

        # Calculate Global Hard Malicious Exclusion Rate
        len_global_malicious_clients = len(global_malicious_clients)
        self.hard_malicious_exclusion_rate_global[server_round] = hard_rate_metric(len_global_malicious_clients,
            len([client for _, client in global_malicious_clients.items() if client.classification == Classification.UNTRUSTED]))

        # Calculate Global Hard Malicious Inclusion Rate
        self.hard_malicious_inclusion_rate_global[server_round] = hard_rate_metric(len_global_malicious_clients,
            len([client for _, client in global_malicious_clients.items() if client.classification == Classification.TRUSTED]))

        # Calculate Global Soft Malicious Inclusion Rate (including Suspicious ones partially)
        self.soft_malicious_inclusion_rate_global[server_round] = soft_target_inclusion_rate_metric(global_malicious_clients, self.reliability_threshold)

        # Calculate Global Soft Benign Inclusion Rate (including Suspicious ones partially)
        global_benign_clients = {cid: client for cid, client in self.clients.items() if client.role == str(Role.BENIGN)}
        self.soft_benign_inclusion_rate_global[server_round] = soft_target_inclusion_rate_metric(global_benign_clients, self.reliability_threshold)

        # Calculate Global Hard Benign Inclusion Rate
        len_global_benign_clients = len(global_benign_clients)
        self.hard_benign_inclusion_rate_global[server_round] = hard_rate_metric(len_global_benign_clients,
            len([client for _, client in global_benign_clients.items() if client.classification == Classification.TRUSTED]))

        # Calculate Global Soft Benign Exclusion Rate (including Suspicious ones partially)
        self.soft_benign_exclusion_rate_global[server_round] = soft_target_exclusion_rate_metric(global_benign_clients, self.reliability_threshold)

        # Calculate Global Hard Benign Exclusion Rate
        self.hard_benign_exclusion_rate_global[server_round] = hard_rate_metric(len_global_benign_clients,
            len([client for _, client in global_benign_clients.items() if client.classification == Classification.UNTRUSTED]))

        if server_round == self.server_rounds:
            self.robustness_metrics_to_csvs()

    def robustness_metrics_to_csvs(self):
        # Round
        base_path = os.path.abspath(f"results/robustness/per_round/")
        os.makedirs(base_path, exist_ok=True)

        filepath = os.path.join(base_path, "robustness_score.csv")
        dict_to_csv(d=self.robustness_score_round, filepath=filepath, column_name="robustness score per round")

        filepath = os.path.join(base_path, "hard_malicious_detection_rate.csv")
        dict_to_csv(d=self.hard_malicious_exclusion_rate_round, filepath=filepath, column_name="hard malicious detection rate per round")

        filepath = os.path.join(base_path, "soft_malicious_detection_rate.csv")
        dict_to_csv(d=self.soft_malicious_exclusion_rate_round, filepath=filepath, column_name="soft malicious detection rate per round")

        filepath = os.path.join(base_path, "soft_malicious_inclusion_rate.csv")
        dict_to_csv(d=self.soft_malicious_inclusion_rate_round, filepath=filepath, column_name="soft malicious inclusion rate per round")

        filepath = os.path.join(base_path, "hard_malicious_inclusion_rate.csv")
        dict_to_csv(d=self.hard_malicious_inclusion_rate_round, filepath=filepath, column_name="hard malicious inclusion rate per round")

        filepath = os.path.join(base_path, "hard_benign_inclusion_rate.csv")
        dict_to_csv(d=self.hard_benign_inclusion_rate_round, filepath=filepath, column_name="hard benign inclusion rate per round")

        filepath = os.path.join(base_path, "soft_benign_inclusion_rate.csv")
        dict_to_csv(d=self.soft_benign_inclusion_rate_round, filepath=filepath, column_name="soft benign inclusion rate per round")

        filepath = os.path.join(base_path, "soft_benign_exclusion_rate.csv")
        dict_to_csv(d=self.soft_benign_exclusion_rate_round, filepath=filepath, column_name="soft benign exclusion rate per round")

        filepath = os.path.join(base_path, "hard_benign_exclusion_rate.csv")
        dict_to_csv(d=self.hard_benign_exclusion_rate_round, filepath=filepath, column_name="hard benign exclusion rate per round")

        filepath = os.path.join(base_path, "num_untrusted_clients.csv")
        dict_to_csv(d=self.num_untrusted_clients_round, filepath=filepath, column_name="num untrusted clients per round")

        filepath = os.path.join(base_path, "num_trusted_clients.csv")
        dict_to_csv(d=self.num_trusted_clients_round, filepath=filepath, column_name="num trusted clients per round")

        filepath = os.path.join(base_path, "num_suspicious_clients.csv")
        dict_to_csv(d=self.num_suspicious_clients_round, filepath=filepath, column_name="num suspicious clients per round")

        # Global
        base_path = os.path.abspath(f"results/robustness/overall/")
        os.makedirs(base_path, exist_ok=True)

        filepath = os.path.join(base_path, "robustness_score.csv")
        dict_to_csv(d=self.robustness_score_global, filepath=filepath, column_name="robustness score overall")

        filepath = os.path.join(base_path, "hard_malicious_detection_rate.csv")
        dict_to_csv(d=self.hard_malicious_exclusion_rate_global, filepath=filepath,
                    column_name="hard malicious detection rate overall")

        filepath = os.path.join(base_path, "soft_malicious_detection_rate.csv")
        dict_to_csv(d=self.soft_malicious_exclusion_rate_global, filepath=filepath,
                    column_name="soft malicious detection rate overall")

        filepath = os.path.join(base_path, "soft_malicious_inclusion_rate.csv")
        dict_to_csv(d=self.soft_malicious_inclusion_rate_global, filepath=filepath,
                    column_name="soft malicious inclusion rate overall")

        filepath = os.path.join(base_path, "hard_malicious_inclusion_rate.csv")
        dict_to_csv(d=self.hard_malicious_inclusion_rate_global, filepath=filepath,
                    column_name="hard malicious inclusion rate overall")

        filepath = os.path.join(base_path, "hard_benign_inclusion_rate.csv")
        dict_to_csv(d=self.hard_benign_inclusion_rate_global, filepath=filepath,
                    column_name="hard benign inclusion rate overall")

        filepath = os.path.join(base_path, "soft_benign_inclusion_rate.csv")
        dict_to_csv(d=self.soft_benign_inclusion_rate_global, filepath=filepath,
                    column_name="soft benign inclusion rate overall")

        filepath = os.path.join(base_path, "soft_benign_exclusion_rate.csv")
        dict_to_csv(d=self.soft_benign_exclusion_rate_global, filepath=filepath,
                    column_name="soft benign exclusion rate overall")

        filepath = os.path.join(base_path, "hard_benign_exclusion_rate.csv")
        dict_to_csv(d=self.hard_benign_exclusion_rate_global, filepath=filepath,
                    column_name="hard benign exclusion rate overall")

        filepath = os.path.join(base_path, "num_untrusted_clients.csv")
        dict_to_csv(d=self.num_untrusted_clients_global, filepath=filepath, column_name="num untrusted clients overall")

        filepath = os.path.join(base_path, "num_trusted_clients.csv")
        dict_to_csv(d=self.num_trusted_clients_global, filepath=filepath, column_name="num trusted clients overall")

        filepath = os.path.join(base_path, "num_suspicious_clients.csv")
        dict_to_csv(d=self.num_suspicious_clients_global, filepath=filepath, column_name="num suspicious clients coverall")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        send_time = time.time()

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        config["send_time"] = send_time
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
    base_path = os.path.abspath(f"results/")
    os.makedirs(base_path, exist_ok=True)

    hyperparameters_path = os.path.join(base_path, "config.json")
    with open(hyperparameters_path, "w") as f:
        json.dump(context.run_config, f, indent=2)

    # Read from config
    no_defense_fedavg = context.run_config["no-defense-fedavg"]
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
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
    late_training_threshold = context.run_config["late-training-threshold"]

    # Initialize model parameters
    dataset = hf_dataset.split('/')[-1]
    net = load_model(dataset)

    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    if no_defense_fedavg:
        strategy = FedAvgWrapper(
            server_rounds=num_rounds,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=parameters,
            fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(context),
            evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation_fn(context),
            min_available_clients=1,
            min_fit_clients=1,
            min_evaluate_clients=1,
            on_fit_config_fn=lambda server_round: {"send_time": time.time(), "dataset": dataset},
            on_evaluate_config_fn=lambda server_round: {"dataset": dataset}
        )
    else:
        strategy = WeightedFedAvg(
            server_rounds=num_rounds,
            base_reliability_threshold=base_reliability_threshold,
            alpha=alpha, beta=beta, anomaly_threshold=anomaly_threshold,
            penalty_severity=penalty_severity,
            gamma=gamma, delta=delta, recovery=recovery, decay=decay,
            late_training_threshold=late_training_threshold,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=1,
            min_fit_clients=1,
            min_evaluate_clients=1,
            initial_parameters=parameters,
            fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(context),
            evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation_fn(context),
            on_fit_config_fn=lambda server_round: {"dataset": dataset},
            on_evaluate_config_fn=lambda server_round: {"dataset": dataset}
        )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
