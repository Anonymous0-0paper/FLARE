"""ByeBye-BadClients: A Flower / PyTorch app."""
import os
import time

import torchvision
from flwr.common import Context, ndarrays_to_parameters, Parameters, FitIns
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from byebye_badclients.result_processing import list_to_csv, list_to_line_plot, list_accumulation, \
    per_second_calculation
from byebye_badclients.task import Net, NetMNIST, get_weights
from byebye_badclients.client_reputation import ClientReputation

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
collect_evaluate_results_calltracker = None
def collect_evaluate_results(metrics_dict: dict[str, float], context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    hf_dataset = context.run_config["dataset"]

    global collect_evaluate_results_calltracker
    if collect_evaluate_results_calltracker is None:
        collect_evaluate_results_calltracker = 0
    collect_evaluate_results_calltracker += 1

    global evaluate_results
    if evaluate_results is None:
        evaluate_results = {}

    for k, v in metrics_dict.items():
        if k not in evaluate_results:
            evaluate_results[k] = []
        evaluate_results[k].append(v)

    if collect_evaluate_results_calltracker >= num_rounds:
        process_evaluate_results(results=evaluate_results, dataset=hf_dataset)

def process_fit_results(results: dict[str, list[float]], dataset: str):
    dataset_name = dataset.split('/')[-1]
    base_path = os.path.abspath(f"plots/{dataset_name}")
    os.makedirs(base_path, exist_ok=True)

    '''Global Results'''
    filepath = os.path.join(base_path, "weighted_avg_loss.csv")
    list_to_csv(l=results["weighted_avg_loss"], filepath=filepath)

fit_results = None
collect_fit_results_calltracker = None
def collect_fit_results(metrics_dict: dict[str, float], context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    hf_dataset = context.run_config["dataset"]

    global collect_fit_results_calltracker
    if collect_fit_results_calltracker is None:
        collect_fit_results_calltracker = 0
    collect_fit_results_calltracker += 1

    global fit_results
    if fit_results is None:
        fit_results = {}

    for k, v in metrics_dict.items():
        if k not in fit_results:
            fit_results[k] = []
        fit_results[k].append(v)

    if collect_fit_results_calltracker >= num_rounds:
        process_fit_results(results=fit_results, dataset=hf_dataset)

def get_fit_metrics_aggregation_fn(context: Context):
    def fit_metrics_aggregation_fn(metrics: list[tuple[int, dict[str, bool | bytes | float | int | str]]]):
        ret_dict = {'weighted_avg_loss': 0.0}
        total_examples = 0
        for num_examples, m in metrics:

            loss = m['loss']
            ret_dict['weighted_avg_loss'] += loss * num_examples
            total_examples += num_examples

        len_metrics = len(metrics)
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clients = {}

    def aggregate_fit(self, server_round, results, failures):
        now = time.time()
        for client, fit_res in results:
            if client.cid not in self.clients.keys():
                client_reputation = ClientReputation(client.cid)
                self.clients[client.cid] = client_reputation
            # compute Mean, Std and Cov
        for client, fit_res in results:
            self.clients[client.cid].update_scores(fit_res, now)
        if server_round == 1:
            # conv^t = 0
            a = None
        else:
            # conv^t = # Measure stable training
            a = None

        # set anomaly rate

        # update reliability threshold based on conv and anomaly rate

        # Gradient Clipping

        # Aggregation

        # Reputation update for next round (decay, recovery)

        return super().aggregate_fit(server_round, results, failures)

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
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights)
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
