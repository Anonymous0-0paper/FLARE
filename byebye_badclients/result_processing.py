import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

def get_precision(labels_over_epochs: list[list[int]], y_pred_over_epochs: list[list[int]]) -> list[float]:
    return [sk.metrics.precision_score(y_true=labels, y_pred=y_pred, average=None)
            for y_pred, labels in zip(y_pred_over_epochs, labels_over_epochs)]


def get_recall(labels_over_epochs: list[list[int]], y_pred_over_epochs: list[list[int]]) -> list[float]:
    return [sk.metrics.recall_score(y_true=labels, y_pred=y_pred, average=None)
            for y_pred, labels in zip(y_pred_over_epochs, labels_over_epochs)]


def get_f1score(labels_over_epochs: list[list[int]], y_pred_over_epochs: list[list[int]]) -> list[float]:
    return [sk.metrics.f1_score(y_true=labels, y_pred=y_pred, average=None)
            for y_pred, labels in zip(y_pred_over_epochs, labels_over_epochs)]


def get_accuracy_score(labels_over_epochs: list[list[int]], y_pred_over_epochs: list[list[int]]) -> list[float]:
    return [sk.metrics.accuracy_score(y_true=labels, y_pred=y_pred, normalize=True)
            for y_pred, labels in zip(y_pred_over_epochs, labels_over_epochs)]

def calc_avg(results: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(value)) for key, value in results.items()}

def list_accumulation(l: list[float]) -> list[float]:
    summ = 0
    ret = []
    for val in l:
        ret.append(val + summ)
        summ += val
    return ret

def per_second_calculation(lis: list[float]) -> list[float]:
    return [1/l for l in lis]

def calc_accumulated_latency(results: dict[str, list[float]]) -> dict[str, list[float]]:
    return {key: list_accumulation(value) for key, value in results.items()}

def calc_potential_responses_per_second(results: dict[str, list[float]]) -> dict[str, list[float]]:
    return {key: per_second_calculation(value) for key, value in results.items()}

def calc_potential_responses_per_second_average(results: dict[str, list[float]]) -> dict[str, float]:
    return {key: 1/float(np.mean(value)) for key, value in results.items()}

def dict_to_line_plot(dict: dict[str, list[float]], title, filepath, xlabel='Epoch', ylabel='Value') -> None:
    df = pd.DataFrame.from_dict(dict)

    df.plot(x=None, y=list(dict.keys()), kind='line')
    plt.plot(np.mean([np.mean(l) for l in dict.values()]), linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()

def dict_to_bar_plot(dict: dict[str, float], title, filepath, xlabel='Epoch', ylabel='Value') -> None:
    df = pd.DataFrame(list(dict.items()), columns=["Client", "Latency"])

    df.plot(x="Client", y="Latency", kind="bar", legend=False)
    plt.plot(np.mean([np.mean(l) for l in dict.values()]), linestyle='--')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def list_to_line_plot(l: list[float], title, filepath, xlabel='Epoch', ylabel='Value') -> None:
    plt.plot(l, marker='o')
    plt.axhline(float(np.mean(l)), color='r', linestyle='--', label='Mean')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def list_to_bar_plot(l: list[float], title, filepath, xlabel='Epoch', ylabel='Value') -> None:
    plt.bar(range(len(l)), l)
    plt.axhline(float(np.mean(l)), color='r', linestyle='--', label='Mean')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()

def list_to_csv(l, filepath) -> None:
    df = pd.DataFrame(l, columns=["Latency"])
    df.to_csv(filepath, index_label="Index")
