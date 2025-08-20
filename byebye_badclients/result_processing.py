import pandas as pd

from byebye_badclients.client_app import Role
from byebye_badclients.client_reputation import ClientReputation, Classification

def robustness_metric(clients: dict[str, ClientReputation], reliability_threshold):
    score = 0
    for cid, client in clients.items():
        if client.role == str(Role.MALICIOUS):
            if client.classification == Classification.TRUSTED:
                score -= 1
            elif client.classification == Classification.UNTRUSTED:
                score += 1
            else:
                score -= client.reputation_score/reliability_threshold
        else:
            if client.classification == Classification.TRUSTED:
                score += 1
            elif client.classification == Classification.UNTRUSTED:
                score -= 1
            else:
                score += client.reputation_score/reliability_threshold
    len_clients = len(clients)
    return (score + len_clients) / (2 * len_clients)

def hard_rate_metric(num_targets, num_found):
    if num_targets == 0:
        return -1

    return num_found / num_targets

def soft_target_exclusion_rate_metric(target_clients: dict[str, ClientReputation], reliability_threshold):
    partial_exclusion = 0
    full_exclusion = 0
    num_malicious_clients = len(target_clients)

    if num_malicious_clients <= 0:
        return -1

    for cid, client in target_clients.items():
        if client.classification == Classification.SUSPICIOUS:
            partial_exclusion += (1 - client.reputation_score/reliability_threshold)
        elif client.classification == Classification.UNTRUSTED:
            full_exclusion += 1
    return (partial_exclusion + full_exclusion) / num_malicious_clients

def soft_target_inclusion_rate_metric(target_clients: dict[str, ClientReputation], reliability_threshold):
    partial_inclusion = 0
    full_inclusion = 0
    num_benign_clients = len(target_clients)

    if num_benign_clients <= 0:
        return -1

    for cid, client in target_clients.items():
        if client.classification == Classification.SUSPICIOUS:
            partial_inclusion += client.reputation_score / reliability_threshold
        elif client.classification == Classification.TRUSTED:
            full_inclusion += 1
    return (partial_inclusion + full_inclusion) / num_benign_clients

def list_to_csv(l, filepath, column_name: str) -> None:
    df = pd.DataFrame(l, columns=[column_name])
    df.index = range(1, len(df) + 1)
    df.to_csv(filepath, index_label="Round")

def dict_to_csv(d, filepath, column_name: str) -> None:
    df = pd.DataFrame(list(d.items()), columns=["Round", column_name])
    df.to_csv(filepath, index=False)
