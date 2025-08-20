import statistics

import numpy as np
from flwr.common import FitRes, parameters_to_ndarrays
from sklearn import metrics
from enum import Enum
from byebye_badclients.util import mahalanobis_distance

class Classification(Enum):
    TRUSTED = 0
    SUSPICIOUS = 1
    UNTRUSTED = 2

class ClientReputation:
    def __init__(self, cid, num_examples, role, attack_pattern):
        self.cid = cid
        self.num_examples = num_examples
        self.exp_mv_avg = None
        self.performance_consistency_score = 0.5
        self.statistical_anomaly_score = None
        self.temporal_behavior_score = None
        self.reputation_score = None
        self.classification = Classification.SUSPICIOUS
        self.role = role
        self.round_tracker = 0
        self.participations = 0
        self.update_inclusions = 0
        self.participation_rate = None
        self.estimated_round_trip_times = []
        self.attack_pattern = attack_pattern

    def update_scores(self,
                      fit_res: FitRes, reduced_update_vector,
                      now: float,
                      mean, inv_covariance, reliability_threshold,
                      reputation_weights, alpha=0.7, anomaly_threshold=5.99, penalty_severity = 2.0,
                      beta=0.6):

        self.update_round_tracker()

        self.estimated_round_trip_times.append(fit_res.metrics["receive_time"] * 2.0 + fit_res.metrics["train_time"])
        if len(self.estimated_round_trip_times) > 1:
            response_time_variance = statistics.variance(self.estimated_round_trip_times)
        else:
            response_time_variance = 0

        ndarrays = parameters_to_ndarrays(fit_res.parameters)
        update_vector = np.concatenate([arr.ravel() for arr in ndarrays])
        self.update_performance_consistency_score(update=update_vector, alpha=alpha)
        print(f"Performance Consistency Score (Client {self.cid}): {self.performance_consistency_score}")

        distance = mahalanobis_distance(update=reduced_update_vector, mean=mean, inv_covariance=inv_covariance)
        self.update_statistical_anomaly_score(distance=distance, anomaly_threshold=anomaly_threshold,
                                              penalty_severity=penalty_severity)
        print(f"Statistical Anomaly Score (Client {self.cid}): {self.statistical_anomaly_score}")

        self.update_temporal_behaviour_score(beta=beta, response_time_variance=response_time_variance)
        print(f"Temporal Behavior Score (Client {self.cid}): {self.temporal_behavior_score}")

        self.update_reputation_score(reputation_weights=reputation_weights)
        print(f"Reputation Score (Client {self.cid}): {self.reputation_score}")

        self.update_classification(reliability_threshold=reliability_threshold)
        print(f"Classification (Client {self.cid}): {self.classification}")
    def update_round_tracker(self):
        self.round_tracker += 1
        return self.round_tracker

    def update_exponential_moving_average(self, update, alpha=0.5):
        if self.exp_mv_avg is None:
            self.exp_mv_avg = update.copy()
        else:
            self.exp_mv_avg = alpha * update + (1.0 - alpha) * self.exp_mv_avg

    def update_performance_consistency_score(self, update, alpha):
        update_1d = update.flatten()
        self.update_exponential_moving_average(update_1d, alpha=0.5)
        update_2d = update_1d.reshape(1, -1)
        exp_mv_avg_2d = self.exp_mv_avg.reshape(1, -1)

        cos_sim = metrics.pairwise.cosine_similarity(update_2d, exp_mv_avg_2d)[0, 0]

        if self.performance_consistency_score is None:
            self.performance_consistency_score = cos_sim
        else:
            self.performance_consistency_score = alpha * self.performance_consistency_score + (1.0 - alpha) * cos_sim

        return self.performance_consistency_score

    def update_statistical_anomaly_score(self, distance, anomaly_threshold, penalty_severity):
        print(f"Distance: {distance}, Anomaly_threshold: {anomaly_threshold}, penalty_severity: {penalty_severity}")
        if distance <= anomaly_threshold:
            self.statistical_anomaly_score = 1.0
        else:
            self.statistical_anomaly_score = np.exp(-penalty_severity * (distance - anomaly_threshold))

    def update_temporal_behaviour_score(self, beta, response_time_variance):
        update_inclusion_rate = self.update_inclusions / self.participations
        self.temporal_behavior_score = beta * update_inclusion_rate + (1 - beta) * (1.0 / (1.0 + response_time_variance))
        # self.temporal_behavior_score = beta * self.participation_rate + (1.0 - beta) * 1.0 / (1.0 + response_time_variance)
        return self.temporal_behavior_score

    def update_reputation_score(self, reputation_weights):
        scores = [
            self.performance_consistency_score,
            self.statistical_anomaly_score,
            self.temporal_behavior_score,
        ]
        self.reputation_score = np.dot(reputation_weights, scores)
        return self.reputation_score

    def reputation_decay_recovery(self, recovery, decay):
        if self.classification == Classification.TRUSTED:
            self.reputation_score = min(self.reputation_score + recovery, 1.0)
        else:
            self.reputation_score = max(self.reputation_score - decay, 0.0)
        return self.reputation_score

    def update_classification(self, reliability_threshold):
        if self.reputation_score >= reliability_threshold:
            self.classification = Classification.TRUSTED
            self.update_inclusions += 1
        elif self.reputation_score < reliability_threshold / 2.0:
            self.classification = Classification.UNTRUSTED
        else:
            self.classification = Classification.SUSPICIOUS
            self.update_inclusions += self.reputation_score / reliability_threshold
        return self.classification
