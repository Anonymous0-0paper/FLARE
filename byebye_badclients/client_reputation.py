import numpy as np
from flwr.common import FitRes
from sklearn import metrics
from sympy.abc import alpha
import time
from util import *
from enum import Enum
from byebye_badclients.util import mahalanobis_distance

class Classification(Enum):
    TRUSTED = 0
    SUSPICIOUS = 1
    UNTRUSTED = 2

class ClientReputation:
    def __init__(self, cid):
        self.cid = cid
        self.exp_mv_avg = None
        self.performance_consistency_score = None
        self.statistical_anomaly_score = None
        self.temporal_behavior_score = None
        self.reputation_score = None
        self.classification = Classification.SUSPICIOUS
        self.round_tracker = 0

    def update_scores(self,
                      fit_res: FitRes, now: float,
                      mean, inv_covariance, participation_rate, response_time_variance,
                      reputation_weights, alpha=0.7, anomaly_threshold=5.99, penalty_severity = 2,
                      beta=0.6):
        estimated_rtt = fit_res.metrics["receive_time"] * 2
        self.update_round_tracker()
        self.update_performance_consistency_score(update=fit_res.parameters, alpha=alpha)
        distance = mahalanobis_distance(update=fit_res.parameters, mean=mean, inv_covariance=inv_covariance)
        self.update_statistical_anomaly_score(update=fit_res.parameters, distance=distance,
                                              anomaly_threshold=estimated_rtt, penalty_severity=penalty_severity)
        self.update_temporal_behaviour_score(beta=beta, participation_rate=participation_rate,
                                             response_time_variance=response_time_variance)
        self.update_reputation_score(reputation_weights=reputation_weights)

    def update_round_tracker(self):
        self.round_tracker += 1
        return self.round_tracker

    def update_exponential_moving_average(self, update, alpha=0.5):
        if self.exp_mv_avg is None:
            self.exp_mv_avg = update
        else:
            self.exp_mv_avg = update * alpha + (1 - alpha) * self.exp_mv_avg

    def update_performance_consistency_score(self, update, alpha):

        self.update_exponential_moving_average(update, alpha=0.5)
        cos_sim = metrics.pairwise.cosine_similarity(update, self.exp_mv_avg)
        self.performance_consistency_score = alpha * self.performance_consistency_score + (1 - alpha) * cos_sim

        return self.performance_consistency_score

    def update_statistical_anomaly_score(self, update, distance, anomaly_threshold, penalty_severity):
        if distance <= anomaly_threshold:
            self.statistical_anomaly_score = 1
        else:
            self.statistical_anomaly_score = np.exp(-penalty_severity * (distance - anomaly_threshold))

    def update_temporal_behaviour_score(self, beta, participation_rate, response_time_variance):
        self.temporal_behavior_score = beta * participation_rate + (1 - beta) * 1 / (1 + response_time_variance)
        return self.temporal_behavior_score

    def update_reputation_score(self, reputation_weights):
        scores = [
            self.performance_consistency_score,
            self.statistical_anomaly_score,
            self.temporal_behavior_score,
        ]
        self.reputation_score = np.dot(reputation_weights, scores)
        return self.reputation_score

    def reputation_decay_recovery(self, reliability_threshold, recovery, decay):
        if self.classification == Classification.TRUSTED:
            self.reputation_score = min(self.reputation_score + recovery, 1)
        else:
            self.reputation_score = min(self.reputation_score - decay, 0)
        return self.reputation_score

    def update_classification(self, reliability_threshold):
        if self.reputation_score >= reliability_threshold:
            self.classification = Classification.TRUSTED
        elif self.reputation_score < reliability_threshold / 2:
            self.classification = Classification.UNTRUSTED
        else:
            self.classification = Classification.SUSPICIOUS
        return self.classification
