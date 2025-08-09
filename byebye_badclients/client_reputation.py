from sklearn import metrics

class ClientReputation:
    def __init__(self, cid):
        self.cid = cid
        self.exp_mv_avg = None
        self.performance_consistency_score = None
        self.statistical_anomaly_score = None
        self.temporal_behavior_score = None
        self.reputation_score = None

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