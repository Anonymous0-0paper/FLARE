import scipy.spatial as spatial

def mahalanobis_distance(update, mean, inv_covariance):
    return spatial.distance.mahalanobis(update, mean, inv_covariance)