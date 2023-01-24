from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np

def _dist_perm(cm):
    s = np.max(cm)
    return s-cm

def clustering_accuracy(true_labels, estimate_labels):
    """ Measures accuracy score of clustering.

    This function searches optimal matching of true_labels and estimate_labels, and
    produce the best accuracy score.

    Parameters
    ----------
    true_labels : numpy.ndarray
        True cluster labels
    estimate_labels : numpy.ndarray
        Estimated cluster labels

    Returns
    -------
    accuracy : float
        Accuracy score.
    indices : tuple of ndarray
        Matching of ``true_labels`` and ``estimated_labels``.
        Label ``indices[0][i]`` of ``true_labels`` and label ``indices[1][i]`` of ``estimated_labels`` correspond each other in the optimal matching.
    """
    cm = confusion_matrix(true_labels, estimate_labels)
    indexes = linear_sum_assignment(_dist_perm(cm))
    cm2 = cm[indexes[0], indexes[1]]
    accuracy = np.sum(cm2)/len(true_labels)

    return accuracy, (indexes[0], indexes[1])

def SSE(X, centers, labels):
    n_cluster = centers.shape[0]
    # returns sum of squared error for kmeans
    sample_coord1 = X['x1'].values
    sample_coord2 = X['x2'].values
    sample_angle = X['angle'].values
    start_index = get_start_indices(X)
    cells = tuple_2dcells()
    lenmat = b_to_b_lenmat()
    dists_from_center = np.zeros((3, n))
    for k in range(n_cluster):
        dists_from_center[k] = geodesic_dist2d(sample_coord1, sample_coord2, sample_angle, start_index, centers.iloc[k]['x1'],centers.iloc[k]['x2'], centers.iloc[k]['angle'], (int(centers.iloc[k]['edge1']), int(centers.iloc[k]['edge2'])), cells, lenmat)
    SSE = np.sum(np.min(dists_from_center, axis=0)**2)
    return SSE
