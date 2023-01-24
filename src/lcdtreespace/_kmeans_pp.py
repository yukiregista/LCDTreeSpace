import numpy as np
import pandas as pd

from ._utils import *
from ._kde import _geodesic_dist2d, _create_distance_mat


def _geodesic_lam(a,b,lam, lenmat = None):
    # Computes (1-lam)*a + lam * b, the point lam*d(a,b) away from a on the geodesic [a,b]
    if lenmat is None:
        lenmat = b_to_b_lenmat()

    cell_a = (int(a[0]), int(a[1]))
    cell_b = (int(b[0]), int(b[1]))

    # Case A: cell_a and cell_b are the same
    if cell_a[0] == cell_b[0] and cell_a[1] == cell_b[1]:
        coords = (1-lam) * a[['x1', 'x2']].to_numpy() + lam * b[['x1', 'x2']].to_numpy()
        p = pd.Series([cell_a[0], cell_a[1], coords[0], coords[1], np.arctan2(coords[1],coords[0])], index = ["edge1", "edge2", "x1", "x2", "angle"])
    else:
        dist_ab = np.array([lenmat[cell_a[0], cell_b[0]], lenmat[cell_a[0], cell_b[1]],
                    lenmat[cell_a[1], cell_b[0]],lenmat[cell_a[1], cell_b[1]]], dtype=np.int8)
        index_min = np.argmin(dist_ab)
        n_ort_btwn = dist_ab[index_min]
        l, r = index_min//2, index_min % 2

        #constants
        PI = np.pi
        PI_2 = np.pi/2

        if n_ort_btwn == 0:
            bd_index = cell_a[l]
            if l: #if l==1
                x_a = a["x1"];
                y_a = a["x2"]; ## y is the shared coordinate
            else:
                x_a = a["x2"]
                y_a = a["x1"] ## y is the shared coordinate
            if r: ## if r==1
                x_b = -b["x1"]
                y_b = b["x2"] ## y is the shared coordinate
            else:
                x_b = -b["x2"]
                y_b = b["x1"] ## y is the shared coordinate
            x_p = (1-lam) * x_a + lam * x_b
            y_p = (1-lam) * y_a + lam * y_b
            if x_p >= 0:
                # p is in the same orthant as a
                if l:
                    p = pd.Series([cell_a[0], cell_a[1], x_p, y_p, np.arctan2(y_p,x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                else:
                    p = pd.Series([cell_a[0], cell_a[1], y_p, x_p, np.arctan2(x_p,y_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
            else:
                # p is in the same orthant as b
                if r:
                    p = pd.Series([cell_b[0], cell_b[1], -x_p, y_p, np.arctan2(y_p,-x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                else:
                    p = pd.Series([cell_b[0], cell_b[1], y_p, -x_p, np.arctan2(-x_p,y_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])

        elif n_ort_btwn == 1:
            bd1_index = cell_a[l]
            bd2_index = cell_b[r]
            if l: #if l==1
                x_a = a["x1"];
                y_a = a["x2"]; ## y is the closest coordinate
                a_angle_to_closest_edge = PI_2 - a["angle"]
            else:
                x_a = a["x2"];
                y_a = a["x1"]; ## y is the closest coordinate
                a_angle_to_closest_edge = a["angle"]
            if r: ## if r==1
                x_b = -b["x2"]; ## x is the closest coordinate
                y_b = -b["x1"];
                b_angle_to_closest_edge = PI_2 - b["angle"]
            else:
                x_b = -b["x1"];
                y_b = -b["x2"]; ## x is the closest coordinate
                b_angle_to_closest_edge = b["angle"]
            ab_angle = a_angle_to_closest_edge + b_angle_to_closest_edge
            if ab_angle < PI/2:
                # not cone path
                x_p = (1-lam) * x_a + lam * x_b
                y_p = (1-lam) * y_a + lam * y_b
                if x_p >= 0 and y_p >= 0:
                    # p is in the same orthant as a
                    if l:
                        p = pd.Series([cell_a[0], cell_a[1], x_p, y_p, np.arctan2(y_p, x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                    else:
                        p = pd.Series([cell_a[0], cell_a[1], y_p, x_p, np.arctan2(x_p, y_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                elif x_p < 0 and y_p>=0:
                    # p is in the orthant [bd1_index,bd2_index] or [bd2_index,bd1_index]
                    if bd1_index < bd2_index:
                        p = pd.Series([bd1_index, bd2_index, y_p, -x_p, np.arctan2(-x_p, y_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                    else:
                        p = pd.Series([bd2_index, bd1_index, -x_p, y_p, np.arctan2(y_p, -x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                elif x_p<0 and y_p < 0:
                    # p is in the same orthant as b
                    if r:
                        p = pd.Series([cell_b[0], cell_b[1], -y_p, -x_p, np.arctan2(-x_p, -y_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                    else:
                        p = pd.Series([cell_b[0], cell_b[1], -x_p, -y_p, np.arctan2(-y_p, -x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
            else:
                # cone path
                norm_a = np.sqrt(x_a**2 + y_a**2)
                norm_b = np.sqrt(x_b**2 + y_b**2)
                mid_lam = norm_a/(norm_a + norm_b) # relative distance from a to the origin
                if lam <= mid_lam:
                    # point is in the same orthant as a
                    relative_lam = lam / mid_lam # relative distance from a to p (compared to dist(a,O))
                    x_p = (1-relative_lam) * a["x1"]; y_p = (1-relative_lam) * a["x2"]
                    p = pd.Series([cell_a[0], cell_a[1], x_p, y_p, np.arctan2(y_p, x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                else:
                    # point is in the same orthant as b
                    relative_lam = (1-lam)/(1-mid_lam) # relative distance from b to p (compared to dist(b,O))
                    x_p = (1-relative_lam) * b["x1"]; y_p = (1-relative_lam) * b["x2"]
                    p = pd.Series([cell_b[0], cell_b[1], x_p, y_p, np.arctan2(y_p, x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
        else:
            # cone path
            norm_a = np.sqrt(a["x1"]**2 + a["x2"]**2)
            norm_b = np.sqrt(b["x1"]**2 + b["x2"]**2)
            mid_lam = norm_a/(norm_a + norm_b) # relative distance from a to the origin
            if lam <= mid_lam:
                # point is in the same orthant as a
                relative_lam = lam / mid_lam # relative distance from a to p (compared to dist(a,O))
                x_p = (1-relative_lam) * a["x1"]; y_p = (1-relative_lam) * a["x2"]
                p = pd.Series([cell_a[0], cell_a[1], x_p, y_p, np.arctan2(y_p, x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
            else:
                # point is in the same orthant as b
                relative_lam = (1-lam)/(1-mid_lam) # relative distance from b to p (compared to dist(b,O))
                x_p = (1-relative_lam) * b["x1"]; y_p = (1-relative_lam) * b["x2"]
                p = pd.Series([cell_b[0], cell_b[1], x_p, y_p, np.arctan2(y_p,x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
    return p


def _geodesic_lam_fast(a_edge1, a_edge2, a_coord1, a_coord2, a_angle,
                    b_edge1, b_edge2, b_coord1, b_coord2, b_angle,
                    lam, lenmat):
    # Computes (1-lam)*a + lam * b, the point lam*d(a,b) away from a on the geodesic [a,b]

    cell_a = (int(a_edge1), int(a_edge2))
    cell_b = (int(b_edge1), int(b_edge2))

    # Case A: cell_a and cell_b are the same
    if cell_a[0] == cell_b[0] and cell_a[1] == cell_b[1]:
        p_edge1 = a_edge1
        p_edge2 = a_edge2
        p_coord1 = (1-lam) * a_coord1 + lam * b_coord1
        p_coord2 = (1-lam) * a_coord2 + lam * b_coord2
        p_angle = np.arctan2(p_coord2, p_coord1)
    else:
        dist_ab = np.array([lenmat[cell_a[0], cell_b[0]], lenmat[cell_a[0], cell_b[1]],
                    lenmat[cell_a[1], cell_b[0]],lenmat[cell_a[1], cell_b[1]]], dtype=np.int8)
        index_min = np.argmin(dist_ab)
        n_ort_btwn = dist_ab[index_min]
        l, r = index_min//2, index_min % 2

        #constants
        PI = np.pi
        PI_2 = np.pi/2

        if n_ort_btwn == 0:
            bd_index = cell_a[l]
            if l: #if l==1
                x_a = a_coord1;
                y_a = a_coord2; ## y is the shared coordinate
            else:
                x_a = a_coord2
                y_a = a_coord1 ## y is the shared coordinate
            if r: ## if r==1
                x_b = -b_coord1
                y_b = b_coord2 ## y is the shared coordinate
            else:
                x_b = -b_coord2
                y_b = b_coord1 ## y is the shared coordinate
            x_p = (1-lam) * x_a + lam * x_b
            y_p = (1-lam) * y_a + lam * y_b
            if x_p >= 0:
                # p is in the same orthant as a
                if l:
                    p_edge1 = a_edge1
                    p_edge2 = a_edge2
                    p_coord1 = x_p
                    p_coord2 = y_p
                    p_angle = np.arctan2(y_p,x_p)
                    #p = pd.Series([cell_a[0], cell_a[1], x_p, y_p, np.arctan2(y_p,x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                else:
                    p_edge1 = a_edge1
                    p_edge2 = a_edge2
                    p_coord1 = y_p
                    p_coord2 = x_p
                    p_angle = np.arctan2(x_p,y_p)
                    #p = pd.Series([cell_a[0], cell_a[1], y_p, x_p, np.arctan2(x_p,y_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
            else:
                # p is in the same orthant as b
                if r:
                    p_edge1 = b_edge1
                    p_edge2 = b_edge2
                    p_coord1 = -x_p
                    p_coord2 = y_p
                    p_angle = np.arctan2(y_p,-x_p)
                    #p = pd.Series([cell_b[0], cell_b[1], -x_p, y_p, np.arctan2(y_p,-x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                else:
                    p_edge1 = b_edge1
                    p_edge2 = b_edge2
                    p_coord1 = y_p
                    p_coord2 = -x_p
                    p_angle = np.arctan2(-x_p,y_p)
                    #p = pd.Series([cell_b[0], cell_b[1], y_p, -x_p, np.arctan2(-x_p,y_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])

        elif n_ort_btwn == 1:
            bd1_index = cell_a[l]
            bd2_index = cell_b[r]
            if l: #if l==1
                x_a = a_coord1;
                y_a = a_coord2; ## y is the closest coordinate
                a_angle_to_closest_edge = PI_2 - a_angle
            else:
                x_a = a_coord2;
                y_a = a_coord1; ## y is the closest coordinate
                a_angle_to_closest_edge = a_angle
            if r: ## if r==1
                x_b = -b_coord2; ## x is the closest coordinate
                y_b = -b_coord1;
                b_angle_to_closest_edge = PI_2 - b_angle
            else:
                x_b = -b_coord1;
                y_b = -b_coord2; ## x is the closest coordinate
                b_angle_to_closest_edge = b_angle
            ab_angle = a_angle_to_closest_edge + b_angle_to_closest_edge
            if ab_angle < PI/2:
                # not cone path
                x_p = (1-lam) * x_a + lam * x_b
                y_p = (1-lam) * y_a + lam * y_b
                if x_p >= 0 and y_p >= 0:
                    # p is in the same orthant as a
                    if l:
                        p_edge1 = a_edge1
                        p_edge2 = a_edge2
                        p_coord1 = x_p
                        p_coord2 = y_p
                        p_angle = np.arctan2(y_p,x_p)
                        #p = pd.Series([cell_a[0], cell_a[1], x_p, y_p, np.arctan2(y_p, x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                    else:
                        p_edge1 = a_edge1
                        p_edge2 = a_edge2
                        p_coord1 = y_p
                        p_coord2 = x_p
                        p_angle = np.arctan2(x_p,y_p)
                        #p = pd.Series([cell_a[0], cell_a[1], y_p, x_p, np.arctan2(x_p, y_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                elif x_p < 0 and y_p>=0:
                    # p is in the orthant [bd1_index,bd2_index] or [bd2_index,bd1_index]
                    if bd1_index < bd2_index:
                        p_edge1 = bd1_index
                        p_edge2 = bd2_index
                        p_coord1 = y_p
                        p_coord2 = -x_p
                        p_angle = np.arctan2(-x_p,y_p)
                        #p = pd.Series([bd1_index, bd2_index, y_p, -x_p, np.arctan2(-x_p, y_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                    else:
                        p_edge1 = bd1_index
                        p_edge2 = bd2_index
                        p_coord1 = -x_p
                        p_coord2 = y_p
                        p_angle = np.arctan2(y_p,-x_p)
                        #p = pd.Series([bd2_index, bd1_index, -x_p, y_p, np.arctan2(y_p, -x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                elif x_p<0 and y_p < 0:
                    # p is in the same orthant as b
                    if r:
                        p_edge1 = b_edge1
                        p_edge2 = b_edge2
                        p_coord1 = -y_p
                        p_coord2 = -x_p
                        p_angle = np.arctan2(-x_p,-y_p)
                        #p = pd.Series([cell_b[0], cell_b[1], -y_p, -x_p, np.arctan2(-x_p, -y_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                    else:
                        p_edge1 = b_edge1
                        p_edge2 = b_edge2
                        p_coord1 = -x_p
                        p_coord2 = -y_p
                        p_angle = np.arctan2(-y_p,-x_p)
                        #p = pd.Series([cell_b[0], cell_b[1], -x_p, -y_p, np.arctan2(-y_p, -x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
            else:
                # cone path
                norm_a = np.sqrt(x_a**2 + y_a**2)
                norm_b = np.sqrt(x_b**2 + y_b**2)
                mid_lam = norm_a/(norm_a + norm_b) # relative distance from a to the origin
                if lam <= mid_lam:
                    # point is in the same orthant as a
                    relative_lam = lam / mid_lam # relative distance from a to p (compared to dist(a,O))
                    p_edge1 = a_edge1
                    p_edge2 = a_edge2
                    p_coord1 = (1-relative_lam) * a_coord1
                    p_coord2 = (1-relative_lam) * a_coord2
                    p_angle = np.arctan2(p_coord2,p_coord1)
                    #p = pd.Series([cell_a[0], cell_a[1], x_p, y_p, np.arctan2(y_p, x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
                else:
                    # point is in the same orthant as b
                    relative_lam = (1-lam)/(1-mid_lam) # relative distance from b to p (compared to dist(b,O))
                    p_edge1 = b_edge1
                    p_edge2 = b_edge2
                    p_coord1 = (1-relative_lam) * b_coord1
                    p_coord2 = (1-relative_lam) * b_coord2
                    p_angle = np.arctan2(p_coord2,p_coord1)
                    #x_p = (1-relative_lam) * b_coord1; y_p = (1-relative_lam) * b_coord2
                    #p = pd.Series([cell_b[0], cell_b[1], x_p, y_p, np.arctan2(y_p, x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
        else:
            # cone path
            norm_a = np.sqrt(a_coord1**2 + a_coord2**2)
            norm_b = np.sqrt(b_coord1**2 + b_coord2**2)
            mid_lam = norm_a/(norm_a + norm_b) # relative distance from a to the origin
            if lam <= mid_lam:
                # point is in the same orthant as a
                relative_lam = lam / mid_lam # relative distance from a to p (compared to dist(a,O))
                p_edge1 = a_edge1
                p_edge2 = a_edge2
                p_coord1 = (1-relative_lam) * a_coord1
                p_coord2 = (1-relative_lam) * a_coord2
                p_angle = np.arctan2(p_coord2,p_coord1)
                #x_p = (1-relative_lam) * a_coord1; y_p = (1-relative_lam) * a_coord2
                #p = pd.Series([cell_a[0], cell_a[1], x_p, y_p, np.arctan2(y_p, x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
            else:
                # point is in the same orthant as b
                relative_lam = (1-lam)/(1-mid_lam) # relative distance from b to p (compared to dist(b,O))
                p_edge1 = b_edge1
                p_edge2 = b_edge2
                p_coord1 = (1-relative_lam) * b_coord1
                p_coord2 = (1-relative_lam) * b_coord2
                p_angle = np.arctan2(p_coord2,p_coord1)
                #x_p = (1-relative_lam) * b_coord1; y_p = (1-relative_lam) * b_coord2
                #p = pd.Series([cell_b[0], cell_b[1], x_p, y_p, np.arctan2(y_p,x_p)], index = ["edge1", "edge2", "x1", "x2", "angle"])
    return p_edge1, p_edge2, p_coord1, p_coord2, p_angle




def frechet_mean(X, max_iter = 5000, eps = 1e-7):
    """ Computes frechet mean of sample points.

    It uses the proximal point algorithm.

    Parameters
    ----------
    X : pandas.DataFrame
        Sample points. See :py:func:`lcmle_2dim` for the required format.
    max_iter : int
        Maximum number of iterations of proximal point algorithm.
        Defaults to 1000.
    eps : float
        When sum of squared distances from current mean estimates does not decrease by more than ``eps``, then the algorithm terminates.

    Returns
    -------
    mean_edge1 : int
        First orthant of the mean.
    mean_edge2
        Second orthant of the mean.
    mean_coord1
        First coordinate of the mean.
    mean_coord2
        Second coordinate of the mean.
    mean_angle
        ``np.arctan(mean_coord2/mean_coord1)``.
    """

    # assume X is sorted
    sample_coord1 = X['x1'].values
    sample_coord2 = X['x2'].values
    sample_edge1 = X['edge1'].values
    sample_edge2 = X['edge2'].values
    sample_angle = X['angle'].values
    start_index = get_start_indices(X)

    cells = tuple_2dcells()
    lenmat = b_to_b_lenmat()

    #mean = X.iloc[-1]
    mean_edge1 = sample_edge1[-1]
    mean_edge2 = sample_edge2[-1]
    mean_coord1 = sample_coord1[-1]
    mean_coord2 = sample_coord2[-1]
    mean_angle = sample_angle[-1]
    n = len(X)
    n_iter = 0
    error = np.inf
    temp = np.inf
    while n_iter < max_iter:
        old_error = error
        n_iter += 1
        lam = 1/(n_iter + 1)
        #mean_old = mean
        for i in range(n):
            #old_temp = temp
            #mean_old = mean
            mean_edge1, mean_edge2, mean_coord1, mean_coord2, mean_angle = _geodesic_lam_fast(
                mean_edge1, mean_edge2, mean_coord1, mean_coord2, mean_angle,
                sample_edge1[i], sample_edge2[i], sample_coord1[i], sample_coord2[i], sample_angle[i],
                (2*lam)/(n + 2 * lam), lenmat
            )
            #mean = _geodesic_lam(mean, X.iloc[i], (2*lam)/(n + 2 * lam),lenmat = lenmat)
            #dists = _geodesic_dist2d(sample_coord1, sample_coord2, sample_angle, start_index, mean['x1'],mean['x2'], mean['angle'], (int(mean['edge1']),int(mean['edge2'])), cells, lenmat)
            #temp = np.sum(dists**2)
        #dists = _geodesic_dist2d(sample_coord1, sample_coord2, sample_angle, start_index, mean['x1'],mean['x2'], mean['angle'], (int(mean['edge1']),int(mean['edge2'])), cells, lenmat)
        dists = _geodesic_dist2d(sample_coord1, sample_coord2, sample_angle, start_index, mean_coord1,mean_coord2, mean_angle, (int(mean_edge1),int(mean_edge2)), cells, lenmat)
        error = np.sum(dists**2)
        if np.abs(old_error - error)/np.abs(old_error) < eps:
            break
    return mean_edge1, mean_edge2, mean_coord1, mean_coord2, mean_angle


def kmeans_pp(X, n_cluster = 2, seed=10, max_iter=1000):
    """ Conducts k-means++ algorithm.

    Parameters
    ----------
    X : pandas.DataFrame
        Sample points. See :py:func:`lcmle_2dim` for the required format.
    n_cluster : int
        Number of clusters.
    seed : float
        Random seed for initialization of centers.
        Defaults to 10.
    max_iter : int
        Maximum number of iterations of EM algorithm.
        Defaults to 1000.

    Returns
    -------
    labels : numpy.ndarray
        Estimated cluster labels.
    cluster_centers : pandas.DataFrame
        DataFrame containing cluster centers (with the same format as input DataFrame).
    """
    #x['y'] = 1
    n = len(X)
    #x_np = x.to_numpy()
    sample_coord1 = X['x1'].values
    sample_coord2 = X['x2'].values
    sample_angle = X['angle'].values
    start_index = get_start_indices(X)
    cells = tuple_2dcells()
    lenmat = b_to_b_lenmat()
    Dmat = _create_distance_mat(sample_coord1, sample_coord2, sample_angle, start_index,cells, lenmat)
    np.random.seed(seed)

    # pick cluster center
    center_indices = []
    center_indices.append(np.random.choice([i for i in range(n)]))
    for k in range(1, n_cluster):
        dk = np.min(Dmat[center_indices[:k]]**2, axis=0) # minimum squared distance to one of the cluster
        dk_sum = np.sum(dk)
        center_indices.append(np.random.choice([i for i in range(n)] , p = dk/dk_sum ))

    # iteration (kmeans)
    # labeling
    dists_from_center = Dmat[center_indices]
    labels = np.argmin(dists_from_center, axis=0)
    labels_old = n_cluster - 1 - labels # Just making it different from label
    n_iter=0
    cluster_edge1 = np.zeros(n_cluster).astype(np.int32)
    cluster_edge2 = np.zeros(n_cluster).astype(np.int32)
    cluster_coord1 = np.zeros(n_cluster)
    cluster_coord2 = np.zeros(n_cluster)
    cluster_angle = np.zeros(n_cluster)

    dists_from_center = np.zeros((n_cluster, n))
    while (n_iter < max_iter) and (not (labels_old==labels).all()):
        n_iter += 1

        # renew cluster center
        for i in range(n_cluster):
            x = X.iloc[labels==i]
            cluster_edge1[i], cluster_edge2[i], cluster_coord1[i], cluster_coord2[i], cluster_angle[i] = frechet_mean(x)
            dists_from_center[i] = _geodesic_dist2d(sample_coord1, sample_coord2, sample_angle, start_index, cluster_coord1[i],cluster_coord2[i], cluster_angle[i], (int(cluster_edge1[i]), int(cluster_edge2[i])), cells, lenmat)
        # renew labels
        labels_old = labels
        labels = np.argmin(dists_from_center, axis=0)
        #print(n_iter)
        #print(np.sum(np.min(dists_from_center,axis=0)**2))
        #print(dists_from_center)
        #print(np.sum(np.min(dists_from_center,axis=0)**2))
    cluster_centers = pd.DataFrame(data = {
        "edge1" : cluster_edge1,
        "edge2" : cluster_edge2,
        "x1" : cluster_coord1,
        "x2" : cluster_coord2,
        "angle" : cluster_angle
    })
    return labels, cluster_centers
