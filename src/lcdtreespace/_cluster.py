from ._optimize import _obj_grad
from ._utils import *
from ._least_concave_func import _num_new_bds
from ._geodesic_sample import geodesic_sample
from ._link_convhull import _link_convex_hull
from ._twoDconvhull import _twoDconvhull
from ._optimize import _calc_integ
from scipy.optimize import minimize
from ._kde import _create_distance_mat
from ._normal import normal_uncentered_2dim
import numpy as np

__all__ = ['cluster']

def lcmix_cluster(X, n_cluster, y_random_seed = None, pi_init = 'uniform', pi_random_seed = None,
    max_em_iter = 100, rtol=1e-6):
    # conducts clustering with log-concave mixture model.
    # INPUT:
    ## X : dataframe containing sample points. It should have following columns:
    ### edge1, edge2 : integer indicating the edge
    ### x1, x2 : coordinates of sample points
    ### angle : arctan(x2/x1)
    ### It should be sorted before conducting clustering.

    ## n_cluster : int, number of cluster.

    ## y_random_seed: seed for random initialization of y.

    ## y_init : choice for initialization. Should be one of the following:
    ### 'uniform' : initial pi gives uniform weight to each cluster
    ### ndarray of length n_cluster.

    ## pi_random_seed: seed for random initialization of pi.

    ## max_em_iter : maximum number of iteration of EM algorithm.

    # sorting by orthants -> assume sorted one as input
    #sort_ind = argsort_by_orthants(X)
    #X = X.iloc[sort_ind]

    # prepare arguments
    sample_coord1 = X['x1'].values
    sample_coord2 = X['x2'].values
    sample_angle = X['angle'].values
    start_index = get_start_indices(X)
    cells = tuple_2dcells()
    orthants = tuple_orthants()

    M = _num_new_bds(start_index, cells)
    embed, embed_indices, embed_n = embed_sample_points_to_nei3cells(sample_coord1, sample_coord2, start_index)

    sample_bd_coord, sample_bd_lam, sample_origin_lam = geodesic_sample(sample_coord1, sample_coord2, sample_angle,start_index, cells)
    edge_indices = _link_convex_hull(sample_angle, start_index)
    ext_coord, ext_lam, simple_indicator = _twoDconvhull(sample_coord1, sample_coord2, sample_angle, start_index,
                        sample_bd_coord, sample_bd_lam,edge_indices)
    n = len(sample_coord1)


    nei3cells = tuple_nei3cells()
    cp_pairs = cone_path_pairs()
    lenmat = b_to_b_lenmat()

    # Initialize likelihood Y (n × n_cluster)
    if y_random_seed is None:
        Y = np.random.normal(size = (n, n_cluster)) - 5
    else:
        Dmat = _create_distance_mat(sample_coord1, sample_coord2, sample_angle, start_index,cells, lenmat)
        np.random.seed(y_random_seed)

        #legacy
        '''
        center_indices = [np.random.choice([i for i in range(n)])]
        center = X.iloc[center_indices[0]]
        Y = np.zeros((n, n_cluster))
        for i in range(n):
            Y[i,0] = np.log(normal_uncentered_2dim((int(center['edge1']), int(center['edge2'])) ,np.array([center['x1'], center['x2']]), 1 ).pdf(sample_coord1[i], sample_coord2[i], int(X.iloc[i]['edge1']), int(X.iloc[i]['edge2'])))
        for k in range(1, n_cluster):
            dk = np.min(Dmat[center_indices[:k]]**2, axis=0)
            dk_sum = np.sum(dk)
            center_indices.append(np.random.choice([i for i in range(n)] , p = dk/dk_sum ))
            center = X.iloc[center_indices[-1]]
            for i in range(n):
                Y[i,k] = np.log(normal_uncentered_2dim((int(center['edge1']), int(center['edge2'])), np.array([center['x1'], center['x2']]), 1).pdf(sample_coord1[i], sample_coord2[i], int(X.iloc[i]['edge1']), int(X.iloc[i]['edge2'])))
        #Y = np.random.normal(size = (n, n_cluster)) - 5
        '''

        # new
        d = np.max(Dmat)
        center_indices = np.random.choice([_ for _ in range(n)], size=n_cluster, replace = False)
        Y = np.zeros((n, n_cluster))
        for k in range(n_cluster):
            center = X.iloc[center_indices[k]]
            for i in range(n):
                Y[i,k] = np.log(normal_uncentered_2dim((int(center['edge1']), int(center['edge2'])), np.array([center['x1'], center['x2']]), d/3).pdf(sample_coord1[i], sample_coord2[i], int(X.iloc[i]['edge1']), int(X.iloc[i]['edge2'])))
        #print("starting centers:")
        #print(X.iloc[center_indices])
        #input()


#########EDITING ###############

    # normalize Y to make it a density
    for k in range(n_cluster):
        y = Y[:,k]
        integ = _calc_integ(y, sample_coord1, sample_coord2, sample_angle, start_index,
                        sample_bd_coord, sample_bd_lam, sample_origin_lam,
                        edge_indices, ext_coord, ext_lam, simple_indicator,
                        embed, embed_indices, embed_n,
                        n, cells, nei3cells, M, cp_pairs)
        Y[:, k] = Y[:, k] - np.log(integ)

    # Initialize pi (cluster probability)
    if type(pi_init) == type("a") and pi_init == "uniform":
        pi  = np.array([1/n_cluster for _ in range(n_cluster)])
    else:
        pi = pi_init

    F = np.exp(Y)

    LogLikelihoods = [-np.inf]
    Y_list = []
    pi_list = []
    # ITERATION
    for iter in range(max_em_iter + 1):
        # EXPECTATIO
        Theta_unnormalized = pi.reshape(1,-1) * F
        normalize_const = F@pi
        Theta = Theta_unnormalized / normalize_const.reshape(-1,1)
        #np.save(f"result/mixture/cluster3_300_0_8_Theta_iter{iter}.npy", Theta)
        pi = np.sum(Theta, axis=0)/n
        LogLikelihood = np.sum( np.log( np.sum(Theta_unnormalized, axis=1) ) )
        LogLikelihoods.append(LogLikelihood)
        print("iteration:", iter, end=" ")
        print("  Loglikelihood:", LogLikelihood)
        #print("pi:", pi)
        #print("Theta:", Theta)
        #print("LogLikelihood:", LogLikelihood)
        #print("accuracy:", np.mean(X['which'] == np.argmin(Theta, axis=1)  ))
        #print("center1:", X.iloc[np.argmax(Y[:,0])])
        #print("center2:", X.iloc[np.argmax(Y[:,1])])
        #print("cells_in_1:\n", X[Theta[:,0] > Theta[:,1]].groupby(['edge1', 'edge2']).size().reset_index(name='Freq'))
        #print("\ncells_in_2\n", X[Theta[:,0] < Theta[:,1]].groupby(['edge1', 'edge2']).size().reset_index(name='Freq'))

        if iter == max_em_iter:
            break

        if np.abs(LogLikelihoods[-1] - LogLikelihoods[-2])/n < rtol:
            break

        # MAXIMIZATION
        for k in range(n_cluster):
            #print("M-step for cluster ",k)
            weight_unnormalized = Theta[:,k]
            weight = weight_unnormalized / np.sum(weight_unnormalized)
            y = Y[:,k]
            res = minimize(_obj_grad, y -np.random.normal(size=n) * 1e-2, jac=True, args = (weight, sample_coord1, sample_coord2, sample_angle, start_index,
                            sample_bd_coord, sample_bd_lam, sample_origin_lam,
                            edge_indices, ext_coord, ext_lam, simple_indicator,
                            embed, embed_indices, embed_n,
                            n, cells, nei3cells, M, cp_pairs, False) ,method="BFGS", options={"maxiter":200})
            integ = _calc_integ(res.x, sample_coord1, sample_coord2, sample_angle, start_index,
                            sample_bd_coord, sample_bd_lam, sample_origin_lam,
                            edge_indices, ext_coord, ext_lam, simple_indicator,
                            embed, embed_indices, embed_n,
                            n, cells, nei3cells, M, cp_pairs)
            #print("integ:", integ)
            Y[:,k] = res.x - np.log(integ)
        pi_list.append(pi)
        Y_list.append(Y)
        F = np.exp(Y)

    return pi, Y, Theta, LogLikelihoods[1:], pi_list, Y_list
