import numpy as np
from numpy import argmax
from numpy.linalg import norm
from ._kde import kernel_density_estimate_2dim
from ._lcmle import logconcave_density_estimate_2dim
from scipy.integrate import quad, dblquad
from scipy.stats import multivariate_normal
from ._utils import *
from ._normal import normal_centered_2dim
import pandas as pd
from scipy.integrate import quad, dblquad



def ise_1dim(true_density, estimate_density, epsabs=1e-4):
    """Computation of integrated squared error (ise) in 1dim tree space.

    Parameters
    ----------
    true_density : density object
    estimate_density : density object
    epsabs : float
        Allowed error for computation of ise.
        Defaults to 1e-4.

    Notes
    -----
    Both ``true_density'' and ``estimate_density'' should be one of the followings:
        - :py:class:`normal_1dim`
        - :py:class:`normal_bend_1dim`
        - :py:class:`coalescent_1dim`
        - :py:class:`exponential_1dim`
        - :py:class:`kernel_density_estimate_1dim`
        - :py:class:`logconcave_density_estimate_1dim`

    Returns
    -------
        ise : Integrated squared error
        err : Error of ise.
    """
    def true_v_estimate(x,cell):
        return (true_density.pdf(x,cell) - estimate_density.pdf(x,cell))**2
    ise = 0; err = 0;
    for i in range(3):
        res = quad(true_v_estimate, 0, np.inf, (i), epsabs=epsabs)
        ise += res[0]; err += res[1]
    return ise, err

def ise_2dim(true_density, estimate_density,epsabs=1e-4):
    """Computation of integrated squared error (ise) in 2dim tree space.

    Parameters
    ----------
    true_density : density object
    estimate_density : density object
    epsabs : float
        Allowed error for computation of ise.
        Defaults to 1e-4.

    Notes
    -----
    Both ``true_density'' and ``estimate_density'' should be one of the followings:
        - :py:class:`normal_centered_2dim`
        - :py:class:`normal_uncentered_2dim`
        - :py:class:`kernel_density_estimate_2dim`
        - :py:class:`logconcave_density_estimate_2dim`

    Returns
    -------
        ise : Integrated squared error
        err : Error of ise.
    """
    cells = tuple_2dcells()
    def true_v_estimate(x1,x2,cell0,cell1):
        return (true_density.pdf(x1,x2,cell0, cell1) - estimate_density.pdf(x1,x2,cell0,cell1))**2
    ise = 0; err = 0
    for i in range(15):
        cell = cells[i]
        res = dblquad(true_v_estimate, 0, np.inf, 0, np.inf, (cell[0], cell[1]) , epsabs=epsabs)
        ise += res[0]; err += res[1]
    return ise, err


def case3_ise(y, X, epsabs=1e-4):
    n = len(y)
    sample_coord1 = X['x1'].values
    sample_coord2 = X['x2'].values
    sample_angle = X['angle'].values
    start_index = get_start_indices(X)
    #for size in [200]:
    cells = tuple_2dcells()
    lenmat = b_to_b_lenmat()
    true_density = normal_centered_2dim(cells,sigma=1)

    # Calculate ISE for KDE
    kde = kernel_density_estimate_2dim(X)

    def true_v_kde_2(x1,x2,cell0,cell1):
        return (true_density.pdf(x1,x2,cell0, cell1) - kde.pdf(x1,x2,cell0,cell1))**2

    ise_kde = 0; err_kde = 0
    for i in range(15):
        cell = cells[i]
        res = dblquad(true_v_kde_2, 0, np.inf, 0, np.inf, (cell[0], cell[1]) , epsabs=epsabs)
        ise_kde += res[0]; err_kde += res[1]
        print(ise_kde, err_kde)

    lcmle = logconcave_density_estimate_2dim(y, X)

    def true_v_lcmle_2(x1,x2,cell0,cell1):
        return (true_density.pdf(x1,x2,cell0, cell1) - lcmle.pdf(x1,x2,cell0,cell1))**2

    ise_lcmle = 0; err_lcmle = 0
    for i in range(15):
        cell = cells[i]
        res = dblquad(true_v_lcmle_2, 0, np.inf, 0, np.inf, (cell[0], cell[1]) , epsabs=epsabs)
        ise_lcmle += res[0]; err_lcmle += res[1]
        print(ise_lcmle, err_lcmle)

    return ise_lcmle, err_lcmle, ise_kde, err_kde

            ########## EDITING #########

        #     # make x depending on the seed
        #     np.random.seed(seed)
        #     cells = list_2dcells()
        #     density_cells = list_2dcells()
        #     x_d = np.random.multivariate_normal(mean=np.zeros(2),cov=np.eye(2), size=size)
        #     x_d = np.abs(x_d)
        #     labels = np.random.choice([i for i in range(15)], size = size)
        #     labels = cells[labels]
        #     x = [[labels[i][0], labels[i][1], x_d[i][0], x_d[i][1]] for i in range(size)]
        #     x = add_angle(x)
        #     x = pd.DataFrame(x, columns=['edge1', 'edge2', 'x1', 'x2', 'angle'])
        #     try:
        #         y = np.load(f'results/pattern2/size{size}_seed{seed}.npy')
        #     except:
        #         continue
        #     P = x
        #     P['y'] = y
        #     P_numpy = P.to_numpy()
        #
        #     kernel_density = kernel_density_estimate(P_numpy)
        #     lcmle_density = logconcave_density_estimate(P_numpy)
        #
        #     def true_v_lcmle(x1,x2, cell0, cell1):
        #         estimated = lcmle_density(x1, x2, cell0, cell1)
        #         true = true_pattern2(x1,x2)
        #         return (estimated - true) ** 2
        #
        #     def true_v_kde(x1,x2, cell0, cell1):
        #         estimated = kernel_density(x1, x2, cell0, cell1)
        #         true = true_pattern2(x1,x2)
        #         return (estimated - true) ** 2
        #
        #     def kde2(x1,x2, cell0, cell1):
        #         return kernel_density(x1, x2, cell0, cell1)**2
        #     int = 0
        #     int_true = 0
        #     int_lcmle = 0
        #     int_true_v_lcmle = 0
        #     err_true_v_lcmle = 0
        #     int_true_v_kde = 0
        #     err_true_v_kde = 0
        #     for cell in density_cells:
        #         print('lcmle evaluating...')
        #         res = dblquad(true_v_lcmle, 0, np.inf, lambda x: 0, lambda x: np.inf,args=(cell), epsabs=1e-4)
        #         int_true_v_lcmle += res[0]
        #         err_true_v_lcmle += res[1]
        #         print(int_true_v_lcmle, err_true_v_lcmle)
        #     for cell in density_cells:
        #         print('kde evaluating...')
        #         res = dblquad(true_v_kde, 0, np.inf, lambda x: 0, lambda x: np.inf,args=(cell), epsabs=1e-4)
        #         int_true_v_kde += res[0]
        #         err_true_v_kde += res[1]
        #         print(int_true_v_kde, err_true_v_kde)
        #
        #     print('lcmle', int_true_v_lcmle, err_true_v_lcmle)
        #     print('kde', int_true_v_kde, err_true_v_kde)
        #     input()
        #     result_dict[seed] = [int_true_v_lcmle, err_true_v_lcmle, int_true_v_kde, err_true_v_kde ]
        # with open(f'results/pattern2/size{size}_20220124.json', 'w') as f:
        #     json.dump(result_dict, f, indent=4)


def case4_ise(y, X, epsabs=1e-4):
    #for size in [200]:
    n = len(y)
    sample_coord1 = X['x1'].values
    sample_coord2 = X['x2'].values
    sample_angle = X['angle'].values
    start_index = get_start_indices(X)
    cells = tuple_2dcells()
    density_cells = np.array([[0,1], [1,6], [6,8], [3,8], [3,4], [0,4]])
    lenmat = b_to_b_lenmat()
    true_density = normal_centered_2dim(density_cells,sigma=1)

    # Calculate ISE for KDE
    kde = kernel_density_estimate_2dim(X)

    def true_v_kde_2(x1,x2,cell0,cell1):
        return (true_density.pdf(x1,x2,cell0, cell1) - kde.pdf(x1,x2,cell0,cell1))**2

    ise_kde = 0; err_kde = 0
    for i in range(15):
        cell = cells[i]
        res = dblquad(true_v_kde_2, 0, np.inf, 0, np.inf, (cell[0], cell[1]) , epsabs=epsabs)
        ise_kde += res[0]; err_kde += res[1]
        print(ise_kde, err_kde)

    lcmle = logconcave_density_estimate_2dim(y,X)

    def true_v_lcmle_2(x1,x2,cell0,cell1):
        return (true_density.pdf(x1,x2,cell0, cell1) - lcmle.pdf(x1,x2,cell0,cell1))**2

    ise_lcmle = 0; err_lcmle = 0
    for i in range(15):
        cell = cells[i]
        res = dblquad(true_v_lcmle_2, 0, np.inf, 0, np.inf, (cell[0], cell[1]) , epsabs=epsabs)
        ise_lcmle += res[0]; err_lcmle += res[1]
        print(ise_lcmle, err_lcmle)

    return ise_lcmle, err_lcmle, ise_kde, err_kde

if __name__ == "__main__":
    X = pd.read_csv("data/case4/testcase_100_0.csv")
    # sorting by orthants
    sort_ind = argsort_by_orthants(X)
    X = X.iloc[sort_ind]

    # prepare arguments
    sample_coord1 = X['x1'].values
    sample_coord2 = X['x2'].values
    sample_angle = X['angle'].values
    start_index = get_start_indices(X)
    cells = tuple_2dcells()
    #orthants = tuple_orthants()

    #M = num_new_bds(start_index, cells)
    #embed, embed_indices, embed_n = embed_sample_points_to_nei3cells(sample_coord1, sample_coord2, start_index)

    #sample_bd_coord, sample_bd_lam, sample_origin_lam = geodesic_sample(sample_coord1, sample_coord2, sample_angle,start_index, cells)
    #edge_indices = link_convex_hull(sample_angle, start_index)
    #ext_coord, ext_lam, simple_indicator = twoDconvhull(sample_coord1, sample_coord2, sample_angle, start_index,
    #                    sample_bd_coord, sample_bd_lam,edge_indices)
    n = len(sample_coord1)
    #nei3cells = tuple_nei3cells()
    #cp_pairs = cone_path_pairs()

    #lenmat = b_to_b_lenmat()
    kde = kernel_density_estimate_2dim(X)
    integ = 0
    for i in range(15):
        print(i)
        cell = cells[i]
        integ += dblquad(kde.pdf, 0, np.inf, 0, np.inf, (cell[0], cell[1]) , epsabs=1e-4)[0]
        print(integ)
    y = np.load("../../result_remote/case4/y_100_0.npy")
    lcmle = logconcave_density_estimate_2dim(y,X)
    integ = 0
    for i in range(15):
        print(i)
        cell = cells[i]
        integ += dblquad(lcmle.pdf, 0, np.inf, 0, np.inf, (cell[0], cell[1]), epsabs=1e-4)[0]
        print(integ)
    print(integ)
