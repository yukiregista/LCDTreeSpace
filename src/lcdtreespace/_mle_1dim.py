from scipy.spatial import ConvexHull
from scipy import interpolate
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm, gaussian_kde
from scipy.optimize import minimize
from scipy import stats
from scipy.optimize import line_search
#import pdb
import time
import os

from ._auxiliary import J, J1



def tree_hull(y,x,labels,n_label):
    max_o = -np.inf
    top_ort = 0
    bottom_ort = 0
    lam_index = [0,0]
    lam = [0,0]
    # determine max o part
    for i in range(n_label):
        xi = x[labels==i]; yi = y[labels==i]
        for j in range(i+1,n_label):
            #take rgeodesic
            xj = x[labels == j]; yj = y[labels == j]
            for k in range(len(xi)):
                xik = xi[k]; yik = yi[k]
                for l in range(len(xj)):
                    xjl = xj[l]; yjl = yj[l]
                    d = xik/(xik+xjl)
                    y0 = (d * yjl + (1-d) * yik)
                    if (yik<yjl):
                        if y0 > max_o:
                            max_o = y0
                            top_ort = j; bottom_ort = i
                            lam_index[0]=l; lam_index[1]=k
                            lam[0] = d; lam[1] = (1-d)
                    else:
                        if y0 > max_o:
                            max_o = y0
                            top_ort = i; bottom_ort = j
                            lam_index[0]=k; lam_index[1]=l
                            lam[0] = (1-d); lam[1] = d

    hull_list = []
    support_list = []
    for i in range(n_label):
        points = np.vstack((np.append([0], x[labels==i]), np.append(max_o,y[labels==i]))).T
        if len(points)==2:
            class _hull:
                def __init__(self, points):
                    self.points = points
            hull = _hull(points)
            support = np.array([0,1])
        else:
            hull = ConvexHull(points, qhull_options = "QJ")
            support = np.sort(np.unique(hull.simplices[hull.equations[:,1]>0]))
        hull_list.append(hull); support_list.append(support)

    return hull_list, support_list, max_o, top_ort, bottom_ort, lam_index, lam

def tree_hull_bend(y,x,labels,n_label):
    max_o = -np.inf
    top_ort = 0
    bottom_ort = 0
    lam_index = [0,0]
    lam = [0,0]
    ratio = n_label - 1
    # determine max o part
    for i in range(n_label):
        xi = x[labels==i]; yi = y[labels==i]
        for j in range(i+1,n_label):
            #take rgeodesic
            xj = x[labels == j]; yj = y[labels == j]
            for k in range(len(xi)):
                xik = xi[k]; yik = yi[k]
                for l in range(len(xj)):
                    xjl = xj[l]; yjl = yj[l]
                    d = xik/(xik+xjl)
                    if (yik<yjl):
                        y0 = (d * yjl + ratio * (1-d) * yik)/(ratio* (1-d) + d)
                        if y0 > max_o:
                            max_o = y0
                            top_ort = j; bottom_ort = i
                            lam_index[0]=l; lam_index[1]=k
                            lam[0] = d/(ratio * (1-d) + d); lam[1] = ratio*(1-d)/(ratio * (1-d) + d)
                    else:
                        y0 = (ratio * d * yjl + (1-d) * yik)/(ratio * d + 1 - d)
                        if y0 > max_o:
                            max_o = y0
                            top_ort = i; bottom_ort = j
                            lam_index[0]=k; lam_index[1]=l
                            lam[0] = (1-d)/(ratio * d + 1 - d); lam[1] = ratio*d/(ratio * d + 1 - d)

    hull_list = []
    support_list = []
    for i in range(n_label):
        points = np.vstack((np.append([0], x[labels==i]), np.append(max_o,y[labels==i]))).T
        if len(points)==2:
            class _hull:
                def __init__(self, points):
                    self.points = points
            hull = _hull(points)
            support = np.array([0,1])
        else:
            hull = ConvexHull(points, qhull_options = "QJ")
            support = np.sort(np.unique(hull.simplices[hull.equations[:,1]>0]))
        hull_list.append(hull); support_list.append(support)

    return hull_list, support_list, max_o, top_ort, bottom_ort, lam_index, lam


def _obj_grad_1dim(y, x, labels, n_label, bend=False, return_only_integ = False,print_objective=False):
    n_s = len(y)
    ep = 1e-4
    if bend:
        hull_list, support_list, max_o, top_ort, bottom_ort, lam_index, lam = tree_hull_bend(y,x,labels,n_label)
    else:
        hull_list, support_list, max_o, top_ort, bottom_ort, lam_index, lam = tree_hull(y,x,labels,n_label)

    llh = -1/n_s * np.sum(y)
    integ = 0

    for i in range(n_label):
        int_i = 0
        support = support_list[i]
        hull = hull_list[i]
        for j in range(len(support)-1):
            xl, yl = hull.points[support[j]]; xr, yr = hull.points[support[j+1]]
            int_i += J(yl,yr,ep) * (xr-xl)
            #if int_i == -np.inf:
            #    pdb.set_trace()
        integ += int_i

    if return_only_integ:
        return integ


    obj = llh + integ
    grad = np.full(n_s, -1/n_s) # gradient from the first term
    origin_J = 0

    for i in range(n_label):
        to_ind = np.where(labels==i)[0]
        support = support_list[i]
        hull = hull_list[i]
        n_sup = len(support)
        #origin
        xc, yc = hull.points[support[0]]
        xr, yr = hull.points[support[1]]
        origin_J += J1(yc,yr,ep) * (xr-xc)
        for j in range(1, n_sup):
            grad_ind = to_ind[support[j]-1] #  -1 because of origin added
            xc, yc = hull.points[support[j]]
            xl, yl = hull.points[support[j-1]]
            if j == n_sup-1: #last point then only one term for integral
                grad[grad_ind] += J1(yc,yl,ep) * (xc-xl)
            else: # if not last then right neighbor exists
                xr, yr = hull.points[support[j+1]]
                grad[grad_ind] += J1(yc,yl,ep) * (xc-xl) + J1(yc,yr,ep) * (xr-xc)

    top_to_ind = np.where(labels==top_ort)[0]
    bottom_to_ind = np.where(labels == bottom_ort)[0]
    grad[top_to_ind[lam_index[0]]]+= origin_J * lam[0]
    grad[bottom_to_ind[lam_index[1]]]+= origin_J * lam[1]

    if print_objective:
        print(f"integral  :  {integ:.5f}", end=" ")
        print(f"objective  :  {obj:.5f}")
    #print(f"grad  :  {np.linalg.norm(grad)}")
    return (obj,grad)

def lcmle_1dim(x,ort,n_ort, bend=False, initial = 'random', random_seed = None,print_objective=False, runs = 5):
    """Calculates one dimensional log-concave m.l.e.

    Parameters
    ----------
    x : numpy.ndarray
        coordinates of sample points.
    ort : numpy.ndarray
        orthants that sample points belong to.
        Should have same length as x.
    n_ort : int
        number of orthants.
        In case of one dimensional tree space, n_ort should be 3.
    bend : bool, optional
        Indicator for allowing bend at the origin point.
        Defaults to False.
    initial : str or nd.array, optional
        How to set initial values of optimized parameters.
        Should be one of the followings:
            - 'random' : initial value is set randomly.random numbers can be controled by setting random_seed argument.
            - numpy.ndarray : used as initial value. The length of the array should be the same size as x and ort.
        Defaults to "random".
    random_seed : None or float, optional
        Random seed to use when initial is "random".
        Defaults to None.
    print_objective : bool, optional
        Whether to show objective values of each run.
        Defaults to False.
    runs : int, optional
        Number of optimization runs.
        Defaults to 5.

    Returns
    -------
    ndarray
        Optimal parameter

    """

    '''Calculates one dimensional log-concave m.l.e.

    Args:
        x (nd.array): coordinates of sample points.
        ort (nd.array): orthants that sample points belong to.
            Should have same length as x.
        n_ort (int): number of orthants.
            In case of one dimensional tree space, n_ort should be 3.
        bend (bool, optional): Indicator for allowing bend at the origin point.
            Defaults to False.
        initial (str or nd.array, optional): How to set initial values of optimized parameters.
            Should be one of the followings:
                "random" : initial value is set randomly.
                    random numbers can be controled by setting random_seed argument.
                nd.array : used as initial value.
                    The length of the array should be the same size as x and ort.
            Defaults to "random".
        random_seed (None or float, optional): Random seed to use when initial is "random".
            Defaults to None.
        print_objective (bool, optional): Whether to show objective values of each run.
            Defaults to False.
        runs (int, optional): Number of optimization runs.
            Defaults to 5.
    '''
    n = len(x)

    if initial == 'random':
        if random_seed is None:
            y = np.random.normal(size=n) - 5
        else:
            np.random.seed(random_seed)
            y = np.random.normal(size=n) - 5
    else:
        y = initial

    res = minimize(_obj_grad_1dim, jac=True, x0=y, args=(x, ort, n_ort, bend,False,False), method='BFGS')
    print(f"run {0}:", res.fun)
    for i in range(1,runs):
        y = np.random.normal(size=n) - 5
        res2 = minimize(_obj_grad_1dim, jac=True, x0=y, args=(x, ort, n_ort, bend,False,False), method='BFGS')
        print(f"run {i}:", res2.fun)
        if res2.fun < res.fun:
            res = res2



    #res = minimize(_obj_grad_1dim, jac=True, x0=y, args=(x, labels, n_ort, bend,False,print_objective), method='BFGS',options={"disp":True})
    #res = minimize(objdake, jac=None, x0=y, args=(x, labels, n_ort, bend,False,print_objective), method='BFGS',options={"disp":True})
    '''
    pbm = PBM(n=n, sense=min)
    y = res.x
    for i in range(10000):
        if i%100 == 0:
            obj, g = _obj_grad_1dim(y,x,labels,n_ort,bend,False,True)
        else:
            obj, g = _obj_grad_1dim(y,x,labels,n_ort,bend,False,False)
        #print(i, obj)
        y = pbm.step(obj, y, g)
    '''
    #res = minimize(_obj_grad_1dim, jac=True, x0=y, args=(x, labels, n_ort, bend,False,print_objective), method='BFGS',options={"disp":True})
    #print(_obj_grad_1dim(res.x,x,labels,n_ort)[0])
    integg = _obj_grad_1dim(res.x,x,ort,n_ort, bend, return_only_integ=True)
    return res.x - np.log(integg)


if __name__ == "__main__":

    d = np.load(f'testcase/k5/sample100/x_seed0.npy')
    labels = np.load(f'testcase/k5/sample100/labels_seed0.npy')
    for i in range(1,5):
        d[labels==i] = -d[labels==i]
    sorted_index = np.argsort(d)
    d = d[sorted_index]
    labels = labels[sorted_index]

    n_label = 5
    size = len(d)
    y = np.random.normal(loc=0, scale=1, size=size)-3

    res = minimize(optimize, jac=True, x0=y, args=(d, labels, n_label), method='BFGS')
