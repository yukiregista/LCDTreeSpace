import numpy as np
import pandas as pd
from ._utils import *
from numpy.linalg import det
from numpy import abs
from scipy.optimize import minimize
from ._least_concave_func import _least_concave_func, _num_new_bds
from ._auxiliary import *
from pprint import pprint
from ._geodesic_sample import geodesic_sample
from ._link_convhull import _link_convex_hull
from ._twoDconvhull import _twoDconvhull
import pickle
from scipy.sparse import csr_matrix, SparseEfficiencyWarning
import warnings


def _calculate_integral(coords, supports):

    integ = 0
    n_simplices = supports.shape[0]
    for i in range(n_simplices):
        support = supports[i]
        A = coords[[support[1], support[2]]][:,:2] - coords[support[0]][:2].reshape(1,2)
        area = abs(det(A))
        J = J000(coords[support[0], 2], coords[support[1], 2], coords[support[2],2])
        integ += J * area
    return integ

def _calculate_subgrad_comp(support0, support1, support2, coords):
    A = coords[[support1, support2]][:,:2] - coords[support0][:2].reshape(1,2)
    area = abs(det(A))
    J1 = J100(coords[support0, 2], coords[support1, 2], coords[support2, 2])
    return J1 * area

def _area(coords, supports):
    area = 0
    n_simplices = supports.shape[0]
    for i in range(n_simplices):
        support = supports[i]
        A = coords[[support[1], support[2]]][:,:2] - coords[support[0]][:2].reshape(1,2)
        area += abs(det(A))
    return area

def _objective(y, weight, sample_coord1, sample_coord2, sample_angle, start_index,
                sample_bd_coord, sample_bd_lam, sample_origin_lam,
                edge_indices, ext_coord, ext_lam, simple_indicator,
                embed, embed_indices, embed_n,
                n, cells, nei3cells, M, cp_pairs,):
    old_bd_coord, old_bd_lam, old_bd_y, max_o, max_o_lam, support_list, orthant_coords, hull_list = _least_concave_func(
                    y, sample_coord1, sample_coord2, sample_angle, start_index,
                    sample_bd_coord, sample_bd_lam, sample_origin_lam,
                    edge_indices, ext_coord, ext_lam, simple_indicator,
                    embed, embed_indices, embed_n,
                    n, cells, nei3cells, M, cp_pairs)
    integ = 0

    for i in range(15):
        supports = support_list[i]
        if supports is None:
            continue
        integ += _calculate_integral(orthant_coords[i], supports)


    llh = - weight @ y

    print(f"integral: {integ}")
    print(f"objective: {llh + integ}")
    #input()
    return llh+integ


def _calc_integ(y, sample_coord1, sample_coord2, sample_angle, start_index,
                sample_bd_coord, sample_bd_lam, sample_origin_lam,
                edge_indices, ext_coord, ext_lam, simple_indicator,
                embed, embed_indices, embed_n,
                n, cells, nei3cells, M, cp_pairs):
    old_bd_coord, old_bd_lam, old_bd_y, max_o, max_o_lam, support_list, orthant_coords, hull_list = _least_concave_func(
                    y, sample_coord1, sample_coord2, sample_angle, start_index,
                    sample_bd_coord, sample_bd_lam, sample_origin_lam,
                    edge_indices, ext_coord, ext_lam, simple_indicator,
                    embed, embed_indices, embed_n,
                    n, cells, nei3cells, M, cp_pairs)
    integ = 0

    for i in range(15):
        supports = support_list[i]
        if supports is None:
            continue
        integ += _calculate_integral(orthant_coords[i], supports)

    #input()
    return integ

def calc_integ(y,X):
    # assuming X is sorted
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


    integ = _calc_integ(y, sample_coord1, sample_coord2, sample_angle, start_index,
                    sample_bd_coord, sample_bd_lam, sample_origin_lam,
                    edge_indices, ext_coord, ext_lam, simple_indicator,
                    embed, embed_indices, embed_n,
                    n, cells, nei3cells, M, cp_pairs)

    return integ




def _check_gradient(y, weight, sample_coord1, sample_coord2, sample_angle, start_index,
                sample_bd_coord, sample_bd_lam, sample_origin_lam,
                edge_indices, ext_coord, ext_lam, simple_indicator,
                embed, embed_indices, embed_n,
                n, cells, nei3cells, M, cp_pairs,
                old_bd_coord, old_bd_lam, old_bd_y, max_o, max_o_lam, support_list, orthant_coords):
    integ = 0

    for i in range(15):
        supports = support_list[i]
        if supports is None:
            continue
        integ += _calculate_integral(orthant_coords[i], supports)


    llh = - weight @ y

    print(f"integral: {integ}")
    print(f"objective: {llh + integ}")

    #subgradient part
    grad = -weight
    J_terms_bd = [np.array([]) for i in range(10)]
    #J_terms_sample = np.zeros(n)
    J_term_origin = 0

    for edge in edge_indices:
        J_terms_bd[edge] = np.zeros(len(old_bd_coord[edge]))

    for i in range(15):
        cell = cells[i]
        supports = support_list[i]
        n_sample = start_index[i+1] - start_index[i]
        n_bd1 = len(old_bd_coord[cell[0]])
        n_bd2 = len(old_bd_coord[cell[1]])
        coords = orthant_coords[i]
        if supports is None:
            continue
        #print(_area(coords, supports))
        #input("AREA")
        for l in range(supports.shape[0]):
            support = supports[l]
            for k in range(3):
                index_in_o = support[k]
                a = _calculate_subgrad_comp(support[k], support[(k+1)%3], support[(k+2)%3], coords)
                if index_in_o < n_sample:
                    # sample point
                    grad[start_index[i] + index_in_o] += a
                elif index_in_o < n_sample + n_bd1:
                    # bd1 point
                    J_terms_bd[cell[0]][index_in_o - n_sample] += a
                elif index_in_o < n_sample + n_bd1 + n_bd2:
                    # bd2 point
                    J_terms_bd[cell[1]][index_in_o - n_sample - n_bd1] += a
                else:
                    #origin
                    J_term_origin += a

    for i in range(10):
        grad += (old_bd_lam[i].toarray().T @ J_terms_bd[i]).reshape(-1)
    grad += (J_term_origin * max_o_lam.toarray()).reshape(-1)
    #print(grad)
    #input()

    return llh+integ, grad



def _obj_grad(y, weight, sample_coord1, sample_coord2, sample_angle, start_index,
                sample_bd_coord, sample_bd_lam, sample_origin_lam,
                edge_indices, ext_coord, ext_lam, simple_indicator,
                embed, embed_indices, embed_n,
                n, cells, nei3cells, M, cp_pairs, print_objective = False):
    old_bd_coord, old_bd_lam, old_bd_y, max_o, max_o_lam, support_list, orthant_coords, hull_list = _least_concave_func(
                    y, sample_coord1, sample_coord2, sample_angle, start_index,
                    sample_bd_coord, sample_bd_lam, sample_origin_lam,
                    edge_indices, ext_coord, ext_lam, simple_indicator,
                    embed, embed_indices, embed_n,
                    n, cells, nei3cells, M, cp_pairs)

    #print(old_bd_coord)
    #print(old_bd_y)
    #print(max_o)
    #print(old_bd_lam)
    #input()
    #print(sample_coord1[[0,43,61]])
    #print(sample_coord2[[0,43,61]])
    #print(cells[4], cells[7])
    #print(y[0],y[43],y[61])
    #print(max_o_lam)
    #input()
    #print(y)

    integ = 0

    for i in range(15):
        supports = support_list[i]
        if supports is None:
            continue
        integ += _calculate_integral(orthant_coords[i], supports)


    llh = - weight @ y
    if print_objective:
        print(f"integral: {integ:.5f},    objective: {(llh + integ):.5f}")

    #subgradient part
    grad = -weight
    J_terms_bd = [np.array([]) for i in range(10)]
    #J_terms_sample = np.zeros(n)
    J_term_origin = 0

    for edge in edge_indices:
        J_terms_bd[edge] = np.zeros(len(old_bd_coord[edge]))

    for i in range(15):
        cell = cells[i]
        supports = support_list[i]
        n_sample = start_index[i+1] - start_index[i]
        n_bd1 = len(old_bd_coord[cell[0]])
        n_bd2 = len(old_bd_coord[cell[1]])
        coords = orthant_coords[i]
        if supports is None:
            continue
        #print(_area(coords, supports))
        #input("AREA")
        for l in range(supports.shape[0]):
            support = supports[l]
            for k in range(3):
                index_in_o = support[k]
                a = _calculate_subgrad_comp(support[k], support[(k+1)%3], support[(k+2)%3], coords)
                if index_in_o < n_sample:
                    # sample point
                    grad[start_index[i] + index_in_o] += a
                elif index_in_o < n_sample + n_bd1:
                    # bd1 point
                    J_terms_bd[cell[0]][index_in_o - n_sample] += a
                elif index_in_o < n_sample + n_bd1 + n_bd2:
                    # bd2 point
                    J_terms_bd[cell[1]][index_in_o - n_sample - n_bd1] += a
                else:
                    #origin
                    J_term_origin += a

    for i in range(10):
        grad += (old_bd_lam[i].toarray().T @ J_terms_bd[i]).reshape(-1)
    grad += (J_term_origin * max_o_lam.toarray()).reshape(-1)

    #print(np.max(np.abs(grad)), np.std(grad))
    #print(grad)
    #input()
    return llh+integ, grad


class _objective_hist():
    def __init__(self,weight, sample_coord1, sample_coord2, sample_angle, start_index,
                    sample_bd_coord, sample_bd_lam, sample_origin_lam,
                    edge_indices, ext_coord, ext_lam, simple_indicator,
                    embed, embed_indices, embed_n,
                    n, cells, nei3cells, M, cp_pairs):
        self.objective = []
        self.integral = []
        self.weight = weight
        self.sample_coord1 = sample_coord1
        self.sample_coord2 = sample_coord2
        self.sample_angle = sample_angle
        self.start_index = start_index
        self.sample_bd_coord = sample_bd_coord
        self.sample_bd_lam = sample_bd_lam
        self.sample_origin_lam = sample_origin_lam
        self.edge_indices = edge_indices
        self.ext_coord = ext_coord
        self.ext_lam = ext_lam
        self.simple_indicator = simple_indicator
        self.embed = embed
        self.embed_indices = embed_indices
        self.embed_n = embed_n
        self.n = n
        self.cells = cells
        self.nei3cells = nei3cells
        self.M = M
        self.cp_pairs = cp_pairs
    def save(self, y):
        old_bd_coord, old_bd_lam, old_bd_y, max_o, max_o_lam, support_list, orthant_coords, hull_list = _least_concave_func(
                        y, self.sample_coord1, self.sample_coord2, self.sample_angle, self.start_index,
                        self.sample_bd_coord, self.sample_bd_lam, self.sample_origin_lam,
                        self.edge_indices, self.ext_coord, self.ext_lam, self.simple_indicator,
                        self.embed, self.embed_indices, self.embed_n,
                        self.n, self.cells, self.nei3cells, self.M, self.cp_pairs)

        integ = 0

        for i in range(15):
            supports = support_list[i]
            if supports is None:
                continue
            integ += _calculate_integral(orthant_coords[i], supports)


        llh = - self.weight @ y

        self.objective.append(llh + integ)
        self.integral.append(integ)

class _y_hist():
    def __init__(self):
        self.y = []
    def save(self, y):
        self.y.append(y)




def lcmle_2dim(X, initialization = 'random', random_seed = None, weight = "uniform",print_objective=False,history=False ):
    # calculates two dimensional log-concave m.l.e.
    # INPUT:
    ## X : dataframe containing sample points. It should have following columns:
    ### edge1, edge2 : integer indicating the edge
    ### x1, x2 : coordinates of sample points
    ### angle : arctan(x2/x1)

    ## initialization : choice for initialization. Should be one of the following:
    ### 'random' : random initialization (gaussian)
    ### 'given' : specified initial value. X should have column named 'y' for that specified init value.

    ## random_seed: seed for random initialization.


    # sorting by orthants
    sort_ind = argsort_by_orthants(X)
    X = X.iloc[sort_ind]

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

    if weight == "uniform":
        weight = np.array([1/n for _ in range(n)])

    nei3cells = tuple_nei3cells()
    cp_pairs = cone_path_pairs()

    if initialization == 'random':
        if random_seed is None:
            y = np.random.normal(size=n) - 5
        else:
            np.random.seed(random_seed)
            y = np.random.normal(size=n) - 5
    else:
        y = X['y'].values

    #print(y)
    #input()

    #result_class = _objective_hist(weight, sample_coord1, sample_coord2, sample_angle, start_index,
    #                sample_bd_coord, sample_bd_lam, sample_origin_lam,
    #                edge_indices, ext_coord, ext_lam, simple_indicator,
    #                embed, embed_indices, embed_n,
    #                n, cells, nei3cells, M, cp_pairs)
    warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
    if history:
        y_his = _y_hist()
        res = minimize(_obj_grad, y, jac=True, args = (weight, sample_coord1, sample_coord2, sample_angle, start_index,
                        sample_bd_coord, sample_bd_lam, sample_origin_lam,
                        edge_indices, ext_coord, ext_lam, simple_indicator,
                        embed, embed_indices, embed_n,
                        n, cells, nei3cells, M, cp_pairs, print) ,method="BFGS", options={"maxiter":500}, callback=y_his.save)
        return res.x, y_his.y
    else:
        res = minimize(_obj_grad, y, jac=True, args = (weight, sample_coord1, sample_coord2, sample_angle, start_index,
                        sample_bd_coord, sample_bd_lam, sample_origin_lam,
                        edge_indices, ext_coord, ext_lam, simple_indicator,
                        embed, embed_indices, embed_n,
                        n, cells, nei3cells, M, cp_pairs, print) ,method="BFGS", options={"maxiter":500})
    return res.x


if __name__ == "__main__":

    check = 0
    if check == 1:
        with open("../temporary/temp.pickle", 'rb') as f:
            point_dict, hull_dict, support_dict, lam_dict, PBO = pickle.load(f)
        oo = orthant_to_ortind()
        old_bd_coord = [np.array([]) for _ in range(10)]
        old_bd_lam = [csr_matrix(np.empty(shape=(0,100))) for _ in range(10)]
        old_bd_y = [np.array([]) for _ in range(10)]

        support_list = [None for _ in range(15)]
        orthant_coords = [np.zeros((0,3)) for _ in range(15)]


        for i in range(10):
            Bi = PBO[(PBO[:,0] == i) * (PBO[:,1]==i)]
            if len(Bi)>0:
                old_bd_coord[i] = Bi[:,2]
                old_bd_y[i] = Bi[:,5]
                temp = np.zeros((len(Bi), 100))
                for l in range(len(Bi)):
                    boundary_index = int(Bi[l, 6])
                    temp[l]= np.array(list(lam_dict[boundary_index].values()))
                old_bd_lam[i] = csr_matrix(temp)

        max_o = PBO[PBO[:,0]==-1][0,5]
        max_o_lam = csr_matrix(np.array(list(lam_dict[-1].values())))

        for key in support_dict.keys():
            temp = support_dict[key] - 1
            temp[temp==-1] = point_dict[key].shape[0]-1
            support_list[oo[key]] = temp
            temp2 = point_dict[key]
            orthant_coords[oo[key]] = vstack((temp2[1:,[2,3,5]], temp2[0,[2,3,5]]))

        X = pd.read_csv("../data/sample_data_100_case4.csv")
        sort_ind = argsort_by_orthants(X)
        X = X.iloc[sort_ind]

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

        np.random.seed(1)
        y = np.random.normal(size=n) - 5
        y = np.load("../temporary/y.npy")
        weight = np.array([1/n for i in range(n)])

        _check_gradient(y, weight, sample_coord1, sample_coord2, sample_angle, start_index,
                        sample_bd_coord, sample_bd_lam, sample_origin_lam,
                        edge_indices, ext_coord, ext_lam, simple_indicator,
                        embed, embed_indices, embed_n,
                        n, cells, nei3cells, M, cp_pairs,
                        old_bd_coord, old_bd_lam, old_bd_y, max_o, max_o_lam, support_list, orthant_coords)
    else:
        X = pd.read_csv("../data/sample_data_100_case4.csv")
        #X['y'] = np.load("../temporary/y.npy")
        #optimize_2dim(X, initialization = 'given')
        print(X); input()
        optimize_2dim(X, random_seed = 1)
