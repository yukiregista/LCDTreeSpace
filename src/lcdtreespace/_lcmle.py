
import numpy as np
from ._kde import *
#import rpy2.robjects as ro
#from rpy2.robjects.packages import importr
#from rpy2.robjects import numpy2ri
from ._least_concave_func import _least_concave_func, _num_new_bds
from ._utils import *
from ._geodesic_sample import geodesic_sample
from ._link_convhull import _link_convex_hull
from ._twoDconvhull import _twoDconvhull
from bisect import bisect
from ._mle_1dim import tree_hull, tree_hull_bend

def _signed_area(a,b,c):
    # INPUTS
    ## a, b, c: ndarray of length 2
    # OUTPUT
    ## area : signed area of triangle abc

    ab = b-a
    ac = c-a
    area = (ab[0] * ac[1] - ab[1] * ac[0])/2
    return area

class logconcave_density_estimate_2dim():
    def __init__(self, y, X):
        sample_coord1 = X['x1'].values
        sample_coord2 = X['x2'].values
        sample_angle = X['angle'].values
        start_index = get_start_indices(X)

        n = len(y)
        cells = tuple_2dcells()
        lenmat = b_to_b_lenmat()
        M = _num_new_bds(start_index, cells)
        embed, embed_indices, embed_n = embed_sample_points_to_nei3cells(sample_coord1, sample_coord2, start_index)

        _sample_bd_coord, _sample_bd_lam, sample_origin_lam = geodesic_sample(sample_coord1, sample_coord2, sample_angle,start_index, cells)
        edge_indices = _link_convex_hull(sample_angle, start_index)
        ext_coord, ext_lam, simple_indicator = _twoDconvhull(sample_coord1, sample_coord2, sample_angle, start_index,
                            _sample_bd_coord, _sample_bd_lam,edge_indices)
        nei3cells = tuple_nei3cells()
        cp_pairs = cone_path_pairs()

        old_bd_coord, old_bd_lam, old_bd_y, self.max_o, max_o_lam, self.support_list, self.orthant_coords, hull_list = _least_concave_func(
                            y, sample_coord1, sample_coord2, sample_angle, start_index,
                            _sample_bd_coord, _sample_bd_lam, sample_origin_lam,
                            edge_indices, ext_coord, ext_lam, simple_indicator,
                            embed, embed_indices, embed_n,
                            n, cells, nei3cells, M, cp_pairs, max_iter = 100)
        self.oo = orthant_to_ortind()
        self.xmax = np.zeros(15)
        self.ymax = np.zeros(15); self.ymax.fill(-np.inf)
        for i in range(15):
            if len(self.orthant_coords[i] > 0):
                self.xmax[i] = np.max(self.orthant_coords[i][:,0])
                self.ymax[i] = np.max(self.orthant_coords[i][:,1])

        self.area_list = [None for i  in range(15)]
        for i in range(15):
            supports = self.support_list[i]
            if supports is not None:
                areas = np.zeros(supports.shape[0])
                for j in range(supports.shape[0]):
                    support = supports[j]
                    points = self.orthant_coords[i][support, :2]
                    areas[j] = _signed_area(points[0], points[1], points[2])
                self.area_list[i] = areas

    def pdf(self,x1,x2,cell0,cell1):
        ortind = self.oo[(cell0,cell1)]
        if self.support_list[ortind] is None:
            return 0
        if x1 > self.xmax[ortind] or x2 > self.ymax[ortind]:
            return 0
        if x1==0 and x2 == 0:
            return np.exp(self.max_o)

        x = np.array([x1,x2])
        supports = self.support_list[ortind]
        areas = self.area_list[ortind]
        points = self.orthant_coords[ortind]
        for i in range(supports.shape[0]):
            support = supports[i]
            pbc_area = _signed_area(x, points[support[1], :2], points[support[2], :2])/areas[i]
            apc_area = _signed_area(points[support[0], :2], x, points[support[2], :2])/areas[i]
            abp_area = _signed_area(points[support[0], :2], points[support[1], :2], x)/areas[i]
            #print("SUM:", pbc_area + apc_area + abp_area)
            if pbc_area >= 0 and apc_area >= 0 and abp_area >= 0:
                # inside the triangle
                return np.exp(pbc_area * points[support[0],2] + apc_area * points[support[1], 2] + abp_area * points[support[2], 2])
        return 0


class logconcave_density_estimate_1dim():
    def __init__(self, y, x, ort, n_label, bend=False):

        self.y = y
        self.x = x
        self.ort=  ort
        self.n_label = n_label
        if bend:
            hull_list, support_list, max_o, top_ort, bottom_ort, lam_index, lam = tree_hull_bend(y,x,ort,n_label)
        else:
            hull_list, support_list, max_o, top_ort, bottom_ort, lam_index, lam = tree_hull(y,x,ort,n_label)

        self.max_o = max_o
        self.support_list = support_list
        self.hull_list = hull_list

        self.xmaxs = np.zeros(n_label)
        for i in range(n_label):
            self.xmaxs[i] = hull_list[i].points[self.support_list[i][-1]][0]

    def pdf(self,x,cell):
        support = self.support_list[cell]
        hull = self.hull_list[cell]

        if x > self.xmaxs[cell] or x<0:
            px = 0
        elif x==0:
            px = np.exp(self.max_o)
        else:
            support_index = bisect(hull.points[support, 0], x)
            below = hull.points[support[support_index-1]]
            above = hull.points[support[support_index]]
            lam = (x-below[0])/(above[0]-below[0])
            px = np.exp((1-lam) * below[1] + lam * above[1])
        return px
