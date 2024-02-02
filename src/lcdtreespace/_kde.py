from ._utils import *
import numpy as np
from numpy.linalg import norm
from numpy import hstack, where, argmin, arctan2, exp, sqrt
import scipy.stats as stats
from bisect import bisect
import itertools
import numbers
import sys

def _geodesic_dist2d(sample_coord1, sample_coord2, sample_angle, start_index, new_coord1,new_coord2, new_angle, cell_new, cells, lenmat):
    # not efficient yet
    n = len(sample_coord1)
    dists = np.zeros(n)
    PI = np.pi
    PI_2 = np.pi/2
    for j in range(15):
        cell_j = cells[j]
        j_start = start_index[j]
        j_end = start_index[j+1]
        if (j_end==j_start):continue;
        num_j = j_end-j_start
        dist_ij = np.array([lenmat[cell_new[0], cell_j[0]], lenmat[cell_new[0], cell_j[1]],
                    lenmat[cell_new[1], cell_j[0]],lenmat[cell_new[1], cell_j[1]]], dtype=np.int8)
        index_min = argmin(dist_ij)
        n_ort_btwn = dist_ij[index_min]
        l, r = index_min//2, index_min % 2
        if cell_new[0]==cell_j[0] and cell_new[1]==cell_j[1]:
            distance = norm(hstack(((sample_coord1[j_start:j_end] - new_coord1).reshape(-1,1), (sample_coord2[j_start:j_end] - new_coord2).reshape(-1,1))), axis=1)
            dists[j_start:j_end] = distance
        elif n_ort_btwn==0:
            sample_coords = [sample_coord1, sample_coord2]; new_coords = [new_coord1, new_coord2]
            coord1 = sample_coords[r][j_start:j_end] - new_coords[l]
            coord2 = sample_coords[1-r][j_start:j_end] + new_coords[1-l]
            distance = norm(hstack((coord1.reshape(-1,1), coord2.reshape(-1,1))), axis=1)
            dists[j_start:j_end] = distance
        elif n_ort_btwn == 1:
            if l: #if l==1
                x_new = new_coord1;
                y_new = new_coord2; ## y is the closest coordinate
                new_angle_to_closest_edge = PI_2 - new_angle
            else:
                x_new = new_coord2;
                y_new = new_coord1; ## y is the closest coordinate
                new_angle_to_closest_edge = new_angle
            if r: ## if r==1
                x_j = sample_coord2[j_start:j_end]; ## x is the closest coordinate
                y_j = sample_coord1[j_start:j_end];
                j_angle_to_closest_edge = PI_2 - sample_angle[j_start:j_end]
            else:
                x_j = sample_coord1[j_start:j_end];
                y_j = sample_coord2[j_start:j_end]; ## x is the closest coordinate
                j_angle_to_closest_edge = sample_angle[j_start:j_end]
            ij_angle = j_angle_to_closest_edge + new_angle_to_closest_edge
            not_cone_path_row = where(ij_angle < PI_2)[0]
            len_ncpr = len(not_cone_path_row)
            if len_ncpr > 0:
                distance = norm(  hstack( ( (x_j[not_cone_path_row] + x_new).reshape(-1,1) , (y_j[not_cone_path_row] + y_new).reshape(-1,1) ) )  ,axis=1)
                dists[not_cone_path_row + j_start] = distance

            cone_path_row = where(ij_angle >= PI_2)[0]
            len_cpr = len(cone_path_row)
            if len_cpr > 0:
                norm_new = sqrt(x_new**2 + y_new**2)
                norm_j = sqrt(x_j[cone_path_row]**2 + y_j[cone_path_row]**2)
                dists[cone_path_row + j_start] = norm_new + norm_j
        elif n_ort_btwn == 2:
            norm_new = sqrt(new_coord1**2 + new_coord2**2)
            norm_j = sqrt(sample_coord2[j_start:j_end]**2 + sample_coord1[j_start:j_end]**2)
            dists[j_start:j_end] = norm_new + norm_j
    return dists

def _geodesic_dist2d2(a_cell, a_coord1, a_coord2, a_angle, b_cell, b_coord1, b_coord2, b_angle, cells, lenmat):
    dist_ab = np.array([lenmat[a_cell[0], b_cell[0]], lenmat[a_cell[0], b_cell[1]],
                lenmat[a_cell[1], b_cell[0]],lenmat[a_cell[1], b_cell[1]]], dtype=np.int8)
    index_min = argmin(dist_ab)
    n_ort_btwn = dist_ab[index_min]
    l, r = index_min//2, index_min % 2
    if a_cell[0]==b_cell[0] and a_cell[1]==b_cell[1]:
        return np.sqrt( (a_coord1 - b_coord1)**2 + (a_coord2 - b_coord2)**2 )
    elif n_ort_btwn==0:
        a_coords = [a_coord1, a_coord2]; b_coords = [b_coord1, b_coord2]
        coord1 = b_coords[r] - a_coords[l]
        coord2 = b_coords[1-r] + a_coords[1-l]
        return np.sqrt( coord1 ** 2 + coord2 ** 2 )
    elif n_ort_btwn == 1:
        if l: #if l==1
            x_a = a_coord1;
            y_a = a_coord2; ## y is the closest coordinate
            a_angle_to_closest_edge = np.pi/2 - a_angle
        else:
            x_a = a_coord2;
            y_a = a_coord1; ## y is the closest coordinate
            a_angle_to_closest_edge = a_angle
        if r: ## if r==1
            x_b = b_coord2; ## x is the closest coordinate
            y_b = b_coord1;
            b_angle_to_closest_edge = np.pi/2 - b_angle
        else:
            x_b = b_coord1;
            y_b = b_coord2 ## x is the closest coordinate
            b_angle_to_closest_edge = b_angle
        ab_angle = a_angle_to_closest_edge + b_angle_to_closest_edge

        if ab_angle < np.pi/2:
            # not cone path
            return np.sqrt( (x_a + x_b)**2 + (y_a + y_b)**2 )
        else:
            # cone path
            return np.sqrt( x_a**2 + y_a**2 ) + np.sqrt( x_b**2 + y_b**2 )
    else:
        return np.sqrt( a_coord1**2 + a_coord2 ** 2 ) + np.sqrt( b_coord1 ** 2 + b_coord2 ** 2 )





def _create_distance_mat(sample_coord1, sample_coord2, sample_angle, start_index,cells, lenmat):
    # uses for bandwidth selection
    # not efficient, (but not too much of a problem)
    Dmat = np.zeros((len(sample_coord1), len(sample_coord1)))
    for i in range(15):
        i_start = start_index[i]; i_end = start_index[i+1]
        cell = cells[i]
        if i_end == i_start:
            continue
        for k in range(i_start, i_end):
            new_angle = sample_angle[k]
            dists = _geodesic_dist2d(sample_coord1, sample_coord2, sample_angle, start_index, sample_coord1[k],sample_coord2[k], new_angle, cell, cells, lenmat)
            Dmat[k] = dists
    return Dmat

def _bw_nn(Dmat, prop=0.2, tol=1e-6):
    out = np.quantile(Dmat, prop, axis=0)
    iszero = out < tol
    if iszero.sum() > 0:
        out[iszero] = np.apply_along_axis(lambda x: np.min(x[x>tol]), 1, Dmat[iszero])
    return out



def _bhv_exact(sample_coord1, sample_coord2, bw):
    sigma = bw/np.sqrt(2)
    c_a = 2 * np.pi * sigma**2 * stats.norm.cdf(sample_coord1/sigma) * stats.norm.cdf(sample_coord2/sigma)
    c_b1 = 2 * np.pi * sigma**2 * stats.norm.cdf(sample_coord1/sigma) * stats.norm.cdf(-sample_coord2/sigma)
    c_b2 = 2 * np.pi * sigma**2 * stats.norm.cdf(-sample_coord1/sigma) * stats.norm.cdf(sample_coord2/sigma)
    c_c = 2 * np.pi * sigma**2 * stats.norm.cdf(-sample_coord1/sigma) * stats.norm.cdf(-sample_coord2/sigma)
    ms = np.linalg.norm(np.hstack((sample_coord1.reshape(-1,1), sample_coord2.reshape(-1,1)))/sigma.reshape(-1,1), axis=1)
    c_d = np.pi * sigma**2 / 2 * ( (np.exp(-ms**2/2) - ms * np.sqrt(2*np.pi) * stats.norm.cdf(-ms)) )
    all = c_a + 2 * (c_b1 + c_b2) + 4 * c_c + 6 * c_d
    return all



def _kernel(x, cell, sample_coord1, sample_coord2, sample_angle, start_index, cells, lenmat , bw=1.0, bhvc=None, delta=2.0):
    new_angle = arctan2(x[1],x[0])
    geodesic_dists = _geodesic_dist2d(sample_coord1, sample_coord2, sample_angle, start_index, x[0],x[1], new_angle, cell, cells, lenmat)
    if bhvc is None:
        return (exp(-np.abs(geodesic_dists/bw)**delta) / bw).mean()
    else:
        return (exp(-np.abs(geodesic_dists/bw)**delta) / bhvc).mean()


class kernel_density_estimate_2dim():
    """Kernel density estimate object in 2dim tree space.
    """
    def __init__(self,X,bandwidth="nn",nn_prop=0.2):
        """
        Parameters
        ----------
        X : pandas.DataFrame
            Sample points. See :py:func:`lcmle_2dim` for the required format.
        bandwidth : float or string
            If float, bandwidth. 
            If string, it should be one of the followings:
                "nn": nearest neighbor approach. Bandwidth is set to the ``nn_prop`` quantile of distances to other points.
        nn_prop : float
            Quantile used for "nn" approach. Ignored if ``bandwidth`` is not "nn".
        """
        self.sample_coord1 = X['x1'].values
        self.sample_coord2 = X['x2'].values
        self.sample_angle = X['angle'].values
        self.start_index = get_start_indices(X)
        self.cells = tuple_2dcells()
        self.lenmat = b_to_b_lenmat()
        #self.Dmat = _create_distance_mat(self.sample_coord1, self.sample_coord2, self.sample_angle, self.start_index,self.cells, self.lenmat)
        if isinstance(bandwidth, numbers.Number):
            assert bandwidth>0, "bandwidth has to be positive"
            self.bw = np.array([bandwidth for i in range(X.shape[0])])
        elif isinstance(bandwidth, str):
            if bandwidth == "nn":
                self.Dmat = _create_distance_mat(self.sample_coord1, self.sample_coord2, self.sample_angle, self.start_index,self.cells, self.lenmat)
                self.bw = _bw_nn(self.Dmat, prop=nn_prop)
            else:
                sys.exit("Invalid bandwidth argument.")
        self.bhv_c = _bhv_exact(self.sample_coord1,self.sample_coord2, self.bw)

    def pdf(self,x1,x2,cell0,cell1):
        """ Returns the value of the density value at a point.

        Parameters
        ----------
        x1 : float
            First coordinate.
        x2 : float
            Second coordinate.
        cell0 : int
            First orthant.
        cell1 : int
            Second orthant.
            cell1 should have a larger value than cell0.

        Returns
        -------
        float
            Density at the point ``(x1,x2)`` in the orthant ``(cell0, cell1)``.
        """
        cell = (cell0, cell1)
        x = np.array([x1,x2])
        estimated_density = _kernel(x, cell, self.sample_coord1, self.sample_coord2, self.sample_angle, self.start_index, self.cells, self.lenmat, bw=self.bw, bhvc = self.bhv_c)
        return estimated_density

def _create_distance_mat_1dim(x, ort):
    n = len(x)
    n1 = len(x[ort==0]); n2 = len(x[ort==1])
    Dmat = np.zeros((n, n))
    for a, b in itertools.combinations([i for i in range(len(x))], 2):
        if bisect([n1,n1 + n2],a) == bisect([n1,n1 + n2],b):
            dist = np.abs(x[b] - x[a])
        else:
            dist = x[b] + x[a]
        Dmat[a,b] = Dmat[b,a] = dist
    return Dmat


class kernel_density_estimate_1dim():
    """Kernel density estimate object in 1dim tree space or more general space of k-spider.
    """
    def __init__(self, x, ort, n_ort, bandwidth="nn", nn_prop = 0.2):
        """
        Parameters
        ----------
        x : numpy.ndarray
            coordinates of sample points.
        ort : numpy.ndarray
            orthants that sample points belong to.
            Should have same length as ``x``.
        n_ort : int
            Number of orthants. (Number of 'spiders'.)
            In case of one dimensional tree space, ``n_ort`` should be 3.
        bandwidth : float or string
            If float, bandwidth. 
            If string, it should be one of the followings:
                "nn": nearest neighbor approach. Bandwidth is set to the ``nn_prop`` quantile of distances to other points.
        nn_prop : float
            Quantile used for "nn" approach. Ignored if ``bandwidth`` is not "nn".
        """
        self.x = x
        self.ort =  ort
        self.n_ort = n_ort
        if isinstance(bandwidth, numbers.Number):
            assert bandwidth>0, "bandwidth has to be positive"
            self.bw = np.array([bandwidth for i in range(len(x))])
        elif isinstance(bandwidth, str):
            if bandwidth == "nn":
                self.Dmat = _create_distance_mat_1dim(x, ort)
                self.bw = _bw_nn(self.Dmat, prop=nn_prop)
            else:
                sys.exit("Invalid bandwidth argument.")
    def pdf(self, x, cell):
        """ Returns the value of the density value at a point.

        Currently, it only supports when ``n_ort`` = 3.

        Parameters
        ----------
        x : coordinate of the point.
        cell : orthant of the point.

        Returns
        -------
        float
            Density at the point ``x`` in the orthant ``cell``.
        """
        if cell == 0:
            dist_x0 = np.abs(self.x[self.ort==0]-x)
            dist_x1 = self.x[self.ort==1]+x
            dist_x2 = self.x[self.ort==2]+x
        elif cell == 1:
            dist_x0 = self.x[self.ort==0]+x
            dist_x1 = np.abs(self.x[self.ort==1]-x)
            dist_x2 = self.x[self.ort==2]+x
        else:
            dist_x0 = self.x[self.ort==0]+x
            dist_x1 = self.x[self.ort==1]+x
            dist_x2 = np.abs(self.x[self.ort==2]-x)
        dists = np.concatenate((dist_x0, dist_x1, dist_x2))
        xs = np.concatenate((self.x[self.ort==0], self.x[self.ort==1], self.x[self.ort==2]))
        bw = np.concatenate((self.bw[self.ort==0], self.bw[self.ort==1], self.bw[self.ort==2]))
        bhvc = np.sqrt(2 * np.pi)*bw*(1 + stats.norm.cdf(-xs/ bw))
        return (np.exp(-np.abs(dists/bw)**2/2) / bhvc).mean()
