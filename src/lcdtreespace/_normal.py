import numpy as np
import pandas as pd
from scipy.stats import norm, rv_continuous, multivariate_normal
#from least_concave_2d import find_neighbors, list_2dcells, add_angle, geodesic_distance_2d
from scipy.integrate import quad, dblquad
from ._utils import *
from ._kde import _geodesic_dist2d2


class normal_centered_2dim():
    """ Centered normal-like density in two dimensional tree space.

    Attributes
    ----------
    cells : list of lists
        supported orthants.
    sigma : float
        variance-like parameter of the density.
        The density is proportional to exp(-d(x,0)^2/sigma^2) for x in supported orthants.
    """
    def __init__(self,cells,sigma):
        """
        parameters
        ----------
        cells : list of lists
            supported orthants.
        sigma : float
            variance-like parameter of the density.
            The density is proportional to exp(-d(x,0)^2) for x in supported orthants.
        """
        self.n_cell = len(cells)
        self.cells = np.array(cells)
        self.tuple_cells = [tuple(item) for item in self.cells]
        self.sigma = sigma
    def sample(self,size,seed=None):
        """ Sample from the density.

        parameters
        ----------
        size : int
            sample size.
        seed : int
            random seed.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing sample points.
        """
        if seed is not None:
            np.random.seed(seed)
        x_d = np.abs(np.random.multivariate_normal(mean=np.zeros(2),cov=np.eye(2) * self.sigma, size=size))
        labels = np.random.choice([i for i in range(self.n_cell)], size = size)
        labels = self.cells[labels]
        angles = np.arctan2(x_d[:,1], x_d[:,0])
        y = np.random.normal(size=size) - 5

        X = np.hstack((labels, x_d, angles.reshape(-1,1), y.reshape(-1,1)))
        X = pd.DataFrame(X, columns = ["edge1", "edge2", "x1", "x2", "angle", "y"])
        sort_ind = argsort_by_orthants(X)
        X = X.iloc[sort_ind].reset_index(drop=True)
        return X.astype({"edge1":"int32", "edge2":"int32"})
    def pdf(self, x1,x2,cell0,cell1):
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
            Density at the point (x1,x2) in the orthant (cell0, cell1).
        """
        if (cell0, cell1) not in self.tuple_cells:
            return 0
        else:
            return multivariate_normal(mean=np.zeros(2), cov=np.eye(2)*self.sigma).pdf(np.array([x1,x2]))*4/self.n_cell

class normal_uncentered_2dim():
    """ Uncentered normal-like density in two dimensional tree space.

    Attributes
    ----------
    cell : list of length 2
        The orthant to which the center belongs.
    mu : float
        The coordinate of the center point.
    sigma : float
        variance-like parameter of the density.
        The density is proportional to exp(-d(x,0)^2) for x in supported orthants.
    """
    def __init__(self, cell, mu, sigma):
        """
        Parameters
        ----------
        cell : list of length 2
            The orthant to which the center belongs.
        mu : float
            The coordinate of the center point.
        sigma : float
            variance-like parameter of the density.
            The density is proportional to exp(-d(x,0)^2) for x in supported orthants.
        """
        if cell[0] != cell[1]:
            self.cell = cell
            self.mu = mu
            self.alpha = np.arctan2(mu[1], mu[0])
            self.sigma = sigma
            self.c_a = 2 * np.pi * sigma**2 * norm.cdf(mu[0]/sigma) * norm.cdf(mu[1]/sigma)
            self.c_b1 = 2 * np.pi * sigma**2 * norm.cdf(mu[0]/sigma) * norm.cdf(-mu[1]/sigma)
            self.c_b2 = 2 * np.pi * sigma**2 * norm.cdf(-mu[0]/sigma) * norm.cdf(mu[1]/sigma)
            self.c_c = 2 * np.pi * sigma**2 * norm.cdf(-mu[0]/sigma) * norm.cdf(-mu[1]/sigma)
            ms = np.linalg.norm(mu/sigma)
            self.c_d = np.pi * sigma**2 / 2 * ( (np.exp(-ms**2/2) - ms * np.sqrt(2*np.pi) * norm.cdf(-ms)) )
            self.all = self.c_a + 2 * (self.c_b1 + self.c_b2) + 4 * self.c_c + 6 * self.c_d
            #self.norm = np.array([self.c_a, self.c_b1, self.c_b2, self.c_c, self.c_d])
            #self.norm = self.norm/np.sum(self.norm)
            self.prop = np.array([self.c_a, 2 * self.c_b1, 2 * self.c_b2, 4 * self.c_c, 6 * self.c_d])/self.all
    def sample(self, size, seed=None):
        """ Sample from the density.

        parameters
        ----------
        size : int
            sample size.
        seed : int
            random seed.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing sample points.
        """
        if seed is not None:
            np.random.seed(seed)
        labels = np.random.multinomial(n=size, pvals=self.prop)
        samples_a = _multivariate_norm(mu=self.mu,sigma=self.sigma).sample(size=labels[0])
        samples_b1 = _multivariate_norm(mu=[self.mu[0], -self.mu[1]] , sigma=self.sigma).sample(size=labels[1])
        samples_b2 = _multivariate_norm(mu=[-self.mu[0], self.mu[1]] , sigma=self.sigma).sample(size=labels[2])
        samples_c = _multivariate_norm(mu=[-self.mu[0], -self.mu[1]] , sigma=self.sigma).sample(size=labels[3])
        samples_d = _others(self.mu, self.sigma, self.c_d).sample(size=labels[4])

        samples_A = np.hstack(( np.array([self.cell for i in range(labels[0])]) , samples_a))

        b1_cells = [[ self.cell[0] , v ] for v in find_neighbors(self.cell[0]) if v != self.cell[1]]
        b1_choice = np.random.choice([0,1], size=labels[1])
        samples_B1 = np.hstack(( np.array( [ b1_cells[choice] for choice in b1_choice ] ) , samples_b1 ))

        b2_cells = [[ v, self.cell[1] ] for v in find_neighbors(self.cell[1]) if v != self.cell[0]]
        b2_choice = np.random.choice([0,1], size=labels[2])
        samples_B2 = np.hstack(( np.array( [ b2_cells[choice] for choice in b2_choice ] ) , samples_b2 ))

        c1_first = [v for v in find_neighbors(self.cell[0]) if v != self.cell[1]]
        c2_first = [v for v in find_neighbors(self.cell[1]) if v != self.cell[0]]
        c1_cells = [[ v, c1_first[0] ] for v in find_neighbors(c1_first[0]) if v != self.cell[0]] \
        + [[ v, c1_first[1] ] for v in find_neighbors(c1_first[1]) if v != self.cell[0]]
        c2_cells = [[ c2_first[0] , v ] for v in find_neighbors(c2_first[0]) if v != self.cell[1]] \
        + [[ c2_first[1] , v ] for v in find_neighbors(c2_first[1]) if v != self.cell[1]]

        samples_c_angle = [np.arctan2(sample[1], sample[0]) for sample in samples_c]
        samples_c_which = (np.array(samples_c_angle) < self.alpha).astype(int)
        c_choice = np.random.choice([0,1,2,3], size=labels[3])
        samples_C = np.hstack((  np.array( [ c1_cells[c_choice[i]] if samples_c_which[i] == 0 else c2_cells[c_choice[i]] for i in range(len(c_choice))] ) , samples_c ))

        existing_cells = [(min(item[0],item[1]), max(item[0],item[1])) for item in [self.cell]+b1_cells+b2_cells+c1_cells+c2_cells]
        all_cells_set = [tuple(item) for item in list_2dcells()]
        remaining_cells = list(set(all_cells_set)-set(existing_cells))

        samples_dtheta = np.random.uniform(0, 3*np.pi, size=labels[4])
        samples_D = []
        for i in range(len(samples_d)):
            if samples_dtheta[i] <= np.pi/2:
                samples_D.append([remaining_cells[0][0], remaining_cells[0][1], samples_d[i]*np.cos(samples_dtheta[i]), samples_d[i]*np.sin(samples_dtheta[i])])
            elif samples_dtheta[i] <= np.pi:
                samples_D.append([remaining_cells[1][0], remaining_cells[1][1], -samples_d[i]*np.cos(samples_dtheta[i]), samples_d[i]*np.sin(samples_dtheta[i])])
            else:
                q, mod = divmod(samples_dtheta[i], np.pi/2)
                choice_inc = int(q) - 2
                choice_01 = (mod > self.alpha).astype(int)
                cell_tmp, theta_tmp = (c2_cells[choice_inc], mod) if choice_01 == 1 else (c1_cells[choice_inc], mod)
                samples_D.append([cell_tmp[0], cell_tmp[1], samples_d[i] * np.cos(theta_tmp), samples_d[i] * np.sin(theta_tmp)])
        samples_D = np.array(samples_D)
        samples = np.vstack([item for item in [samples_A, samples_B1, samples_B2, samples_C, samples_D] if len(item) > 0])
        samples = np.apply_along_axis(lambda x: x if x[0] < x[1] else np.array([x[1],x[0],x[3],x[2]]),axis=1, arr=samples)

        angles = np.arctan2(samples[:,3], samples[:,2])
        y = np.random.normal(size=size) - 5

        X = np.hstack((samples, angles.reshape(-1,1), y.reshape(-1,1)))
        X = pd.DataFrame(X, columns = ["edge1", "edge2", "x1", "x2", "angle", "y"])
        sort_ind = argsort_by_orthants(X)
        X = X.iloc[sort_ind].reset_index(drop=True)

        return X.astype({"edge1":"int32", "edge2":"int32"})
    def pdf(self, x1, x2, cell0, cell1):
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
            Density at the point (x1,x2) in the orthant (cell0, cell1).
        """
        # cannot derive in general case: can only be valid when x1 or x2 strictly equals to zero
        if self.mu[0]==0:
            ind = 0
        elif self.mu[1]==0:
            ind = 1
        cell = [cell0, cell1]
        point = [cell[0],cell[1],x1,x2,np.arctan2(x2,x1),1]
        center = [self.cell[0], self.cell[1], self.mu[0], self.mu[1], self.alpha, 1]
        #px_unnormalized = np.exp(-geodesic_distance_2d(center, point)**2/(2*self.sigma**2))
        dd = _geodesic_dist2d2(self.cell, self.mu[0], self.mu[1], self.alpha, cell, x1, x2, np.arctan2(x2,x1), tuple_2dcells(), b_to_b_lenmat())
        px_unnormalized = np.exp(-dd**2/(2 * self.sigma**2))
        return px_unnormalized / self.all



class _multivariate_norm():
    def __init__(self, mu, sigma):
        self.mu = np.array(mu)
        self.sigma = sigma
    def sample(self, size):
        i = 0
        samples = []
        while True:
            sample = np.random.multivariate_normal(mean=self.mu, cov=self.sigma**2*np.eye(2))
            if sample[0] > 0 and sample[1] > 0:
                samples.append(sample)
                i+=1
            if i==size:
                break
        return np.array(samples)

class _others():
    def __init__(self,mu,sigma,c_d):
        self.mu = np.array(mu)
        self.sigma = sigma
        self.c_d = c_d
        self.ms = np.linalg.norm(mu/sigma)
    def sample(self, size):
        samples_r = _r_others(a=0).rvs(ms=self.ms, sigma=self.sigma, c_d = self.c_d, size=size)
        #print(quad(lambda x: _r_others(a=0).pdf(x, self.ms, self.sigma, self.c_d), 0, np.inf))
        return samples_r

class _r_others(rv_continuous):
    def _pdf(self, r, ms, sigma, c_d):
        normalize_factor = c_d / (np.pi * sigma**2 / 2)
        px = np.exp(-(r+ms)**2/2)*r/normalize_factor
        return px

'''
def visualize_points(x, c=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    thetas = [np.pi/2 + i * 2 * np.pi / 5 for i in range(5)]

    points = [(np.cos(theta), np.sin(theta)) for theta in thetas]
    points_inside= [(0.5*np.cos(theta), 0.5*np.sin(theta)) for theta in thetas]
    lines = [[points[i], points_inside[i]] for i in range(5)]
    lines_2 = [[points_inside[(2*i)%5], points_inside[(2*(i+1))%5]] for i in range(5)]
    lc = mc.LineCollection(lines + lines_2, colors='black', linewidth=1)
    p = pat.Polygon(xy = points, fc='white', ec='black', linewidth=1)
    ax.scatter([p[0] for p in points + points_inside], [p[1] for p in points + points_inside],zorder=2,c='black')
    ax.add_patch(p)
    ax.add_collection(lc)
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)

    #P = np.load('results/clustering2_result/iter99size200_seed2_P.npy')
    #labels = (P[:,0] < P[:,1]).astype(int)

    vertices = points + points_inside
    #c = []
    x_coordinate = []
    y_coordinate = []
    for i in range(len(x)):
        point = x.iloc[i]
        #label = labels[i]
        vertex0 = np.array(vertices[int(point['edge1'])])
        vertex1 = np.array(vertices[int(point['edge2'])])
        angle = point['angle']
        x_coordinate.append((1-2*angle/np.pi) * vertex0[0] + 2*angle/np.pi * vertex1[0])
        y_coordinate.append((1-2*angle/np.pi) * vertex0[1] + 2*angle/np.pi * vertex1[1])
        #c.append(label)
    if c is None:
        ax.scatter(x_coordinate, y_coordinate, zorder=3)
    else:
        ax.scatter(x_coordinate, y_coordinate, c=c, zorder=3)
    plt.axis('off')
    plt.show()
'''

if __name__ == "__main__":
    #samples1 = normal([0,1], np.array([1,1]), 1).sample(100, seed=0)
    #print(samples1)
    cells = np.array([[0,1], [1,6], [6,8], [3,8], [3,4], [0,4]])
    samples1 = normal_centered(cells,1).sample(100, seed=0)
    samples2 = normal_uncentered([0,1], np.array([0,1]), 1).sample(100,seed=0)
    print(samples1)
    print(samples2)
    #cells = np.array(list_2dcells())
    np.random.seed(0)
    x_d = np.random.multivariate_normal(mean=np.zeros(2),cov=np.eye(2), size=100)
    x_d = np.abs(x_d)
    labels = np.random.choice([i for i in range(len(cells))], size = 100)
    labels = cells[labels]
    x = np.array([[labels[i][0], labels[i][1], x_d[i][0], x_d[i][1]] for i in range(100)])
    X = pd.read_csv("data/case4/testcase_100_0.csv")
    print(X)
