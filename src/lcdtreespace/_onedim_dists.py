from scipy import stats
import numpy as np
from scipy.stats import norm

class _coalescent_1dim(stats.rv_continuous):
    # 象限1はプラス方向，象限2と3は合わせてマイナス方向
    # T has to be strictly positive
    def _pdf(self, x, T):
        if x<0:
            px = 2/3 * np.exp(x-T)
        else:
            px = -1/6 * np.exp(-x-T) + 1/2 * np.exp(-x+T-2 * max(0, T-x))
        return px
    def _cdf(self, x, T):
        if x<0:
            cx = 2/3 * np.exp(x-T)
        elif x<T:
            cx = np.exp(-T)/6 * (1-np.exp(-x)) * (3*np.exp(x)-1) + 2/3 * np.exp(-T)
        else:
            cx = 1/6 * (np.exp(-x-T) - np.exp(-2*T)) +1/2*(1 - np.exp(-x+T)) +  1/6 * (1-np.exp(-T)) * (3-np.exp(-T)) + 2/3 * np.exp(-T)
        return cx

class coalescent_1dim():
    """ Density object corresponding to simple coalescent process.

    Attributes
    ----------
    T : float
        The length of internal edge of the species tree.
    """
    def __init__(self,T):
        """
        Parameters
        ----------
        T : float
            The length of internal edge of the species tree.
        """
        self.T = T
        self.rv = _coalescent_1dim()

    def pdf(self,x,cell):
        """ Returns the value of the density value at a point.

        Parameters
        ----------
        x : coordinate of the point.
        cell : orthant of the point.

        Returns
        -------
        float
            Density at the point ``x`` in the orthant ``cell``.
        """
        T = self.T
        if x==0:
            px = self.rv._pdf(x, T=self.T)
        elif cell == 0:
            px = self.rv._pdf(x, T=self.T)
        else:
            px = self.rv._pdf(-x, T=self.T)/2
        return px
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
        x : numpy.ndarray
            Coordinates of sample points.
        ort : numpy.ndarray
            Orthants of sample points.
        """
        if seed is not None:
            np.random.seed(seed)
        x = self.rv.rvs(size=size, T=self.T)
        labels = np.zeros(size)
        for j in range(size):
            if x[j] > 0:
                continue
            else:
                labels[j] = np.random.choice([1,2])
        x0 = np.sort(x[labels==0])
        x1 = np.sort(-x[labels==1])
        x2 = np.sort(-x[labels==2])
        x = np.concatenate([x0,x1,x2])
        labels = np.concatenate([ [0 for j in range(len(x0))], [1 for j in range(len(x1))], [2 for j in range(len(x2))] ])
        return x, labels


class _normal_bend_1dim(stats.rv_continuous):
    # 象限1はプラス方向，象限2と3は合わせてマイナス方向
    # mu has to be strictly positive
    def _pdf(self, x, mu, sigma,k):
        normalize_factor = 1.0/(2.0*np.pi*sigma**2)**(1/2)
        if x >= 0:
            px = normalize_factor * (np.exp(-(x-mu)**2/(2*sigma**2)) - np.exp(-(x+mu)**2/(2*sigma**2)) * (k-2)/k )
        else:
            px = 2*(k-1)/k * normalize_factor * np.exp(-(x-mu)**2/(2*sigma**2))
        return px
    def _cdf(self, x, mu, sigma,k):
        if x<=0:
            cx = 2*(k-1)/k * norm.cdf(x,loc=mu,scale=sigma)
        else:
            cx = 2*(k-1)/k * norm.cdf(0,loc=mu,scale=sigma) + norm.cdf(x,loc=mu,scale=sigma) -norm.cdf(0,loc=mu,scale=sigma)\
            - norm.cdf(x,loc=-mu,scale=sigma)* (k-2)/k + norm.cdf(0,loc=-mu,scale=sigma)* (k-2)/k
        return cx


class normal_bend_1dim():
    """ normal density object that bends at the origin on 1dim tree space or more general space of k-spiders.

    This density corresponds to the transition density of the Brownian motion on k-spider.

    Attributes
    ----------
    mu : float
        Position of the starting point for Brownian motion.
    sigma : float
        Variance-like parameter.
    k : int
        Number of orthants. (Number of 'spiders'.)
        In case of one dimensional tree space, ``k`` should be 3.
    """
    def __init__(self,mu,sigma,k):
        """
        Parameters
        ----------
        mu : float
            Position of the starting point for Brownian motion.
        sigma : float
            Variance-like parameter.
        k : int
            Number of orthants. (Number of 'spiders'.)
            In case of one dimensional tree space, ``k`` should be 3.
        """
        self.mu = mu
        self.sigma = sigma
        self.k = k
        self.rv = _normal_bend_1dim()

    def pdf(self, x, cell):
        """ Returns the value of the density value at a point.

        Parameters
        ----------
        x : coordinate of the point.
        cell : orthant of the point.

        Returns
        -------
        float
            Density at the point ``x`` in the orthant ``cell``.
        """
        mu = self.mu; sigma = self.sigma; k = self.k
        if x==0:
            px = self.rv._pdf(x,mu,sigma,k)
        elif cell == 0:
            px = self.rv._pdf(x,mu,sigma,k)
        else:
            px = self.rv._pdf(-x,mu,sigma,k)/(k-1)
        return px
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
        x : numpy.ndarray
            Coordinates of sample points.
        ort : numpy.ndarray
            Orthants of sample points.
        """
        if seed is not None:
            np.random.seed(seed)
        x = self.rv.rvs(size=size, mu=self.mu, sigma = self.sigma, k = self.k)
        labels = np.zeros(size)
        for j in range(size):
            if x[j] > 0:
                continue
            else:
                labels[j] = np.random.choice([t for t in range(1,self.k)])
        nums = np.zeros(self.k).astype(int)
        x_sort = np.sort(x[labels==0]); nums[0] = len(x_sort)
        for l in range(1,self.k):
            add = np.sort(-x[labels==l])
            nums[l] = len(add)
            x_sort = np.append(x_sort,add)
        labels = np.concatenate([ [l for t in range(nums[l])] for l in range(self.k) ])
        return x_sort, labels


class _normal_1dim(stats.rv_continuous):
    def _pdf(self, x, mu, sigma):
        plus = norm.cdf(-mu, loc=0, scale=sigma)
        normalize_factor = 1.0/(2.0*np.pi*sigma**2)**(1/2)
        normalize_factor = normalize_factor/(1+plus)
        if x>=0:
            px = normalize_factor * np.exp(-(x-mu)**2/(2*sigma**2))
        else:
            px = normalize_factor * np.exp(-(x-mu)**2/(2*sigma**2)) * 2
        return px

    def _cdf(self, x, mu, sigma):
        plus = norm.cdf(-mu, loc=0, scale=sigma)
        if x<0:
            cx = norm.cdf(x, loc=mu, scale=sigma) / (1 + plus) *2
        else:
            cx = norm.cdf(x, loc=mu, scale=sigma)/(1+plus)  + norm.cdf(0, loc=mu, scale=sigma)/(1+plus)
        return cx



class normal_1dim():
    """ normal-like density object in 1dim tree space.

    The density is proportional to exp(-d(x,mu)^2/sigma^2) for x in supported orthants.

    Attributes
    ----------
    mu : float
        Position of the center.
    sigma : float
        Variance-like parameter.
    """
    def __init__(self, mu, sigma):
        """
        Parameters
        ----------
        mu : float
            Position of the center.
        sigma : float
            Variance-like parameter.
        """
        self.mu = mu
        self.sigma = sigma
        self.rv = _normal_1dim()
    def pdf(self, x, cell):
        """ Returns the value of the density value at a point.

        Parameters
        ----------
        x : coordinate of the point.
        cell : orthant of the point.

        Returns
        -------
        float
            Density at the point ``x`` in the orthant ``cell``.
        """
        plus = norm.cdf(-self.mu, loc=0, scale=self.sigma)
        normalize_factor = 1.0/(2.0*np.pi*self.sigma**2)**(1/2)
        normalize_factor = normalize_factor/(1+plus)
        if cell == 0:
            px = self.rv._pdf(x, self.mu, self.sigma)
        else:
            px = normalize_factor * np.exp(-(-x-self.mu)**2/(2*self.sigma**2))
        return px
    def sample(self,size, seed=None):
        """ Sample from the density.

        parameters
        ----------
        size : int
            sample size.
        seed : int
            random seed.

        Returns
        -------
        x : numpy.ndarray
            Coordinates of sample points.
        ort : numpy.ndarray
            Orthants of sample points.
        """
        if seed is not None:
            np.random.seed(seed)
        x = self.rv.rvs(size=size, mu=self.mu, sigma=self.sigma)
        labels = np.zeros(size)
        for j in range(size):
            if x[j] > 0:
                continue
            else:
                labels[j] = np.random.choice([1,2])
        x0 = np.sort(x[labels==0])
        x1 = np.sort(-x[labels==1])
        x2 = np.sort(-x[labels==2])
        x = np.concatenate([x0,x1,x2])
        labels = np.concatenate([ [0 for j in range(len(x0))], [1 for j in range(len(x1))], [2 for j in range(len(x2))] ])
        return x, labels


class _exponential_1dim(stats.rv_continuous):
    # 中心からの距離$x$に対しての密度．
    def _pdf(self, x, mu, lam):
        normalize_factor = 1 + np.exp(-lam * mu)
        if x>= mu:
            px = 0
        elif x >= 0:
            px = lam * np.exp(-lam * (mu-x)) / normalize_factor
        else:
            px = 2 * lam * np.exp(-lam * (mu-x)) / normalize_factor
        return px
    def _cdf(self, x, mu, lam):
        normalize_factor = 1 + np.exp(-lam * mu)
        if x<0:
            cx = np.exp(-lam * (mu-x))/normalize_factor * 2
        elif x < mu:
            cx = (np.exp(-lam * (mu-x)) + np.exp(-lam * mu) )/normalize_factor
        else:
            cx = 1
        return cx

class exponential_1dim():
    """
    Exponential-like density in 1dim tree space.

    Attributes
    ----------
    mu : float
        Location parameter.
    lam : float
        Rate parameter
    """
    def __init__(self, mu, lam):
        """
        Parameters
        ----------
        mu : float
            Location parameter.
        lam : float
            Rate parameter
        """
        self.mu = mu
        self.lam = lam
        self.rv = _exponential_1dim()
    def pdf(self, x, cell):
        """ Returns the value of the density value at a point.

        Parameters
        ----------
        x : coordinate of the point.
        cell : orthant of the point.

        Returns
        -------
        float
            Density at the point ``x`` in the orthant ``cell``.
        """
        normalize_factor = 1 + np.exp(-self.lam * self.mu)
        if cell == 0:
            if x<=1:
                px = self.lam * np.exp(-self.lam * (self.mu-x)) / normalize_factor
            else:
                px = 0
        else:
            px = self.lam * np.exp(-self.lam * (self.mu+x)) / normalize_factor
        return px
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
        x : numpy.ndarray
            Coordinates of sample points.
        ort : numpy.ndarray
            Orthants of sample points.
        """
        if seed is not None:
            np.random.seed(seed)
        x = self.rv.rvs(size=size, mu=self.mu, lam=self.lam)
        labels = np.zeros(size)
        for j in range(size):
            if x[j] > 0:
                continue
            else:
                labels[j] = np.random.choice([1,2])
        x0 = np.sort(x[labels==0])
        x1 = np.sort(-x[labels==1])
        x2 = np.sort(-x[labels==2])
        x = np.concatenate([x0,x1,x2])
        labels = np.concatenate([ [0 for j in range(len(x0))], [1 for j in range(len(x1))], [2 for j in range(len(x2))] ])
        return x, labels


'''
class exponential_1dim(stats.rv_continuous):
    # 中心からの距離$x$に対しての密度．
    def _pdf(self, x, mu, lam):
        normalize_factor = 1 + np.exp(-lam * mu)
        if x < 0:
            px = 0
        elif x < mu:
            px = lam * np.exp(-lam * x) / normalize_factor
        else:
            px = 2 * lam * np.exp(-lam * x) / normalize_factor
        return px
    def _cdf(self, x, mu, lam):
        normalize_factor = 1 + np.exp(-lam * mu)
        if x<0:
            cx = 0
        elif x < mu:
            cx = (1 - np.exp(-lam*x))/normalize_factor
        else:
            cx = (1 - np.exp(-lam*x))/normalize_factor + (np.exp(-lam*mu) - np.exp(-lam*x))/normalize_factor
        return cx
'''
