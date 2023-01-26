import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.collections as mc
import matplotlib.cm as cm

import numpy as np

from ._utils import *


def plot_petersen(X, c=None):
    """Scatter plot points in 2dim tree space on petersen graph.

    Parameters
    ----------
    X : pandas.DataFrame
        Sample points. See :py:func:`lcmle_2dim` for the required format.
    c : None or numpy.ndarray
        Colors for each sample point. Should be of the same length as ``X``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure of the plot.
    """
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

    for i in range(10):
        if i==3:
            ax.annotate(i, ((points + points_inside)[i][0], (points + points_inside)[i][1]) , textcoords = "offset points", xytext=(3,5))
        elif i==5:
            ax.annotate(i, ((points + points_inside)[i][0], (points + points_inside)[i][1]) , textcoords = "offset points", xytext=(3,5))
        elif i==7:
            ax.annotate(i, ((points + points_inside)[i][0], (points + points_inside)[i][1]) , textcoords = "offset points", xytext=(-5,5))
        elif i==2:
            ax.annotate(i, ((points + points_inside)[i][0], (points + points_inside)[i][1]) , textcoords = "offset points", xytext=(-2,7))
        else:
            ax.annotate(i, ((points + points_inside)[i][0], (points + points_inside)[i][1]) , textcoords = "offset points", xytext=(0,5))

    #P = np.load('results/clustering2_result/iter99size200_seed2_P.npy')
    #labels = (P[:,0] < P[:,1]).astype(int)

    vertices = points + points_inside
    #c = []
    x_coordinate = []
    y_coordinate = []
    for i in range(len(X)):
        point = X.iloc[i]
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
    return fig


def plot_density_2dim(density, xmax, ymax):
    """Heatmap of the density in 2dim tree space on each orthant.

    Parameters
    ----------
    density : 2 dimensional density object
        Should be one of the followings:

            - :py:class:`kernel_density_estimate_2dim`
            - :py:class:`logconcave_density_estimate_2dim`
            - :py:class:`normal_centered_2dim`
            - :py:class:`normal_uncentered_2dim`
    xmax, ymax : float or numpy.ndarray
        Maximum value for x-axis and y-axis.
        if numpy.ndarray is provided, then the length should be 15 (the number of orthants in 2dim tree space).

    Returns
    -------
    matplotlib.figure.Figure
        Figure of the plot.
    """
    # plot 2d density on each orthant
    # INPUTS:
    ## density: density object that has method pdf(x1,x2,cell1,cell2)
    ## xmax, xmin, ymax, ymin : float or ndarray of length 15.

    xmax = np.asarray(xmax); ymax = np.asarray(ymax)
    if xmax.ndim == 0:
        xmax = np.repeat(xmax, 15)
    if ymax.ndim == 0:
        ymax = np.repeat(ymax, 15)

    cells = tuple_2dcells()

    fig, axes = plt.subplots(nrows = 5, ncols = 3, figsize = (10,20))

    vmin = 0

    xxs = []; yys = []; ffs = []
    pdf_vec = np.vectorize(density.pdf, excluded=[2,3])

    mesh_size = 100

    for i in range(15):
        #print(i)
        cell = cells[i]
        xmax_i = xmax[i]; ymax_i = ymax[i]
        step_x = xmax_i/mesh_size; step_y = ymax_i/mesh_size
        xx,yy = np.mgrid[ 0:xmax_i:step_x, 0:ymax_i:step_y ]
        xxs.append(xx); yys.append(yy)
        ff = pdf_vec(xx,yy, cell[0], cell[1])
        ffs.append(ff)
    vmaxs = [np.max(item) for item in ffs]
    vmax = np.max(vmaxs)

    for i in range(15):
        #print(i)
        # plot density at each orthant
        cell = cells[i]
        ax = axes[i//3, i%3]
        xmax_i = xmax[i]; ymax_i = ymax[i]
        #step_x = xmax_i/mesh_size; step_y = ymax_i/mesh_size
        #xx,yy = np.mgrid[ 0:xmax_i:step_x, 0:ymax_i:step_y ]
        ax.set_xlim(0, xmax_i)
        ax.set_ylim(0, ymax_i)
        ax.set_xlabel(f'cell [{cell[0]}, {cell[1]}]', fontsize = 10, loc = 'right')
        ax.set_ylabel(None)


        #pdf_vec = np.vectorize(density.pdf, excluded=[2,3])
        #ff = pdf_vec(xx,yy, cell[0], cell[1])
        '''
        for i in range(1000):
            for j in range(1000):
                ff[i,j] = density.pdf(xx[i,j], yy[i,j], cell[0], cell[1])
        '''

        #conf = ax.contourf(xx, yy, ff, cmap='coolwarm')
        im = ax.imshow(ffs[i], cmap='Reds', extent=[0, xmax_i, 0, ymax_i], vmin = vmin, vmax = vmax, origin="lower")
        #con = ax.contour(xx, yy, ff, colors='k')
        #ax.clabel(con, inline=1, fontsize=10)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig
    #fig.savefig("a.png")

def plot_scatter_2dim(X, xmax, ymax, c=None):
    """Scatter plot points in 2dim tree space on each orthant.

    Parameters
    ----------
    X : pandas.DataFrame
        Sample points. See :py:func:`lcmle_2dim` for the required format.
    xmax, ymax : float or numpy.ndarray
        Maximum value for x-axis and y-axis.
        if numpy.ndarray is provided, then the length should be 15 (the number of orthants in 2dim tree space).
    c : None or numpy.ndarray
        Colors for each sample point. Should be of the same length as ``X``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure of the plot.
    """
    # plot 2d density on each orthant
    # INPUTS:
    ## density: density object that has method pdf(x1,x2,cell1,cell2)
    ## xmax, xmin, ymax, ymin : float or ndarray of length 15.

    xmax = np.asarray(xmax); ymax = np.asarray(ymax)
    if xmax.ndim == 0:
        xmax = np.repeat(xmax, 15)
    if ymax.ndim == 0:
        ymax = np.repeat(ymax, 15)

    cells = tuple_2dcells()

    fig, axes = plt.subplots(nrows = 5, ncols = 3, figsize = (10,20))

    if c is not None:
        vmin = np.min(c)
        vmax = np.max(c)

    for i in range(15):
        #print(i)
        # plot density at each orthant
        cell = cells[i]
        X_i = X[(X['edge1'] == cell[0]) & (X['edge2'] == cell[1])]
        if X_i.shape[0] == 0:
            continue
        indices = X_i.index.values
        ax = axes[i//3, i%3]
        xmax_i = xmax[i]; ymax_i = ymax[i]
        #step_x = xmax_i/mesh_size; step_y = ymax_i/mesh_size
        #xx,yy = np.mgrid[ 0:xmax_i:step_x, 0:ymax_i:step_y ]
        ax.set_xlim(0, xmax_i)
        ax.set_ylim(0, ymax_i)
        ax.set_xlabel(f'cell [{cell[0]}, {cell[1]}]', fontsize = 10, loc = 'right')
        ax.set_ylabel(None)


        #pdf_vec = np.vectorize(density.pdf, excluded=[2,3])
        #ff = pdf_vec(xx,yy, cell[0], cell[1])
        '''
        for i in range(1000):
            for j in range(1000):
                ff[i,j] = density.pdf(xx[i,j], yy[i,j], cell[0], cell[1])
        '''

        #conf = ax.contourf(xx, yy, ff, cmap='coolwarm')
        if c is None:
            ax.scatter(X_i['x1'].values, X_i['x2'].values)
        else:
            ax.scatter(X_i['x1'].values, X_i['x2'].values, c = c[indices],vmin=vmin,vmax=vmax)
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(im, cax=cbar_ax)
    #fig.savefig("b.png")
    return fig
