import numpy as np
#from scipy.optimize import linprog
from ._geodesic_sample import geodesic_sample
#from numpy import argmax, tile, where, arange
#from scipy.sparse import csr_matrix
#from numpy.linalg import inv
from ._utils import *
from collections import deque
import pandas as pd


def _link_convex_hull(sample_angle, start_index):
    # if len(sample_angle)==49:
    #     import pdb
    #     pdb.set_trace()
    ort_counts = start_index[1:] - start_index[:-1]
    frontier_angles = deque(sample_angle) # stores angle of each point in the frontier set
    A_ortind = []
    A_angles = list(sample_angle)
    for i in range(26):
        A_ortind.extend([i for j in range(start_index[i+1] - start_index[i])])
    frontier_ortind = deque(A_ortind) # stores orthant index of each point in the frontier set

    orthants = tuple_orthants()
    oo = orthant_to_ortind()
    while len(frontier_ortind) > 0:
        angle = frontier_angles.popleft()
        ortind = frontier_ortind.popleft()
        if ortind < 15:
            # It is the top-dimensional point
            cell = orthants[ortind]
            colors = ['white' for i in range(11)] # for managing DFS visit: last one corresponds to starting point
            parents = [None for i in range(11)] # for managing parents in DFS: last one corresponds to starting point
            angles = [None for i in range(11)] # for managing angles in DFS: last one corresponds to starting point
            survives = [False for i in range(10)] # if that vertex will be added to the queue
            angles[10] = 0

            # depth-first search
            angles[cell[0]] = angle; parents[cell[0]] = 10; colors[cell[0]] = 'gray'
            angles[cell[1]] = np.pi/2 - angle; parents[cell[0]] = 10; colors[cell[1]] = 'gray'
            parents, colors, angles, survives = _DFS_visit(cell[0], angles, colors, parents, survives, ort_counts, A_angles, start_index, oo)
            parents, colors, angles, survives = _DFS_visit(cell[1], angles, colors, parents, survives, ort_counts, A_angles, start_index, oo)
        else:
            colors = ['white' for i in range(10)] # for managing DFS visit
            parents = [None for i in range(10)] # for managing parents in DFS
            angles = [None for i in range(10)] # for managing angles in DFS:
            survives = [False for i in range(10)] # if that vertex will be added to F and x

            cell = orthants[ortind]
            # depth-first search
            angles[cell[0]] = 0; parents[cell[0]] = None; colors[cell[0]] = 'gray'
            parents, colors, angles, survives= _DFS_visit(cell[0], angles, colors, parents, survives, ort_counts, A_angles, start_index, oo)
        for i in range(10):
            if survives[i]:
                ortind_i = oo[(i,i)]
                if ort_counts[ortind_i]==0:
                    ort_counts[ortind_i] = 1
                    A_ortind.append(ortind_i)
                    A_angles.append(0)
                    frontier_ortind.append(ortind_i)
                    frontier_angles.append(0)
    edge_indices = []
    for i in range(10):
        if ort_counts[oo[(i,i)]] > 0:
            edge_indices.append(i)
    return edge_indices


def _DFS_visit(u, angles, colors, parents, survives, ort_counts, A_angles, start_index, oo):
    colors[u] = 'gray' # visiting
    if angles[u] >= np.pi/2:
        # cannot visit neighbors
        neighbors = find_neighbors(u)
        if parents[u] in neighbors:
            neighbors.remove(parents[u])
        for neighbor in neighbors:
            if neighbor > u:
                cell = (u,neighbor) # 角度は小さい方が入っている確率高い
                cell_ortind = oo[cell]
                if ort_counts[cell_ortind] == 0:
                    continue
                cell_sample = np.array(A_angles[start_index[cell_ortind] : start_index[cell_ortind+1]])
                threshold = np.pi - angles[u]
                # If angle of any cell_sample is SMALLER than this threshold, then u survives
                if np.any(cell_sample < threshold):
                    survives[u]=True
                    colors[u]='black'
                    break
            else:
                cell = (neighbor,u) # 角度は大きい方が入っている確率高い
                cell_ortind = oo[cell]
                if ort_counts[cell_ortind] == 0:
                    continue
                cell_sample = np.array(A_angles[start_index[cell_ortind] : start_index[cell_ortind+1]])
                threshold = np.pi - angles[u]
                # If angle of any cell_sample is LARGER than this threshold, then u survives
                if np.any(np.pi/2 - cell_sample < threshold):
                    survives[u]=True
                    colors[u]='black'
                    break
        colors[u] = 'black'
    else:
        neighbors = find_neighbors(u)
        if parents[u] in neighbors:
            neighbors.remove(parents[u])
        for neighbor in neighbors:
            if colors[neighbor] == 'white':
                parents[neighbor] = u
                angles[neighbor] = angles[u] + np.pi/2
                if neighbor > u:
                    cell = (u, neighbor)
                else:
                    cell = (neighbor, u)
                cell_ortind = oo[cell]
                if ort_counts[cell_ortind] > 0:
                    survives[u]=True
                elif ort_counts[oo[(neighbor, neighbor)]] > 0:
                    survives[u]=True
                parents, colors, angles, survives = _DFS_visit(neighbor, angles, colors, parents, survives, ort_counts, A_angles, start_index, oo)
                if survives[neighbor] == True:
                    survives[u] = True
        if not survives[u]:
            for neighbor in neighbors:
                if colors[neighbor]=='white':
                    neighbor_boundary_ind = oo[(neighbor,neighbor)]
                    if ort_counts[neighbor_boundary_ind] > 0:
                        survives[u]=True
                        break
        colors[u] = 'black'
    return parents, colors, angles, survives

if __name__ == "__main__":
    X = pd.read_csv("../data/sample_data_100_case3.csv")
    sort_ind = argsort_by_orthants(X)
    X = X.iloc[sort_ind]

    # prepare arguments
    sample_coord1 = X['x1'].values
    sample_coord2 = X['x2'].values
    sample_angle = X['angle'].values
    start_index = get_start_indices(X)
    cells = tuple_2dcells()
    orthants = tuple_orthants()

    #sample_bd_coord, sample_bd_lam, sample_origin_lam = geodesic_sample(sample_coord1, sample_coord2, sample_angle,start_index, cells)
    edge_indices = _link_convex_hull(sample_angle, start_index)
    print(edge_indices)
    #edge_indices, ext_coord, ext_lam, simple_indicator = twoDconvhull(sample_coord1, sample_coord2, sample_angle, start_index,
    #                    sample_bd_coord, sample_bd_lam)
