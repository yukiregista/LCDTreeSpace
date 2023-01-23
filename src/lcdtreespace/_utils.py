import numpy as np
import pandas as pd

def find_neighbors(u):
    # u: int, index of edge, should be one of 0-9
    # returns the list of indices of neighboring edges
    neighbors = [None]*3
    if (u<5):
        neighbors[0] = (u+1) % 5 + (u//5)*5;
        neighbors[1] = (u+4) % 5 + (u//5)*5;
        neighbors[2] = (u+5) % 10;
    else:
        neighbors[0] = 5 + (u+2) % 5;
        neighbors[1] = 5 + (u-2) % 5;
        neighbors[2] = (u+5) % 10;

    return neighbors

def tuple_2dcells():
    # tuple of 2d cells represented by list of two edges
    cells = []
    for i in range(10):
        for item in find_neighbors(i):
            if i < item:
                cells.append((i,item))
    return tuple(cells)

def list_2dcells():
    # list of 2d cells represented by tuple of two edges
    cells = []
    for i in range(10):
        for item in find_neighbors(i):
            if i < item:
                cells.append((i,item))
    return cells

def tuple_orthants():
    # tuple of all orthants, including 1dim boundary and origin
    # each 1 dim boundary is represented by [i,i] where i is the edge index
    # the origin point is represented by [-1,-1]
    orthants = list_2dcells()
    for i in range(10):
        orthants.append((i,i))
    orthants.append((-1,-1))
    return tuple(orthants)

def orthant_to_ortind():
    # returns a dict
    #   key: orthant tuple
    #   value: orthant index
    orthants = tuple_orthants()
    oo = {orthants[i]:i for i in range(len(orthants))}
    return oo

def argsort_by_orthants(X):
    # X: dataframe, having columns named "edge1" and "edge2"
    # returns argsorted index by orthants
    orts =tuple(zip(X['edge1'].values, X['edge2'].values))
    oo = orthant_to_ortind()
    ind = [oo[item] for item in orts]
    return np.argsort(ind, kind='stable')

def get_start_indices(X):
    # X: dataframe, having columns named "edge1" and "edge2",
    #    ALREADY SORTED BY ORTHANTS
    # returns list of start_indices
    n = len(X)
    start_indices = np.empty(27, dtype=np.int64)
    start_indices.fill(-1)
    orts =tuple(zip(X['edge1'].values, X['edge2'].values))
    oo = orthant_to_ortind()
    ind = [oo[item] for item in orts]
    unique, index = np.unique(ind, return_index=True)
    start_indices[unique] = index
    fill_num = n
    for i in range(26, -1, -1):
        if start_indices[i]==-1:
            start_indices[i] = fill_num
        else:
            fill_num = start_indices[i]
    return start_indices

def b_to_b_lenmat():
    lenmat = np.zeros((10,10), dtype=np.int8)
    lenmat.fill(2)
    np.fill_diagonal(lenmat,0)
    for i in range(10):
        for item in find_neighbors(i):
            lenmat[i, item] = 1
    return lenmat

def tuple_nei3cells():
    nei3cells = []
    cells = tuple_2dcells()
    for i in range(15):
        u,v = cells[i]
        u_neighbor = find_neighbors(u)
        v_neighbor = find_neighbors(v)

        for s in u_neighbor:
            if s==v: continue
            for t in v_neighbor:
                if t==u: continue
                nei = (s, u, v, t)
                nei3cells.append(nei)
    return tuple(nei3cells)

def embed_sample_points_to_nei3cells(sample_coord1, sample_coord2, start_index):
    # Embed points
    cells = tuple_2dcells()
    cell_to_cell_index = dict(zip(cells, [i for i in range(15)]))
    nei3cells = tuple_nei3cells()
    embeddings = [None]*60
    embedding_sample_indices = [None]*60
    embedding_n_each_cell = [None]*60
    #print(cell_to_cell_index)
    for i in range(60):
        nei3cell = nei3cells[i]
        first_reversed = int(nei3cell[0] > nei3cell[1]) # 1 indicates the order is reversed
        third_reversed = int(nei3cell[2] > nei3cell[3]) # 1 indicates the order is reversed
        first_cell = (nei3cell[first_reversed], nei3cell[1-first_reversed])
        second_cell = (nei3cell[1], nei3cell[2])
        third_cell = (nei3cell[third_reversed+2], nei3cell[3-third_reversed])

        first_cell_ind = cell_to_cell_index[first_cell]
        second_cell_ind = cell_to_cell_index[second_cell]
        third_cell_ind = cell_to_cell_index[third_cell]

        n_first_cell = start_index[first_cell_ind+1] - start_index[first_cell_ind]
        n_second_cell = start_index[second_cell_ind+1] - start_index[second_cell_ind]
        n_third_cell = start_index[third_cell_ind+1] - start_index[third_cell_ind]

        embedding_n_each_cell[i] = (n_first_cell, n_second_cell, n_third_cell)

        n_all = n_first_cell + n_second_cell + n_third_cell
        n_12 = n_first_cell + n_second_cell

        if n_all == 0:
            continue

        embedding = np.zeros((n_all, 2))

        if first_reversed:
            embedding[:n_first_cell, 0] = sample_coord1[start_index[first_cell_ind]:start_index[first_cell_ind+1]]
            embedding[:n_first_cell, 1] = -sample_coord2[start_index[first_cell_ind]:start_index[first_cell_ind+1]]
        else:
            embedding[:n_first_cell, 0] = sample_coord2[start_index[first_cell_ind]:start_index[first_cell_ind+1]]
            embedding[:n_first_cell, 1] = -sample_coord1[start_index[first_cell_ind]:start_index[first_cell_ind+1]]

        embedding[n_first_cell:n_12, 0] = sample_coord1[start_index[second_cell_ind]:start_index[second_cell_ind+1]]
        embedding[n_first_cell:n_12, 1] = sample_coord2[start_index[second_cell_ind]:start_index[second_cell_ind+1]]

        if third_reversed:
            embedding[n_12:n_12 + n_third_cell, 0] = -sample_coord1[start_index[third_cell_ind]:start_index[third_cell_ind+1]]
            embedding[n_12:n_12 + n_third_cell, 1] = sample_coord2[start_index[third_cell_ind]:start_index[third_cell_ind+1]]
        else:
            embedding[n_12:n_12 + n_third_cell, 0] = -sample_coord2[start_index[third_cell_ind]:start_index[third_cell_ind+1]]
            embedding[n_12:n_12 + n_third_cell, 1] = sample_coord1[start_index[third_cell_ind]:start_index[third_cell_ind+1]]

        embeddings[i] = embedding
        first_arange = np.arange(start_index[first_cell_ind], start_index[first_cell_ind + 1])
        second_arange = np.arange(start_index[second_cell_ind], start_index[second_cell_ind + 1])
        third_arange = np.arange(start_index[third_cell_ind], start_index[third_cell_ind + 1])
        embedding_sample_indices[i] = np.hstack((first_arange,second_arange,third_arange))
    return embeddings, embedding_sample_indices, embedding_n_each_cell

def cone_path_pairs():
    bb_lenmat = b_to_b_lenmat()
    temp = np.where(bb_lenmat==2)
    tf = temp[0] < temp[1]
    return (temp[0][tf], temp[1][tf])



if __name__ == "__main__":
    #print(find_neighbors(0))
    #print(tuple_2dcells())
    #print(tuple_orthants())
    #print(orthant_to_ortind())
    X = pd.read_csv("../data/sample_data_100_case4.csv")
    sort_ind = argsort_by_orthants(X)
    X = X.iloc[sort_ind]
    sample_coord1 = X['x1'].values
    sample_coord2 = X['x2'].values
    start_index = get_start_indices(X)
    embeds, embeds_indices, embedding_n = embed_sample_points_to_nei3cells(sample_coord1, sample_coord2, start_index)
    #print(get_start_indices(X))
    #print(b_to_b_lenmat())
