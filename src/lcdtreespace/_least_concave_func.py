import numpy as np
from numpy import argmax, where, hstack, vstack, unique, repeat, tile, arange, sqrt
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from scipy import sparse
from scipy.sparse import csr_matrix
from ._link_convhull import _link_convex_hull
from ._utils import *
from ._geodesic_sample  import geodesic_sample
from ._twoDconvhull import _twoDconvhull
import time

def _determine_max_o(y, sample_coord1, sample_coord2, start_index, bd_coord, bd_y, bd_lam, max_o,
                    nei3cells, embed, embed_indices, embed_n):
    beq = np.array([1,0,0])
    max_o = -np.inf
    for i in range(60):
        nei3cell = nei3cells[i]
        embedding = embed[i]
        if embedding is None:
            continue
        embedding_y = y[embed_indices[i]]
        n_bd1, n_bd2, n_bd3, n_bd4 = len(bd_coord[nei3cell[0]]) , len(bd_coord[nei3cell[1]]) , len(bd_coord[nei3cell[2]]) , len(bd_coord[nei3cell[3]])
        n_bd = n_bd1 + n_bd2 + n_bd3 + n_bd4
        n_sample = len(embed_indices[i])
        n_all = n_bd + n_sample
        if n_all < 3:
            continue
        if (embed_n[i][0] == 0 and n_bd1 == 0) or (embed_n[i][2]==0 and n_bd4 == 0):
            continue
        A = np.zeros((3, n_all))
        A[0].fill(1)
        A[1:,:n_sample] = embedding.T
        e = n_sample + n_bd1; A[2, n_sample:e] = -bd_coord[nei3cell[0]]
        s = e + n_bd2;        A[1, e:s] = bd_coord[nei3cell[1]]
        e = s + n_bd3;        A[2, s:e] = bd_coord[nei3cell[2]]
        A[1, e:e+n_bd4] = -bd_coord[nei3cell[3]]
        c = np.zeros(n_all)
        c[:n_sample] = -embedding_y
        c[n_sample:] = -hstack((bd_y[nei3cell[0]],bd_y[nei3cell[1]],bd_y[nei3cell[2]],bd_y[nei3cell[3]] ))
        res = linprog(c, A_eq = A, b_eq = beq)
        if res.status==0:
            max_o_cand = -res.fun
            if max_o_cand > max_o:
                LAMS_p = np.zeros(( n_all  , len(y)))
                LAMS_p[[i for i in range(n_sample)], embed_indices[i]] = 1
                #LAMS_p = csr_matrix(LAMS_p)
                s = n_sample + n_bd1; LAMS_p[n_sample:s] = bd_lam[nei3cell[0]].toarray() #might be slow
                t = s + n_bd2; LAMS_p[s:t] = bd_lam[nei3cell[1]].toarray()#might be slow
                s = t + n_bd3; LAMS_p[t:s] = bd_lam[nei3cell[2]].toarray()#might be slow
                LAMS_p[s:n_all] = bd_lam[nei3cell[3]].toarray()
                max_o = max_o_cand
                origin_lam = csr_matrix(res.x @ LAMS_p)

    return max_o, origin_lam

def _num_new_bds(start_index, cells):
    # Returns matrix M of size 10 * 10
    # for vector nn = (#new_bd[0], ..., #new_bd[9]),
    # M @ nn returns how many new bds are going to be added
    M = np.zeros((10,10), dtype=np.int64)
    for i in range(15):
        num_i = start_index[i+1] - start_index[i]
        u, v = cells[i]
        u_neighbor = find_neighbors(u)
        v_neighbor = find_neighbors(v)
        for item in u_neighbor:
            if item == v:
                continue
            M[u, item] = M[u,item] + num_i # 1こitemが増えたら，num_iこuのbd pointが新しくできる．
        for item in v_neighbor:
            if item == u:
                continue
            M[v, item] = M[v,item] + num_i # 1こitemが増えたら，num_iこvのbd pointが新しくできる．
    return M

def _generate_new_bd(y, sample_coord1, sample_coord2, start_index, cells,
                    now_bd_coord, now_bd_y, now_bd_lam):
    M = _num_new_bds(start_index, cells)
    n = len(sample_coord1)
    n_now_bd = np.array([len(now_bd_coord[i]) for i in range(10)], dtype=np.int64)
    n_new_bd = M @ n_now_bd
    new_bd_coord = [np.zeros(n_new_bd[i]) for i in range(10)]
    new_bd_lam = [np.zeros((n_new_bd[i], n)) for i in range(10)]
    new_bd_y = [np.zeros(n_new_bd[i]) for i in range(10)]
    new_bd_counters = [0 for i in range(10)]
    dict_laminv = {}

    max_o_cand = -np.inf
    max_o_cands_lam = csr_matrix(np.zeros((0,n)))
    for i in range(15):
        num_i = start_index[i+1] - start_index[i]
        if num_i == 0:
            continue
        u, v = cells[i]
        u_neighbor = find_neighbors(u)
        v_neighbor = find_neighbors(v)
        cell_sample_coord1 = sample_coord1[start_index[i]:start_index[i+1]]
        cell_sample_coord2 = sample_coord2[start_index[i]:start_index[i+1]]
        cell_y = y[start_index[i]:start_index[i+1]]
        cell_lam = np.zeros((num_i, n))
        cell_lam[[_ for _ in range(num_i)], arange(start_index[i], start_index[i+1])] = 1
        for item in u_neighbor:
            if item == v:
                continue
            x_item = now_bd_coord[item]
            lam_item = now_bd_lam[item].toarray() #might be slow
            y_item = now_bd_y[item]
            n_item = len(x_item)
            if n_item == 0:
                continue
            nxt_counter = new_bd_counters[u]+ n_item * num_i

            lam1 =  tile(x_item, num_i) / (tile(x_item, num_i) + repeat(cell_sample_coord2, n_item))

            # appending newly created boundary points
            new_bd_coord[u][new_bd_counters[u]:nxt_counter] = lam1 * repeat(cell_sample_coord1, n_item) # append to new_bd_coord
            new_bd_lam[u][new_bd_counters[u]:nxt_counter] = lam1.reshape(-1,1) * repeat(cell_lam, n_item, axis=0) + (1-lam1).reshape(-1,1) * tile(lam_item, (num_i, 1))
            new_bd_y[u][new_bd_counters[u]:nxt_counter] = lam1 * repeat(cell_y, n_item) + (1-lam1) * tile(y_item, num_i)

            new_bd_counters[u] = nxt_counter

        for item in v_neighbor:
            if item == u:
                continue
            x_item = now_bd_coord[item]
            lam_item = now_bd_lam[item].toarray() #might be slow
            y_item = now_bd_y[item]
            n_item = len(x_item)
            if n_item == 0:
                continue
            nxt_counter = new_bd_counters[v]+ n_item * num_i

            lam1 =  tile(x_item, num_i) / (tile(x_item, num_i) + repeat(cell_sample_coord1, n_item))

            # appending newly created boundary points
            new_bd_coord[v][new_bd_counters[v]:nxt_counter] = lam1 * repeat(cell_sample_coord2, n_item) # append to new_bd_coord
            new_bd_lam[v][new_bd_counters[v]:nxt_counter] = lam1.reshape(-1,1) * repeat(cell_lam, n_item, axis=0) + (1-lam1).reshape(-1,1) * tile(lam_item, (num_i, 1))
            new_bd_y[v][new_bd_counters[v]:nxt_counter] = lam1 * repeat(cell_y, n_item) + (1-lam1) * tile(y_item, num_i)

            new_bd_counters[v] = nxt_counter

        uv_neighbor = u_neighbor
        uv_neighbor.extend(v_neighbor)
        nonneighbors = set([ i for i in range(10)]) - set(uv_neighbor)
        for item in nonneighbors:
            # cone paths
            x_item = now_bd_coord[item]
            lam_item = now_bd_lam[item].toarray() #might be slow
            y_item = now_bd_y[item]
            n_item = len(x_item)
            if n_item == 0:
                continue
            len_sample = sqrt(cell_sample_coord1**2 + cell_sample_coord2**2)
            lam1 =  tile(x_item, num_i) / (tile(x_item, num_i) + repeat(len_sample, n_item))
            max_o_cands = lam1 * repeat(cell_y, n_item) + (1-lam1) * tile(y_item, num_i)

            arg_max_o_cands = argmax(max_o_cands)
            max_o_cand_item = max_o_cands[arg_max_o_cands]
            if max_o_cand_item > max_o_cand:
                max_o_cand = max_o_cand_item
                sample_index = arg_max_o_cands // n_item
                item_index = arg_max_o_cands % n_item
                max_o_cands_lam = lam1[arg_max_o_cands] * cell_lam[sample_index] + (1-lam1[arg_max_o_cands]) * lam_item[item_index]

    # might not be necessary
    for i in range(10):
        new_bd_lam[i] = csr_matrix(new_bd_lam[i])
    max_o_cands_lam = csr_matrix(max_o_cands_lam)
    return new_bd_coord, new_bd_lam, new_bd_y, max_o_cand, max_o_cands_lam




# Hasn't tested for the case where one of simple_indicator is zero.
def _least_concave_func(y, sample_coord1, sample_coord2, sample_angle, start_index,
                    _sample_bd_coord, _sample_bd_lam, sample_origin_lam,
                    edge_indices, ext_coord, ext_lam, simple_indicator,
                    embed, embed_indices, embed_n,
                    n, cells, nei3cells, M, cp_pairs, max_iter = 100):
    n_edges = len(edge_indices)


    sample_bd_coord = [np.array([]) for _ in range(10)]
    sample_bd_lam = [csr_matrix(np.empty(shape=(0,n))) for _ in range(10)]
    sample_bd_y = [np.array([]) for _ in range(10)]

    ## Initialize origin points
    origin_candidates = sample_origin_lam @ y
    arg_max_o = argmax(origin_candidates)
    max_o = origin_candidates[arg_max_o]
    max_o_lam = sample_origin_lam[arg_max_o]

    ## Initialize bd points
    for ind in edge_indices:
        ind_x = _sample_bd_coord[ind]
        ind_lam = _sample_bd_lam[ind]
        ind_y = ind_lam @ y
        ext_x = ext_coord[ind]
        ext_y = (ext_lam[ind] @ y)[0]
        eps_tmp = 1e-5
        ## First exclude any points that are below the segment between the origin and extreme value
        slope = (ext_y - max_o)/ext_x
        y_on_slope = slope * ind_x + max_o

        remain_indices = where(y_on_slope - eps_tmp < ind_y)[0]


        ## Construct 2d convex hull and extract vertices
        if len(remain_indices)==0:
            if simple_indicator[ind]==1:
                # the extreme value is already included in sample_bd_coord
                raise Exception("ERROR.")
            # Just save extreme points only
            sample_bd_coord[ind] = np.array([ext_x])
            sample_bd_lam[ind] = ext_lam[ind].reshape(1,-1)
            sample_bd_y[ind] = ext_y
        else:
            points = hstack((ind_x[remain_indices].reshape(-1,1), ind_y[remain_indices].reshape(-1,1)))
            if simple_indicator[ind]==1:
                points = vstack((points, [0,max_o])) # this might be slow
                if points.shape[0]==2:
                    sample_bd_coord[ind] = ind_x[remain_indices]
                    sample_bd_lam[ind] =ind_lam[remain_indices]
                    sample_bd_y[ind] = ind_lam[remain_indices] @ y
                else:
                    try:
                        hull = ConvexHull(points)
                    except:
                        hull = ConvexHull(points, qhull_options = "QJ")
                    vertices = unique(hull.simplices[hull.equations[:,1]>0])
                    vertices = vertices[vertices < len(remain_indices)]
                    sample_bd_coord[ind] = ind_x[remain_indices][vertices]
                    sample_bd_lam[ind] =ind_lam[remain_indices][vertices]
                    sample_bd_y[ind] = ind_lam[remain_indices][vertices] @ y
            else:
                points = vstack((points, [0,max_o], [ext_x, ext_y]))
                try:
                    hull = ConvexHull(points)
                except:
                    hull = ConvexHull(points, qhull_options = "QJ")
                vertices = unique(hull.simplices[hull.equations[:,1]>0])
                vertices = vertices[vertices < len(remain_indices)]
                sample_bd_coord[ind] = np.append(ind_x[remain_indices][vertices], ext_x)
                temp = ind_lam[remain_indices][vertices];
                temp.resize(temp.shape[0]+1, temp.shape[1]);
                temp[len(vertices)] = ext_lam[ind];
                sample_bd_lam[ind] = temp
                sample_bd_y[ind] = np.append(ind_lam[remain_indices][vertices] @ y, ext_y)
    now_bd_coord = sample_bd_coord
    now_bd_lam = sample_bd_lam
    now_bd_y = sample_bd_y

    old_bd_coord = [np.array([]) for i in range(10)]
    old_bd_lam = [csr_matrix(np.empty(shape=(0,n))) for i in range(10)]
    old_bd_y = [np.array([]) for i in range(10)]

    # ITERATION
    for iter_n in range(max_iter):
        ## determine max o
        max_o_cand, max_o_lam_cand = _determine_max_o(y, sample_coord1, sample_coord2, start_index, sample_bd_coord, sample_bd_y, sample_bd_lam, max_o,
                            nei3cells, embed, embed_indices, embed_n)
        if max_o_cand > max_o:
            max_o = max_o_cand
            max_o_lam = max_o_lam_cand
        ## take geodesics between sample and now bd
        ### we don't consider the case when sample includes unresolved trees. Thus, we don't consider cone paths between sample and now_bds

        #start = time.perf_counter()
        new_bd_coord, new_bd_lam, new_bd_y, max_o_cand, max_o_cands_lam = _generate_new_bd(y, sample_coord1, sample_coord2, start_index, cells,
                            now_bd_coord, now_bd_y, now_bd_lam)
        #end=time.perf_counter()
        #print(end-start)

        if max_o_cand > max_o:
            max_o = max_o_cand
            max_o_lam = max_o_cands_lam
            #print("\nMAXOO\n\n")
            #print(max_o, max_o_lam)


        ## take geodesics between now_bds and (now_bds, old_bds) to find candidates for max_o
        for l in range(len(cp_pairs[0])):
            u = cp_pairs[0][l]
            v = cp_pairs[1][l]
            n_now_u = len(now_bd_coord[u])
            n_now_v = len(now_bd_coord[v])
            n_old_u = len(old_bd_coord[u])
            n_old_v = len(old_bd_coord[v])

            s = n_now_u * n_now_v; t = n_old_u * n_now_v; p = n_now_u * n_old_v

            if s+t+p==0:
                continue

            temp_ovals = np.zeros(s+t+p)
            #print(f"s{s}, t{t}, p{p}")
            # new u vs new v
            if s>0:
                lam_s = tile(now_bd_coord[v], n_now_u)/(repeat(now_bd_coord[u], n_now_v) + tile(now_bd_coord[v], n_now_u))
                temp_ovals[:s] = lam_s * repeat(now_bd_y[u], n_now_v) + (1-lam_s) * tile(now_bd_y[v], n_now_u)

            # now u vs new v
            if t>0:
                lam_t = tile(now_bd_coord[v], n_old_u)/(repeat(old_bd_coord[u], n_now_v) + tile(now_bd_coord[v], n_old_u))
                temp_ovals[s:s+t] = lam_t * repeat(old_bd_y[u], n_now_v) + (1-lam_t) * tile(now_bd_y[v], n_old_u)

            # new u vs now v
            if p>0:
                lam_p = tile(old_bd_coord[v], n_now_u)/(repeat(now_bd_coord[u], n_old_v) + tile(old_bd_coord[v], n_now_u))
                temp_ovals[s+t:] = lam_p * repeat(now_bd_y[u], n_old_v) + (1-lam_p) * tile(old_bd_y[v], n_now_u)

            argmax_temp_ovals = argmax(temp_ovals)
            temp_max = temp_ovals[argmax_temp_ovals]
            if temp_max > max_o:
                max_o = temp_max
                if argmax_temp_ovals < s:
                    # generated by now vs now
                    u_ind = argmax_temp_ovals // n_now_v; v_ind = argmax_temp_ovals % n_now_v
                    max_o_lam = csr_matrix(lam_s[argmax_temp_ovals] * now_bd_lam[u][u_ind].toarray() + (1-lam_s[argmax_temp_ovals]) * now_bd_lam[v][v_ind].toarray())
                elif argmax_temp_ovals < s+t:
                    # generated by old vs now
                    index = argmax_temp_ovals - s
                    u_ind = index // n_now_v; v_ind = index % n_now_v
                    max_o_lam = csr_matrix(lam_t[index] * old_bd_lam[u][u_ind].toarray() + (1-lam_t[index]) * now_bd_lam[v][v_ind].toarray())
                else:
                    # generated by now vs old
                    index = argmax_temp_ovals - s - t
                    u_ind = index // n_old_v; v_ind = index % n_old_v
                    max_o_lam = csr_matrix(lam_p[index] * now_bd_lam[u][u_ind].toarray() + (1-lam_p[index]) * old_bd_lam[v][v_ind].toarray())


        ## take convex hull at each edge
        renewed = 0
        for ind in edge_indices:
            n_old = old_bd_coord[ind].shape[0]; n_now = now_bd_coord[ind].shape[0]; n_new = new_bd_coord[ind].shape[0]
            n_oldnow = n_old + n_now
            old_stacked = hstack((old_bd_coord[ind].reshape(-1,1), old_bd_y[ind].reshape(-1,1)))
            now_stacked = hstack((now_bd_coord[ind].reshape(-1,1), now_bd_y[ind].reshape(-1,1)))
            old_and_now = vstack((old_stacked,now_stacked))
            old_and_now_bd_lam = csr_matrix(vstack((old_bd_lam[ind].toarray(), now_bd_lam[ind].toarray())))
            if len(new_bd_coord[ind]) == 0:
                before_points = vstack((old_and_now, [0,max_o]))
                if before_points.shape[0]>2:
                    try:
                        hull = ConvexHull(before_points, incremental = True)
                    except:
                        hull = ConvexHull(before_points, qhull_options = "QJ", incremental = True)
                    vertices = unique(hull.simplices[hull.equations[:,1]>0])
                    old_now_remains = vertices[vertices < n_oldnow]
                    old_bd_coord[ind] = old_and_now[old_now_remains,0]
                    old_bd_y[ind] = old_and_now[old_now_remains,1]
                    old_bd_lam[ind] = old_and_now_bd_lam[old_now_remains]
                else:
                    old_bd_coord[ind] = old_and_now[:,0]
                    old_bd_y[ind] = old_and_now[:,1]
                    old_bd_lam[ind] = old_and_now_bd_lam

                # renewing nows
                now_bd_coord[ind] = np.array([])
                now_bd_y[ind] = np.array([])
                now_bd_lam[ind] = csr_matrix(np.empty(shape=(0,n)))
            else:
                before_points = vstack((old_and_now, [0,max_o]))
                if before_points.shape[0]>2:
                    try:
                        hull = ConvexHull(before_points, incremental = True)#, qhull_options="QJ")
                    except:
                        hull = ConvexHull(before_points, qhull_options = "QJ", incremental = True)
                    hull.add_points(hstack((new_bd_coord[ind].reshape(-1,1), new_bd_y[ind].reshape(-1,1))))
                else:
                    #hull = ConvexHull(before_points, incremental = True, qhull_options="QJ")
                    add = hstack((new_bd_coord[ind].reshape(-1,1), new_bd_y[ind].reshape(-1,1)))
                    try:
                        hull = ConvexHull(vstack((before_points, add)))#, qhull_options="QJ")
                    except:
                        hull = ConvexHull(vstack((before_points, add)), qhull_options = "QJ")
                vertices = unique(hull.simplices[hull.equations[:,1]>0])
                new_remains =vertices[vertices > n_old + n_now] - n_oldnow - 1
                old_now_remains = vertices[vertices < n_oldnow]

                # renewing olds
                old_bd_coord[ind] = old_and_now[old_now_remains][:,0]
                old_bd_y[ind] = old_and_now[old_now_remains][:,1]
                old_bd_lam[ind] = old_and_now_bd_lam[old_now_remains]
                #if iter_n > 5:
                #    print(new_remains)
                #    print(old_now_remains)
                #    input()
                if len(new_remains) > 0:
                    renewed = 1
                    # renewing nows
                    now_bd_coord[ind] = new_bd_coord[ind][new_remains]
                    now_bd_y[ind] = new_bd_y[ind][new_remains]
                    now_bd_lam[ind] = new_bd_lam[ind][new_remains]
                else:
                    now_bd_coord[ind] = np.array([])
                    now_bd_y[ind] = np.array([])
                    now_bd_lam[ind] = csr_matrix(np.empty(shape=(0,n)))

        if renewed == 0:
            break

    support_list = [0 for i in range(15)]
    orthant_coords = [np.zeros((0,3)) for i in range(15)]
    hull_list = [None for i in range(15)]
    for i in range(15):
        cell = cells[i]
        u = cell[0]; v = cell[1]
        cell_sample_n = start_index[i+1] - start_index[i]
        n_bd_u = len(old_bd_coord[u]); n_bd_v = len(old_bd_coord[v])
        n_all = cell_sample_n + n_bd_u + n_bd_v

        if cell_sample_n == 0 and (n_bd_u == 0 or n_bd_v==0):
            # This case, the orthant is not included in the convex hull
            support_list[i] = None
            continue
        if n_all == 1:
            # meaning only 1 cell_sample is there.
            raise Exception("Error: convex hull does not satisfy sufficient condition.")
        # If not included in the previous cases, then #sample points are larger than or equal to 2.
        x1_stack = hstack((sample_coord1[start_index[i] : start_index[i+1]], old_bd_coord[u], np.zeros(n_bd_v), [0]))
        x2_stack = hstack((sample_coord2[start_index[i] : start_index[i+1]], np.zeros(n_bd_u), old_bd_coord[v], [0]))
        y_stack = hstack((y[start_index[i]:start_index[i+1]], old_bd_y[u], old_bd_y[v], [max_o]))
        orthant_coords[i] = hstack((x1_stack.reshape(-1,1), x2_stack.reshape(-1,1), y_stack.reshape(-1,1)))
        if n_all == 2:
            support_list[i] = np.array([[0,1,2]])
        else:
            try:
                hull = ConvexHull(orthant_coords[i])
            except:
                hull = ConvexHull(orthant_coords[i], qhull_options = "QJ")
            hull_list[i] = hull
            support_list[i] = hull.simplices[hull.equations[:,2]>0]

    return old_bd_coord, old_bd_lam, old_bd_y, max_o, max_o_lam, support_list, orthant_coords, hull_list












if __name__ == "__main__":
    X = pd.read_csv("../data/sample_data_1000_case4.csv")
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
    #print(simple_indicator)
    n = len(sample_coord1)


    y = X['y'].values

    nei3cells = tuple_nei3cells()
    #bb_lenmat = b_to_b_lenmat()
    #print(bb_lenmat[where(bb_lenmat == 2)])
    cp_pairs = cone_path_pairs()
    #print(b_to_b_lenmat()[cone_path_pairs()])
    old_bd_coord, old_bd_lam, old_bd_y, max_o, max_o_lam, support_list, orthant_coords, hull_list = _least_concave_func(y, sample_coord1, sample_coord2, sample_angle, start_index,
                        sample_bd_coord, sample_bd_lam, sample_origin_lam,
                        edge_indices, ext_coord, ext_lam, simple_indicator,
                        embed, embed_indices, embed_n,
                        n, cells, nei3cells, M, cp_pairs)

    print(hull_list)
    print(orthant_coords[0])
    input()
    print(support_list)
