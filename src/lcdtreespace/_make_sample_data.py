from ._normal import normal_centered_2dim, normal_uncentered_2dim
from ._utils import *
import pandas as pd
import numpy as np
from importlib.resources import files
from ._onedim_dists import *
import os

def make_lc_data():

    # case 3
    all_cells = tuple_2dcells()
    sigma = 1
    for size in (50,100,200,300,500,1000):
        np.random.seed(3 * size)
        seed_size = np.random.randint(10000,size=10)
        for i in range(10):
            samples = normal_centered_2dim(all_cells, sigma).sample(size, seed=seed_size[i])
            samples.to_csv(files("lcdtreespace.data").joinpath(f"case3/testcase_{size}_{i}.csv"), index=False)

    # case 4
    all_cells = np.array([[0,1], [1,6], [6,8], [3,8], [3,4], [0,4]])
    sigma = 1
    for size in (50,100,200,300,500,1000):
        np.random.seed(4 * size)
        seed_size = np.random.randint(10000,size=10)
        for i in range(10):
            samples = normal_centered_2dim(all_cells, sigma).sample(size, seed=seed_size[i])
            samples.to_csv(files("lcdtreespace.data").joinpath(f"case4/testcase_{size}_{i}.csv"), index=False)


def make_clustering_data():
    size = 200
    base = 3571
    np.random.seed(base)
    seed_size = np.random.randint(10000, size = 3*10)
    for i in range(10):
        np.random.seed(seed_size[3*i])
        n1 = np.random.binomial(n=200, p=0.5)
        #samples, which = density_sample(seed = 10, pi = 0.5, size=size)
        samples1 = normal_uncentered_2dim([0,1], np.array([1,1]), 1).sample(200-n1, seed = seed_size[3*i+1])
        samples2 = normal_uncentered_2dim([2,3], np.array([1,1]), 0.5).sample(n1, seed=seed_size[3*i+2])
        samples = pd.concat((samples1, samples2)).reset_index(drop=True)
        which = [0 for _ in range(200-n1)] + [1 for _ in range(n1)]
        samples['which'] = which

        sort_ind = argsort_by_orthants(samples)
        samples = samples.iloc[sort_ind].reset_index(drop=True)
        samples.to_csv(files("lcdtreespace.data").joinpath(f"mixture/{size}_{i}.csv"), index=False)

        # x = add_angle(samples1.tolist())
        # x = pd.DataFrame(x, columns=['edge1', 'edge2', 'x1', 'x2', 'angle'])
        # samples2 = normal([2,3], np.array([1,1]), 0.5).sample(n1, seed=seed+1)
        # x2 = add_angle(samples2.tolist())
        # x2 = pd.DataFrame(x2, columns=['edge1', 'edge2', 'x1', 'x2', 'angle'])
        #
        # x = pd.concat([x,x2])
        # which = np.array([0 for i in range(100)] + [1 for i in range(100)]).astype(int)
        # x['y'] = np.zeros(200)
        # x['which'] = which
        # print(x)
        # x.to_csv(f"data/clustering/seed{seed}.csv", index=False)

def make_clustering_data3(size=200):
    base = 3571
    np.random.seed(base)
    seed_size = np.random.randint(10000, size = 4*10)
    for i in range(10):
        np.random.seed(seed_size[4*i])
        n1 = np.random.binomial(n=size, p= 3/10)
        n2 = np.random.binomial(n=size-n1, p = 3/7)
        #samples, which = density_sample(seed = 10, pi = 0.5, size=size)
        samples1 = normal_uncentered_2dim([0,1], np.array([4,4]), 2*np.sqrt(2)).sample(size-n1-n2, seed = seed_size[4*i+1])
        samples2 = normal_uncentered_2dim([2,3], np.array([1,1]), 0.25).sample(n1, seed=seed_size[4*i+2])
        samples3 = normal_uncentered_2dim([5,8], np.array([1,1]), 0.25).sample(n2, seed=seed_size[4*i+3])
        samples = pd.concat((samples1, samples2, samples3)).reset_index(drop=True)
        which = [0 for _ in range(size-n1-n2)] + [1 for _ in range(n1)] + [2 for _ in range(n2)]
        samples['which'] = which

        sort_ind = argsort_by_orthants(samples)
        samples = samples.iloc[sort_ind].reset_index(drop=True)
        samples.to_csv(files("lcdtreespace.data").joinpath(f"mixture/cluster3_{size}_{i}.csv"), index=False)

        # x = add_angle(samples1.tolist())
        # x = pd.DataFrame(x, columns=['edge1', 'edge2', 'x1', 'x2', 'angle'])
        # samples2 = normal([2,3], np.array([1,1]), 0.5).sample(n1, seed=seed+1)
        # x2 = add_angle(samples2.tolist())
        # x2 = pd.DataFrame(x2, columns=['edge1', 'edge2', 'x1', 'x2', 'angle'])
        #
        # x = pd.concat([x,x2])
        # which = np.array([0 for i in range(100)] + [1 for i in range(100)]).astype(int)
        # x['y'] = np.zeros(200)
        # x['which'] = which
        # print(x)
        # x.to_csv(f"data/clustering/seed{seed}.csv", index=False)


def make_1dim_data():
    # this renews all the data, so should be used carefully
    # case1
    mu = 1
    os.makedirs(files("lcdtreespace").joinpath("data",'case1'), exist_ok=True)
    for n in [100,200,300,500,1000]:
        for i in range(10):
            np.random.seed(n+i)
            x, labels = normal_1dim(mu=mu, sigma=1).sample(size=n)
            #d = normal_1dim().rvs(size=n, mu=mu, sigma=1)
            #labels = np.zeros(n)
            #for k in range(n):
            #    if d[k] <= 2*mu:
            #        continue
            #    else:
            #        labels[k] = np.random.choice([0,1,2])
            #x1 = np.sort(d[labels==0])
            #x2 = np.sort(d[labels==1] - 2*mu)
            #x3 = np.sort(d[labels==2] - 2*mu)
            #x = np.concatenate([x1,x2,x3])
            #labels = np.concatenate([ [0 for j in range(len(x1))], [1 for j in range(len(x2))], [2 for j in range(len(x3))] ])
            np.save(files("lcdtreespace").joinpath("data", 'case1').joinpath(f"testcase_{n}_{i}_X.npy"),x)
            np.save(files("lcdtreespace").joinpath("data",'case1').joinpath(f"testcase_{n}_{i}_ort.npy"),labels)

    os.makedirs(files("lcdtreespace").joinpath("data",'case2'), exist_ok=True)
    for n in [100,200,300,500,1000]:
        for i in range(10):
            np.random.seed(n+i)
            x, labels = exponential_1dim(mu=1, lam=1).sample(size=n)
            #d2 = exponential_1dim().rvs(size=n, mu=1, lam=1)
            #labels = np.zeros(n)
            #for k in range(n):
            #    if d2[k] <= mu:
            ##        d2[k] = mu-d2[k]
            #        continue
            #    else:
            #        labels[k] = np.random.choice([1,2])
            #x1 = np.sort(d2[labels==0])
            #x2 = np.sort(d2[labels==1] - mu)
            #x3 = np.sort(d2[labels==2] - mu)
            #x = np.concatenate([x1,x2,x3])
            #labels = np.concatenate([ [0 for j in range(len(x1))], [1 for j in range(len(x2))], [2 for j in range(len(x3))] ])
            np.save(files("lcdtreespace").joinpath("data", 'case2').joinpath(f"testcase_{n}_{i}_X.npy"),x)
            np.save(files("lcdtreespace").joinpath("data", 'case2').joinpath(f"testcase_{n}_{i}_ort.npy"),labels)

    # case 5
    mu=1; sigma=5
    os.makedirs(files("lcdtreespace").joinpath("data",'case5'), exist_ok= True)
    for n in [100,200,300,500,1000]:
        #os.makedirs(f'brown/sample{n}', exist_ok=True)
        np.random.seed(n+i)
        for i in range(10):
            x, labels = normal_bend_1dim(mu=mu,sigma=sigma,k=3).sample(size=n)
            #d = normal_bend_1dim().rvs(size=n, mu=mu, sigma=sigma,k=3)
            #labels = np.zeros(n)
            #for j in range(n):
            #    if d[j] > 0:
            #        continue
            #    else:
            #        labels[j] = np.random.choice([1,2])
            np.save(files("lcdtreespace").joinpath("data",'case5').joinpath(f"testcase_{n}_{i}_X.npy"),x)
            np.save(files("lcdtreespace").joinpath("data",'case5').joinpath(f"testcase_{n}_{i}_ort.npy"),labels)

    # case 6
    os.makedirs(files("lcdtreespace").joinpath("data", 'case6'), exist_ok= True)
    for n in [100,200,300,500,1000]:
        for i in range(10):
            np.random.seed(n+i)
            x, labels = coalescent_1dim(T=1).sample(size=n)
            #d = coalescent_1dim().rvs(size=n, T=1)
            #labels = np.zeros(n)
            #for j in range(n):
            #    if d[j] > 0:
            #        continue
            #    else:
            #        labels[j] = np.random.choice([1,2])
            np.save(files("lcdtreespace").joinpath("data",'case6').joinpath(f"testcase_{n}_{i}_X.npy"),x)
            np.save(files("lcdtreespace").joinpath("data",'case6').joinpath(f"testcase_{n}_{i}_ort.npy"),labels)
#make_lc_data()
#make_clustering_data3()
