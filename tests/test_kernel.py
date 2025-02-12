from lcdtreespace import *
import lcdtreespace as lcd
import numpy as np
import pytest
from scipy.integrate import quad, dblquad


def test_kernel_density_1dim():
    size=100
    x,ort = normal_1dim(1,1).sample(size)

    # adaptive kernel density
    dens = kernel_density_estimate_1dim(x, ort, 3)
    integ0, err0 = quad(dens.pdf, 0, np.inf, args = (0), epsabs=1e-10, limit=100)
    integ1, err1 = quad(dens.pdf, 0, np.inf, args = (1), epsabs=1e-10, limit=100)
    integ2, err2 = quad(dens.pdf, 0, np.inf, args = (2), epsabs=1e-10, limit=100)
    print(integ0, integ1, integ2, integ0 + integ1 + integ2)
    print("err:", err1 + err2 + err0)
    assert np.abs(integ0 + integ1 + integ2-1) < 1e-8

    # fixed bandwidth
    dens = kernel_density_estimate_1dim(x, ort, 3, bandwidth=1)
    integ0, err0 = quad(dens.pdf, 0, np.inf, args = (0), epsabs=1e-10, limit=100)
    integ1, err1 = quad(dens.pdf, 0, np.inf, args = (1), epsabs=1e-10, limit=100)
    integ2, err2 = quad(dens.pdf, 0, np.inf, args = (2), epsabs=1e-10, limit=100)
    print(integ0, integ1, integ2, integ0 + integ1 + integ2)
    print("err:", err1 + err2 + err0)
    assert np.abs(integ0 + integ1 + integ2-1) < 1e-8

def test_kernel_density_2dim():
    size = 100
    X = normal_uncentered_2dim([0,1], np.array([0.5,0.5]), 1).sample(size)

    # adaptive kernel density
    dens = kernel_density_estimate_2dim(X, nn_prop=0.3)
    allcells = lcd._utils.list_2dcells()
    integ = 0
    for cell in allcells:
        integ0, err0 = dblquad(dens.pdf, 0, np.inf, 0, np.inf, args=(cell[0],cell[1]), epsabs=1e-5)
        integ += integ0
        print(integ, flush=True)
    assert np.abs(integ - 1) < 1e-3

    # fixed kernel density
    dens = kernel_density_estimate_2dim(X, bandwidth=1)
    allcells = lcd._utils.list_2dcells()
    integ = 0
    for cell in allcells:
        integ0, err0 = dblquad(dens.pdf, 0, np.inf, 0, np.inf, args=(cell[0],cell[1]), epsabs=1e-5)
        integ += integ0
        print(integ, flush=True)
    assert np.abs(integ - 1) < 1e-3
    

def test_bias_1dim():
    size = 1000
    true_dens = normal_1dim(1,1)
    true_origin_val = true_dens.pdf(0, 0)
    
    origin_pdf_vals = list()
    origin_pdf_vals_bias = list()
    
    for i in range(10):
        print(i, flush=True)
        x,ort = true_dens.sample(size)
        # bias-free ver
        dens = kernel_density_estimate_1dim(x, ort, 3, bandwidth="scott-like", bias_free=True)
        origin_pdf_vals.append(dens.pdf(0,0))
        
        # bias ver
        dens2 = kernel_density_estimate_1dim(x, ort, 3, bandwidth="scott-like", bias_free=False)
        origin_pdf_vals_bias.append(dens2.pdf(0,0))


    print(f"true value: {true_origin_val}")
    print(f"bias-free kde estimate mean: {np.mean(origin_pdf_vals)}")
    print(f"biased kde estimate mean: {np.mean(origin_pdf_vals_bias)}")
    print(f"theoretical biased kde value: {true_origin_val * 3 * np.log(3/2)}")
    
def test_bias_2dim():
    size = 10000
    true_dens = normal_uncentered_2dim([0,1], np.array([0.5,0.5]), 1)
    X = true_dens.sample(size)

    # bias_free_ver
    dens = kernel_density_estimate_2dim(X, bandwidth="scott-like", bias_free=True)
    # bias ver
    dens2 = kernel_density_estimate_2dim(X, bandwidth="scott-like", bias_free=False)
    
    print(f"true value: {true_dens.pdf(0,0,0,1)}")
    print(f"bias-free kde estimate mean: {dens.pdf(0,0,0,1)}")
    print(f"biased kde estimate mean: {dens2.pdf(0,0,0,1)}")

    
