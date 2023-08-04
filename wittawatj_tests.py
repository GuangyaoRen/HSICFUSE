"""
The tests in this file are taken from the implementation of Wittawat Jitkrittum (https://github.com/wittawatj/).

- Test: NFSIC (Finite Set Independence Criterion)
- Paper: [An Adaptive Test of Independence with Analytic Kernel Embeddings](http://proceedings.mlr.press/v70/jitkrittum17a/jitkrittum17a.pdf)
- Authors: Wittawat Jitkrittum, Zoltán Szabó, Arthur Gretton
- Code: [fsic-test repository](https://github.com/wittawatj/fsic-test) by [Wittawat Jitkrittum](https://github.com/wittawatj)
"""


import numpy as np
import fsic.util as util
import fsic.indtest as it
import fsic.kernel as kernel
import fsic.feature as fea
from fsic.indtest import GaussNFSIC as fsic_GaussNFSIC
from fsic.data import PairedData as fsic_PairedData
import logging


def kl_kgauss_median(pdata):
    """
    Get two Gaussian kernels constructed with the median heuristic.
    """
    xtr, ytr = pdata.xy()
    dx = xtr.shape[1]
    dy = ytr.shape[1]
    medx2 = util.meddistance(xtr, subsample=1000)**2
    medy2 = util.meddistance(ytr, subsample=1000)**2
    k = kernel.KGauss(medx2)
    l = kernel.KGauss(medy2)
    return k, l


def kl_kgauss_median_bounds(pdata):
    k, l = it.kl_kgauss_median(pdata)
    # make sure that the widths are not too small. 
    k.sigma2 = max(k.sigma2, 1e-1)
    l.sigma2 = max(l.sigma2, 1e-1)
    return k, l
    

# based on function job_nfsicJ10_stoopt()
# https://github.com/wittawatj/fsic-test/blob/master/fsic/ex/ex2_prob_params.py
def nfsic(X, Y, r, J=10, n_permute=500, alpha=0.05):
    pdata = fsic_PairedData(X, Y)
    tr, te = pdata.subsample(X.shape[0], seed=r+4).split_tr_te(tr_proportion=0.5, seed=r+5)
    nfsic_opt_options = {'n_test_locs':J, 'max_iter':200,
    'V_step':1, 'W_step':1, 'gwidthx_step':1, 'gwidthy_step':1,
    'batch_proportion':0.7, 'tol_fun':1e-4, 'step_pow':0.5, 'seed':r+2,
    'reg': 1e-6}
    op_V, op_W, op_gwx, op_gwy, info = fsic_GaussNFSIC.optimize_locs_widths(tr,
            alpha, **nfsic_opt_options )
    nfsic_opt = fsic_GaussNFSIC(op_gwx, op_gwy, op_V, op_W, alpha, reg='auto', n_permute=n_permute, seed=r+3)
    nfsic_opt_result  = nfsic_opt.perform_test(te)
    return int(nfsic_opt_result['h0_rejected'])


# job_nyhsic_med(
def nyhsic(X, Y, r, n_features=10, alpha=0.05):
    """
    HSIC with Nystrom approximation. 
    """
    n_simulate = 2000
    # use full sample for testing. Merge training and test sets
    pdata = fsic_PairedData(X, Y)
    tr, te = pdata.subsample(X.shape[0], seed=r+4).split_tr_te(tr_proportion=0.5, seed=r+5)
    pdata = tr + te
    X, Y = pdata.xy()
    k, l = kl_kgauss_median_bounds(pdata)
    # randomly choose the inducing points from X, Y
    induce_x = util.subsample_rows(X, n_features, seed=r+2)
    induce_y = util.subsample_rows(Y, n_features, seed=r+3)

    nyhsic = it.NystromHSIC(k, l, induce_x, induce_y, n_simulate=n_simulate, alpha=alpha, seed=r+89)
    nyhsic_result = nyhsic.perform_test(pdata)
    return int(nyhsic_result['h0_rejected'])


# job_fhsic_med(
def fhsic(X, Y, r, n_features=10, alpha=0.05):
    """
    HSIC with random Fourier features. 
    """
    n_simulate = 2000
    # use full sample for testing. Merge training and test sets
    pdata = fsic_PairedData(X, Y)
    tr, te = pdata.subsample(X.shape[0], seed=r+4).split_tr_te(tr_proportion=0.5, seed=r+5)
    pdata = tr + te
    X, Y = pdata.xy()
    medx = util.meddistance(X, subsample=1000)
    medy = util.meddistance(Y, subsample=1000)
    sigmax2 = medx**2
    sigmay2 = medy**2
    fmx = fea.RFFKGauss(sigmax2, n_features=n_features, seed=r+1)
    fmy = fea.RFFKGauss(sigmay2, n_features=n_features, seed=r+2)
    ffhsic = it.FiniteFeatureHSIC(fmx, fmy, n_simulate=n_simulate, alpha=alpha, seed=r+89)
    ffhsic_result = ffhsic.perform_test(pdata)
    return int(ffhsic_result['h0_rejected'])











