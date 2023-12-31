"""
Code is taken from:
    An Adaptive Test of Independence with Analytic Kernel Embeddings
    Wittawat Jitkrittum, Zoltan Szabo, Arthur Gretton
    ICML 2017
    https://github.com/wittawatj/fsic-test/blob/master/fsic/util.py 
"""

"""A module containing convenient methods for general machine learning"""

__author__ = 'wittawat'

import numpy as np
import time 

class ContextTimer(object):
    """
    A class used to time an executation of a code snippet. 
    Use it with with .... as ...
    For example, 

        with ContextTimer() as t:
            # do something 
        time_spent = t.secs

    From https://www.huyng.com/posts/python-performance-analysis
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start 
        if self.verbose:
            print('elapsed time: %f ms' % (self.secs*1000))

# end class ContextTimer

class NumpySeedContext(object):
    """
    A context manager to reset the random seed by numpy.random.seed(..).
    Set the seed back at the end of the block. 
    """
    def __init__(self, seed):
        self.seed = seed 

    def __enter__(self):
        rstate = np.random.get_state()
        self.cur_state = rstate
        np.random.seed(self.seed)
        return self

    def __exit__(self, *args):
        np.random.set_state(self.cur_state)


def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X**2, 1)
    sy = np.sum(Y**2, 1)
    D2 =  sx[:, np.newaxis] - 2.0*X.dot(Y.T) + sy[np.newaxis, :] 
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D


def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and 
        there are more slightly more 0 than 1. In this case, the m

    Return
    ------
    median distance
    """
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)


def is_real_num(x):
    """return true if x is a real number"""
    try:
        float(x)
        return not (np.isnan(x) or np.isinf(x))
    except ValueError:
        return False
    

def tr_te_indices(n, tr_proportion, seed=9282 ):
    """Get two logical vectors for indexing train/test points.

    Return (tr_ind, te_ind)
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    Itr = np.zeros(n, dtype=bool)
    tr_ind = np.random.choice(n, int(tr_proportion*n), replace=False)
    Itr[tr_ind] = True
    Ite = np.logical_not(Itr)

    np.random.set_state(rand_state)
    return (Itr, Ite)

def subsample_ind(n, k, seed=32):
    """
    Return a list of indices to choose k out of n without replacement
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    ind = np.random.choice(n, k, replace=False)
    np.random.set_state(rand_state)
    return ind

def subsample_rows(X, k, seed=29):
    """
    Subsample k rows from the matrix X.
    """
    n = X.shape[0]
    if k > n:
        raise ValueError('k exceeds the number of rows.')
    ind = subsample_ind(n, k, seed=seed)
    return X[ind, :]
    

def cca(X, Y, reg=1e-5):
    """
    - X: n x dx data matrix
    - Y: n x dy data matrix
    
    Return (vals, Vx, Vy) where vals is a numpy array of decreasing eigenvalues,
        Vx is a square matrixk whose columns are eigenvectors for X corresponding to vals.
        Vy is a square matrixk whose columns are eigenvectors for Y corresponding to vals.
    """
    #return _cca_one_eig(X, Y, reg)
    return _cca_two_eig(X, Y, reg)

def _cca_two_eig(X, Y, reg=1e-5):
    """
    CCA formulation solving two eigenvalue problems.
    """
    dx = X.shape[1]
    dy = Y.shape[1]
    assert X.shape[0] == Y.shape[0]
    n = X.shape[0]
    mx = np.mean(X, 0)
    my = np.mean(Y, 0)
    # dx x dy
    Cxy = X.T.dot(Y)/n - np.outer(mx, my)
    Cxx = np.cov(X.T)
    #print Cxx
    Cyy = np.cov(Y.T)
    # Cxx, Cyy have to be invertible

    if dx == 1:
        CxxICxy = Cxy/Cxx
    else:
        CxxICxy = np.linalg.solve(Cxx + reg*np.eye(dx), Cxy)

    if dy==1:
        CyyICyx = Cxy.T/Cyy    
    else:
        CyyICyx = np.linalg.solve(Cyy + reg*np.eye(dy), Cxy.T)

    # problem for a
    avals, aV = np.linalg.eig(CxxICxy.dot(CyyICyx))
    #print avals
    #print 'aV'
    #print aV
    # problem for b
    bvals, bV = np.linalg.eig(CyyICyx.dot(CxxICxy))
    #print bvals
    #print 'bV'
    #print bV

    #from IPython.core.debugger import Tracer 
    #Tracer()()

    dim = min(dx, dy)
    # sort descendingly
    Ia = np.argsort(-avals)
    avals = avals[Ia[:dim]]
    aV = aV[:, Ia[:dim]]

    Ib = np.argsort(-bvals)
    bvals = bvals[Ib[:dim]]
    bV = bV[:, Ib[:dim]]
    np.testing.assert_array_almost_equal(avals, bvals)
    return np.real(avals), np.real(aV), np.real(bV)


def _cca_one_eig(X, Y, reg=1e-5):
    """
    CCA formulation with one big block diagonal eigenvalue problem.
    """
    #raise RuntimeError('There is a bug in this one. Eigenvalues can be outside [-1, 1]. See _cca_one_eig() instead')
    
    dx = X.shape[1]
    dy = Y.shape[1]
    assert X.shape[0] == Y.shape[0]
    n = X.shape[0]
    mx = np.mean(X, 0)
    my = np.mean(Y, 0)
    # dx x dy
    Cxy = X.T.dot(Y)/n - np.outer(mx, my)
    Cxx = np.cov(X.T)
    #print Cxx
    Cyy = np.cov(Y.T)
    # Cxx, Cyy have to be invertible
    if dx == 1:
        CxxICxy = Cxy/Cxx
    else:
        CxxICxy = np.linalg.solve(Cxx+reg*np.eye(dx), Cxy)
        
    if dy==1:
        CyyICyx = Cxy.T/Cyy    
    else:
        CyyICyx = np.linalg.solve(Cyy+reg*np.eye(dy), Cxy.T)
    # CCA block matrix
    #print CyyICyx
    R1 = np.hstack((np.zeros((dx, dx)), CxxICxy ))
    R2 = np.hstack((CyyICyx, np.zeros((dy, dy))) )
    B = np.vstack((R1, R2))
    assert B.shape[0] == B.shape[1]

    # eigen problem
    vals, V = np.linalg.eig(B)
    dim = min(dx, dy)
    # sort descendingly
    I = np.argsort(-vals)
    vals = vals[I[:dim]]
    V = V[:, I]
    Vx = V[:dx, :dim]
    Vy = V[dx:, :dim]
    return np.real(vals), np.real(Vx), np.real(Vy)


def fit_gaussian_draw(X, J, seed=28, reg=1e-7, eig_pow=1.0):
    """
    Fit a multivariate normal to the data X (n x d) and draw J points 
    from the fit. 
    - reg: regularizer to use with the covariance matrix
    - eig_pow: raise eigenvalues of the covariance matrix to this power to construct 
        a new covariance matrix before drawing samples. Useful to shrink the spread 
        of the variance.
    """
    with NumpySeedContext(seed=seed):
        d = X.shape[1]
        mean_x = np.mean(X, 0)
        cov_x = np.cov(X.T)
        if d==1:
            cov_x = np.array([[cov_x]])
        [evals, evecs] = np.linalg.eig(cov_x)
        evals = np.maximum(0, np.real(evals))
        assert np.all(np.isfinite(evals))
        evecs = np.real(evecs)
        shrunk_cov = evecs.dot(np.diag(evals**eig_pow)).dot(evecs.T) + reg*np.eye(d)
        V = np.random.multivariate_normal(mean_x, shrunk_cov, J)
    return V

def bound_by_data(Z, Data):
    """
    Determine lower and upper bound for each dimension from the Data, and project 
    Z so that all points in Z live in the bounds.

    Z: m x d 
    Data: n x d

    Return a projected Z of size m x d.
    """
    n, d = Z.shape
    Low = np.min(Data, 0)
    Up = np.max(Data, 0)
    LowMat = np.repeat(Low[np.newaxis, :], n, axis=0)
    UpMat = np.repeat(Up[np.newaxis, :], n, axis=0)

    Z = np.maximum(LowMat, Z)
    Z = np.minimum(UpMat, Z)
    return Z


def one_of_K_code(arr):
    """
    Make a one-of-K coding out of the numpy array.
    For example, if arr = ([0, 1, 0, 2]), then return a 2d array of the form 
     [[1, 0, 0], 
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, 2]]
    """
    U = np.unique(arr)
    n = len(arr)
    nu = len(U)
    X = np.zeros((n, nu))
    for i, u in enumerate(U):
        Ii = np.where( np.abs(arr - u) < 1e-8 )
        #ni = len(Ii)
        X[Ii[0], i] = 1
    return X

def fullprint(*args, **kwargs):
    "https://gist.github.com/ZGainsforth/3a306084013633c52881"
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold='nan')
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)

def standardize(X):
    mx = np.mean(X, 0)
    stdx = np.std(X, axis=0)
    # Assume standard deviations are not 0
    Zx = (X-mx)/stdx
    assert np.all(np.isfinite(Zx))
    return Zx