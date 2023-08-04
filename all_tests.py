import numpy as np
from hsicfuse import hsicfuse
from hsic import hsic, human_readable_dict
from agginc.jax import agginc, human_readable_dict
from wittawatj_tests import nfsic, nyhsic, fhsic
from nystromhsic import nystromhsic


def hsicfuse_test(X, Y, key, seed):
    return int(hsicfuse(X, Y, key))

def hsic_test(X, Y, key, seed):
    return int(hsic(X, Y, key))

# hsicagginc200_test
def hsicagginc_test(X, Y, key, seed):
    return int(agginc("hsic", X, Y, R=200, return_dictionary=False))

# hsicaggincquad_test
def hsicagg_test(X, Y, key, seed):
    return int(agginc("hsic", X, Y, R=X.shape[0]-1, return_dictionary=False))

def nfsic_test(X, Y, key, seed):
    return int(nfsic(X, Y, seed))

def nyhsic_test(X, Y, key, seed):
    return int(nyhsic(X, Y, seed))

def fhsic_test(X, Y, key, seed):
    return int(fhsic(X, Y, seed))






