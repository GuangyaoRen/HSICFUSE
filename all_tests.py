import numpy as np
from hsicfuse import hsicfuse
from hsic import hsic, human_readable_dict
from agginc.jax import agginc, human_readable_dict
from wittawatj_tests import nfsic


def hsicfuse_test(X, Y, key, seed):
    return int(hsicfuse(X, Y, key))

def hsic_test(X, Y, key, seed):
    return int(hsic(X, Y, key))

def hsicagginc1_test(X, Y, key, seed):
    return int(agginc("hsic", X, Y, R=1, return_dictionary=False))

def hsicagginc100_test(X, Y, key, seed):
    return int(agginc("hsic", X, Y, R=100, return_dictionary=False))

def hsicagginc200_test(X, Y, key, seed):
    return int(agginc("hsic", X, Y, R=200, return_dictionary=False))

def hsicaggincquad_test(X, Y, key, seed):
    return int(agginc("hsic", X, Y, R=X.shape[0]-1, return_dictionary=False))

def nfsic_test(X, Y, key, seed):
    return int(nfsic(X, Y, seed))




