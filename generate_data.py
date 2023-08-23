import numpy as np
import data as data

def generate_data(datatype, seed, n, d, freq):
    """
    Generate data pairs and split them in train and set samples.
    Inputs:
        datatype: "sin" or "gsign" (string)
        seed: random seed (integer)
        n: sample size (integer)
        d: dimension (integer)
        freq: frequence (positive number)
    Output:
        (tr, te) where (X_tr, Y_tr) = tr.xy() and (X_te, Y_te)
    """
    if datatype == "sin":
        ps = data.PSSinFreq(freq=freq, d=d)
    elif datatype == "gsign":
        ps = data.PSGaussSign(dx=d)
    else:
        raise ValueError("datatype should be either 'sin' or 'gsign' or 'msd'.")
    pdata = ps.sample(n, seed=seed)
    tr, te = pdata.split_tr_te(tr_proportion=0.5, seed=seed+5)
    return tr, te

def get_samples(tr, te):
    X_tr, Y_tr = tr.xy()
    X_te, Y_te = te.xy()
    return (X_tr, Y_tr), (X_te, Y_te) 




if __name__ == "__main__":
    datatype = "sin"
    seed = 0
    n = 100
    d = 1
    freq = 4
    tr, te = generate_data(datatype, seed, n, d, freq)
    (X_tr, Y_tr), (X_te, Y_te) = get_samples(tr, te)  
    import time
    t = time.time()
    generate_data(datatype, seed, n, d, freq)
    t1 = time.time()
    print(t1-t)