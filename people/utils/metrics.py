import numpy as np

def abs_mpjpe(p1, p2):
    return np.mean(np.sqrt(np.sum(np.square(p1 - p2), axis=-1)))

def rel_mpjpe(p1, p2):
    p1r = p1.copy() - p1[:, 0:1, :]
    p2r = p2.copy() - p2[:, 0:1, :]
    return abs_mpjpe(p1r, p2r)
