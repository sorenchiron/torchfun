from __future__ import division,print_function
import numpy as np
import torch as t

def flatten(x):
    shapes = x.shape
    n = shapes[0]
    total_numbers = np.prod(shapes)
    flatten_length = total_numbers // n
    return x.view(-1,flatten_length)