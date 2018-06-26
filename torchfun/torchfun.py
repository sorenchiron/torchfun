from __future__ import division,print_function
import numpy as np
import torch as t

def flatten(x):
    shapes = x.shape
    n = shapes[0]
    total_numbers = np.prod(shapes)
    flatten_length = total_numbers // n
    return x.view(-1,flatten_length)

def imshow(x):
    '''only deal with torch channel-first image batch'''
    import torchvision
    shapes = x.shape
    if len(shapes)==3:
        x = t.unsqueeze(x,dim=0)
    grid = torchvision.utils.make_grid(x)
    gridnp = grid.numpy()
    max_intensity = gridnp.max()
    min_intensity = gridnp.min()
    if min_intensity>=0 and max_intensity>1:
        # 0 - 255
        gridnp /= 255
    elif min_intensity<0 and min_intensity>=-0.5 and max_intensity>0 and max_intensity <=0.5:
        # -0.5 - 0.5
        gridnp += 0.5
    elif min_intensity<-0.5 and min_intensity>=-1 and max_intensity>0.5 and max_intensity <=1:
        # -1 - 1
        gridnp /= 2
        gridnp += 0.5

    import matplotlib.pyplot as plt
    plt.imshow(np.transpose(gridnp,(1,2,0)))
    plt.show()
    plt.close()
    del plt
