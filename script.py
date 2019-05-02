# for running on the cloud

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import seaborn as sns

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull

import itertools
import random

from utils import *
from collections import Counter

plt.style.use('seaborn-darkgrid')


np.random.seed(260647775)

def condition(coord):
    x, y = coord
    
    return (x-25)**2 + (y-25)**2 < 225 # radius 15

grid      = [[(x, y) for y in range(50)] for x in range(50)]
grayscale = [[2 if condition(tup) else 0.5 for tup in row] for row in grid]
noise = np.random.normal(0, 0.7, size=(50, 50))
noisy_image = grayscale - noise

lam = 15
k   = 15

coordinates = np.array([[(i + 0.5, j + 0.5) for i in range(50)] for j in range(50)])
coordinates = coordinates.reshape(-1, 2)

NUM_ITER = 24000

points  = random_tessellation_points(k)
old_Voronoi = UpdatedVoronoi(points)

x_sample = []
k_sample = []

count = 0
death = 0
birth = 0
skip  = 0

while len(x_sample) < NUM_ITER:

    if count % 100 == 0:
        print('Done sampling ', count, ', current k = ', k)
    
    points  = random_tessellation_points(k)
    old_Voronoi = UpdatedVoronoi(points)
    
    counter = Counter(old_Voronoi.x_heights)
    ni = np.array([counter[r] for r in range(k)])
    sum_yi = np.array([sum(noisy_image.reshape(-1)[np.argwhere(np.array(old_Voronoi.x_heights) == r).reshape(-1)]) for r in range(k)])
    
    heights = np.random.normal(loc=(sum_yi + 0.7**2)/ni, scale=np.sqrt(0.7**2/ni))
    
#     if any(heights <= 0):
#         invalid_heights = np.argwhere(heights <= 0).reshape(-1)
#         heights[invalid_heights] = 0.1 # sketchy solution

    new_point   = np.random.uniform(low=0, high=50, size=[1, 2])
    temp_points = np.concatenate((points, new_point.reshape(1, 2)))
    new_Voronoi = UpdatedVoronoi(temp_points)

#    J = get_neighbors(new_Voronoi, k) # k is last index of new voronoi, J as defined in Green (1995)
    diff_areas = (old_Voronoi.areas - new_Voronoi.areas[:-1])
    J = np.argwhere(diff_areas > 1e-7).reshape(-1).astype(np.int32)
    
#    assert all(np.argwhere(diff_areas > 1e-7).reshape(-1).astype(np.int32) == np.sort(J))

    S, T = diff_areas[J], new_Voronoi.areas[J] # change in areas, new areas
    
    try:
        assert np.abs(sum(S) - new_Voronoi.areas[-1]) < 1e-7 # change in areas same as area of new region
    except:
        print('Error in birth process')
        print('S: ', S)
        print('Expected area: ', new_Voronoi.areas[-1])
        print('Current number of tiles (k): ', k)
        print('Neighbors of i*: ', J)
        print('')

    v = inverse_v(np.random.uniform(0, 1)) # ~ f(v)
    h_tilde = np.exp(1/sum(S)*(S@np.log(heights[J]))) # no worries about height = 0
    h_star  = h_tilde*v
    heights = np.array(list(heights) + [h_star])

    new_heights  = heights[J]**(1+S/T)*(np.tile(h_star, len(J))**(-S/T))
    loglikeratio = log_likelihood_ratio(noisy_image.reshape(-1,), 
                                        heights[new_Voronoi.x_heights], 
                                        heights[old_Voronoi.x_heights])
    # check if R < or > than 0
    logR = loglikeratio + \
           np.log(lam) - \
           h_star - \
           sum(new_heights - heights[J]) - \
           np.log(poisson_ratio_pdf(k)*h_tilde/f(v)) - \
           np.log(sum((S+T)*new_heights/(T*heights[J])))



    if logR < 0:
        if np.random.binomial(n=1, p=np.exp(logR)):
            birth += 1
            
            heights[J] = new_heights
            assert len(heights) == max(new_Voronoi.x_heights) + 1 # birth

            k = k+1
            x_sample.append(heights[new_Voronoi.x_heights])
            k_sample.append(k)
            
        else:
            x_sample.append(heights[old_Voronoi.x_heights])
            k_sample.append(k)

    else:
        if np.random.binomial(n=1, p=np.exp(-logR)):
            
            if k < 1:
                skip += 1
                continue
                
            death += 1
            
            heights = heights[:-1] # removing h_star
            

            delete_tile = random.choice(range(len(points)))
            temp_points = np.delete(points, delete_tile, axis=0)

            new_Voronoi = UpdatedVoronoi(temp_points)

            #J = get_neighbors(old_Voronoi, delete_tile) # tile no longer exists in new_Voronoi
            
            _temp_areas = np.insert(new_Voronoi.areas, delete_tile, 0)
            diff_areas  = (_temp_areas - old_Voronoi.areas)

            J = np.argwhere(diff_areas > 1e-7).reshape(-1).astype(np.int32)

            S, T = diff_areas[J], old_Voronoi.areas[J] # change in areas, new areas

            try:
                assert np.abs(sum(S) - old_Voronoi.areas[delete_tile]) < 1e-7 # change in areas same as area of new region
            except Exception as e:
                print('Error in death process')
                print('S: ', S)
                print('Expected area: ', new_Voronoi.areas[-1])
                print('Current number of tiles (k): ', k)
                print('Neighbors of delete_tile: ', J)
                print('')

            v = inverse_v(np.random.uniform(0, 1))
            h_tilde = np.exp(1/sum(S)*(S@np.log(heights[J]))) # no worries about height = 0
            h_star  = h_tilde*v
            #height
            s = np.insert(heights, delete_tile, 0)

            new_heights  = heights[J]**(T/(S+T))*(h_star**(S/(S+T)))
            heights[J] = new_heights
            heights = np.delete(heights, delete_tile)

            assert len(heights) == max(new_Voronoi.x_heights) + 1 # death
            
            k = k-1
            x_sample.append(heights[new_Voronoi.x_heights])
            k_sample.append(k)
        
        else:
            x_sample.append(heights[old_Voronoi.x_heights])
            k_sample.append(k)

    count += 1

np.save('x_samples.npy', np.array(x_sample))
np.save('k_samples.npy', np.array(k_sample))
