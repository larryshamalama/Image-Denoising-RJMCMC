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

from helper import *

# def get_neighbors(updated_vor, i):
#     assert i in updated_vor.updated_vertices.keys()
    
#     i_vertices = updated_vor.updated_vertices[i]
#     neighbors  = []
    
#     for j in range(len(updated_vor.points)):
        
#         if i == j:
#             continue
            
#         j_vertices = updated_vor.updated_vertices[j] # temp
        
#         x_intersects = np.intersect1d(i_vertices[:, 0], j_vertices[:, 0])
#         y_intersects = np.intersect1d(i_vertices[:, 1], j_vertices[:, 1])
        
#         same_x_coord = np.array([_i for (_i, x) in enumerate(x_intersects) if x in i_vertices])
#         same_y_coord = np.array([_j for (_j, x) in enumerate(y_intersects) if x in j_vertices])
        
#         if same_x_coord != []:
#             vertices_in_common = np.intersect1d(same_x_coord, same_y_coord)
            
#             if len(vertices_in_common) == len(same_x_coord):
#                 #assert len(vertices_in_common) == 2
#                 neighbors.append(j)
            
#     return np.array(neighbors)


class UpdatedVoronoi:
    def __init__(self, _points):
        self.points = _points

        if len(_points) > 2:
            self.vor = Voronoi(_points)
            self.regions, self.vertices = voronoi_finite_polygons_2d(self.vor)

            self.areas = np.array([PolyArea(self.updated_vertices[i]) for i in range(len(_points))]) # hashable
            self.x_heights = [] # indices
            
            coordinates = np.array([[(i + 0.5, j + 0.5) for i in range(50)] for j in range(50)])
            coordinates = coordinates.reshape(-1, 2)

            for (u, v) in coordinates:
                self.x_heights.append(np.argmin(cdist(_points, [[u, v]]).reshape(-1,)))

        elif len(_points) == 2:
            if _points[0][0] < _points[1][0]:
                p0, p1 = _points
            else:
                p1, p0 = _points # unpacking is possible

            m, b = slope_intercept(p0, p1)

            _vertices = {0: [], 1: []}

            vertices = [[], []]

            for i, corner in enumerate([(0, 0), (0, 50), (50, 0), (50, 50)]):

                if l2_norm(p0, corner) < l2_norm(p1, corner):
                    # closer to p0
                    _vertices[0].append(corner)
                    vertices[0].append(i)
                else:
                    _vertices[1].append(corner)
                    vertices[1].append(i)

            other_candidates = [(0, b), ((50-b)/m, 50), (50, 50*m + b), (-b/m, 0)]
            other_candidates = list(filter(lambda p: not out_of_bounds(p), other_candidates))

            for edge_point in other_candidates:
                _vertices[0].append(edge_point)
                _vertices[1].append(edge_point)

            vertices[0] += [4, 5]
            vertices[1] += [4, 5]

            self.points    = _points
            self._vertices = _vertices
            self.vertices  = vertices
            self.regions   = np.array([(0, 0), (0, 50), (50, 0), (50, 50)] + other_candidates)
    
            self.areas = np.array([PolyArea(self.updated_vertices[i]) for i in range(len(_points))]) # hashable
            self.x_heights = [] # indices
            
            coordinates = np.array([[(i + 0.5, j + 0.5) for i in range(50)] for j in range(50)])
            coordinates = coordinates.reshape(-1, 2)

            for (u, v) in coordinates:
                self.x_heights.append(np.argmin(cdist(_points, [[u, v]]).reshape(-1,)))


        elif len(_points) == 1:
            self.regions   = np.array([(0, 0), (0, 50), (50, 0), (50, 50)])
            self.vertices  = [0, 1, 2, 3]
            self.areas     = np.array([2500.])
            self.x_heights = [0]*2500
            self._vertices = {0: self.vertices}


    
    @property
    def updated_vertices(self):
        if len(self.points) in [1, 2]:
            return self._vertices

        new_vertices = {i: restricted_vertices(self.vertices[self.regions[i]], self.points[i]) for i in range(len(self.points))}

        for corner in [(0, 0), (0, 50), (50, 0), (50, 50)]:

            region_where_corner_belongs = np.argmin([l2_norm(p, corner) for p in self.points])
            temp_vertices = new_vertices[region_where_corner_belongs].copy()

            new_vertices[region_where_corner_belongs] = np.unique(np.array(list(temp_vertices) + [corner]), axis=0)
        
        
        return new_vertices