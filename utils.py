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


def random_tessellation_points(k):
    
    return np.random.uniform(low=0, high=50, size=[k, 2])


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """
    
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)



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


def add_first(array):
    assert isinstance(array, np.ndarray)

    return np.concatenate((array, np.array([array[0]])))


# compute Voronoi tesselation
def plot(points, limits=True):
    vor = Voronoi(points)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)


    # colorize
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.4)
        plt.plot(add_first(vertices[region][:, 0]), 
                 add_first(vertices[region][:, 1]),
                 c='k') # lines

    plt.plot(points[:,0], points[:,1], 'ko')
    for i, xy in enumerate(zip(points[:,0], points[:, 1])):
        plt.annotate(i, xy=xy, fontsize=13)
    plt.axis('equal')
    if limits:
        plt.xlim(0, 50)
        plt.ylim(0, 50)
    
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axvline(x=50, linestyle='--', color='k')
    
    
def out_of_bounds(coord, xlim=(0, 50), ylim=(0, 50)):
    assert len(xlim) == 2
    assert len(ylim) == 2
    
    _x, _y = coord
    
    x_lower, x_upper = xlim
    y_lower, y_upper = ylim
    
    return any([(_x < x_lower), (_x > x_upper), (_y < y_lower), (_y > y_upper)])
    
    
def PolyArea(coords):
    
    x, y = zip(*coords)
    
    right_indices = ConvexHull(coords).vertices
    x_temp = np.array(x)[right_indices]
    y_temp = np.array(y)[right_indices]
        
    return 0.5*np.abs(np.dot(x_temp, np.roll(y_temp, 1)) - np.dot(y_temp, np.roll(x_temp, 1)))



def filter_vertices(coords):
    # helper function for function below
    output = []
    
    for coord in coords:
        if not out_of_bounds(coord):
            output.append(coord)
    
    return np.array(output)


def get_coords_border(coi, close_coord):
    _x1, _y1 = coi
    _x2, _y2 = close_coord
    
    assert out_of_bounds(coi)
    assert not out_of_bounds(close_coord)
    
    m = (_y2 - _y1)/(_x2 - _x1)
    
    b = - m*_x1 + _y1
    assert np.abs(b + m*_x2 - _y2) < 1e-7 # tolerance, sanity check
    
    out1 = (50, m*50 + b)
    out2 = (0, b)
    out3 = (-b/m, 0)
    out4 = ((50-b)/m, 50)
    
    outputs = filter_vertices(np.array([out1, out2, out3, out4]))
    index   = np.argmin(cdist(outputs, coi.reshape(1, -1)))
    
    return list(outputs[index]) # just for coherence with function below


def compute_points_line(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    
    m = (y2 - y1)/(x2 - x1)
    b = y2 - m*x2
    
    x_linspace = np.linspace(x1, x2, num=1001)
    
    return list(zip(x_linspace, m*x_linspace + b))


# taken from https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def ccw(A,B,C):
    a_x, a_y = A
    b_x, b_y = B
    c_x, c_y = C
    
    return (c_y - a_y) * (b_x - a_x) > (b_y - a_y) * (c_x - a_x)

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def slope_intercept(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    
    m = (y2 - y1)/(x2 - x1)
    b = y2 - m*x2
    
    return m, b

def find_point_of_intersection(point1, point2):
    x_1, y_1 = point1
    x_2, y_2 = point2
    
    #hard-coded
    output = []

    if intersect(point1, point2, (0, 0), (0, 50)): # left line
        m, b = slope_intercept(point1, point2)
        output.append((0, b))

    if intersect(point1, point2, (0, 50), (50, 50)): # top line
        m, b = slope_intercept(point1, point2)
        output.append(((50-b)/m, 50))
        
    if intersect(point1, point2, (50, 50), (50, 0)): # right line
        m, b = slope_intercept(point1, point2)
        output.append((50, 50*m + b))
    
    if intersect(point1, point2, (50, 0), (0, 0)): # right line
        m, b = slope_intercept(point1, point2)
        output.append((-b/m, 0))
        
    assert all(not out_of_bounds(p) for p in output)
    
    return output # type list


def restricted_vertices(coords, point):
    
    def create_neighbor_indices(n):
        return list(zip(list(range(1, n)) + [0], [n-1] + list(range(n-1))))
    
    # only the ones within the 50 x 50 square
    filtered_coords = filter_vertices(coords) # coords in square
    global output
    output = []
    n = len(coords)
    
    neighbor_indices = create_neighbor_indices(n)
    bounds = [out_of_bounds(coord) for coord in coords]
    
    for i, (_x, _y) in enumerate(coords):
        
        previous, next = neighbor_indices[i]
        
        if out_of_bounds((_x, _y)):
            
            if out_of_bounds(coords[next]):
                if len(find_point_of_intersection((_x, _y), coords[next])) != 0:
                    for additional_coord in find_point_of_intersection((_x, _y), coords[next]):
                        output.append(additional_coord)

            if sum([bounds[previous], bounds[next]]) == 2:
                # both neighbors out of bounds
                
                pass

            elif sum([bounds[previous], bounds[next]]) == 0:
                # neither neighbor out of bounds
                # clockwise...
                output.append(get_coords_border(np.array([_x, _y]), coords[previous]))
                output.append(get_coords_border(np.array([_x, _y]), coords[next]))
            
                
            elif bounds[previous]:
                output.append(get_coords_border(np.array([_x, _y]), coords[next]))
                
            elif bounds[next]:
                output.append(get_coords_border(np.array([_x, _y]), coords[previous]))
            
            else:
                # shouldn't reach this step
                raise ValueError

        
        else:
            output.append([_x, _y])
            
    return np.array(output)



def l2_norm(point1, point2):
    # for adding corners and when cdist is too complicated
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


def inverse_v(u):
    # u ~ U(0, 1)
    # can extend to multidimensional

    
    return np.power(1/(1-u) - 1, 1/5)

def truncated_poisson(lam, size):
    KMAX = 30
    output = []
    
    while len(output) < size:
        candidate = np.random.poisson(lam=lam, size=1)[0]
        
        if candidate <= KMAX:
            output.append(candidate)
            
    return np.array(output)


def log_likelihood_ratio(y, x_new, x_old):
    # x_new is from new model
    # x_old is from old model
    return -1/(2*0.7**2)*(np.sum((y - x_new)**2) - np.sum((y - x_old)**2))


def poisson_ratio_pdf(k, lam=15):
    # ratio d_k+1/b_k = p(k)/p(k+1)
    return (k+1)/lam


def f(v):
    return 5*v**4/((1+v**5)**2)



class UpdatedVoronoi:
    def __init__(self, _points):
        self.points = _points
        self.vor = Voronoi(_points)
        self.regions, self.vertices = voronoi_finite_polygons_2d(self.vor)
        

        self.areas = np.array([PolyArea(self.updated_vertices[i]) for i in range(len(_points))]) # hashable
        self.x_heights = [] # indices
        
        coordinates = np.array([[(i + 0.5, j + 0.5) for i in range(50)] for j in range(50)])
        coordinates = coordinates.reshape(-1, 2)

        for (u, v) in coordinates:
            self.x_heights.append(np.argmin(cdist(_points, [[u, v]]).reshape(-1,)))
    
    @property
    def updated_vertices(self):
        new_vertices = {i: restricted_vertices(self.vertices[self.regions[i]], self.points[i]) for i in range(len(self.points))}

        for corner in [(0, 0), (0, 50), (50, 0), (50, 50)]:

            region_where_corner_belongs = np.argmin([l2_norm(p, corner) for p in self.points])
            temp_vertices = new_vertices[region_where_corner_belongs].copy()

            new_vertices[region_where_corner_belongs] = np.array(list(temp_vertices) + [corner])
            
        return new_vertices