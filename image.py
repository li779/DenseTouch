import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tacto
import pybullet as p
import pybulletX as px
import time
import ipdb

def mask_size(depth):
    return np.count_nonzero(np.array(depth)>1e-5)

def initial_process(depth):
    reference_points = []
    mask = np.array(depth)>1e-5
    indices = np.argwhere(mask)
    mid = depth.shape[1]//2
    reference_points.append([indices[0,0], mid])
    reference_points.append([indices[-1,0], mid])
    mid = (indices[0,0]+indices[-1,0])//2
    indice_row = np.argwhere(mask[mid,:])
    reference_points.append(np.array([mid, indice_row[0,0]]))
    reference_points.append(np.array([mid, indice_row[-1,0]]))
    # fig = plt.figure()
    # plt.imshow(depth)
    # plt.scatter([reference_points[0][1], reference_points[1][1], reference_points[2][1], reference_points[3][1]], [reference_points[0][0], reference_points[1][0], reference_points[2][0], reference_points[3][0]])
    # fig.savefig('output_tmp/test.png')
    return reference_points

def process(depth, edge):
    # ipdb.set_trace()
    reference_points = []
    mask = np.array(depth)>1e-5
    center = 0.5*(edge[0].pixel_loc+edge[1].pixel_loc)
    i = 0
    p3 = center.astype(int)
    while mask[p3[0], p3[1]]:
        i += 1
        p3 = center + i*(edge[2].pixel_loc-center)/np.linalg.norm(edge[2].pixel_loc-center)
        p3 = p3.astype(int)
    reference_points.append(edge[0].pixel_loc)
    reference_points.append(edge[1].pixel_loc)
    reference_points.append(p3)
    
    # fig = plt.figure()
    # plt.imshow(depth)
    # plt.scatter([reference_points[0][1], reference_points[1][1], reference_points[2][1], reference_points[3][1]], [reference_points[0][0], reference_points[1][0], reference_points[2][0], reference_points[3][0]])
    # fig.savefig('output_tmp/test.png')
    return reference_points