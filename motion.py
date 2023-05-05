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
import imageio
from utils import LoopThread
from geometry import GeometryRegister, ReferencePoint
from image import mask_size, initial_process, process

import logging

log = logging.getLogger(__name__)


def motion_generator(origin_pose):
    pos, ori = origin_pose
    for i in range(100):
        adjust_pos = (pos[0], pos[1], pos[2]- i * 0.001)
        yield adjust_pos, ori

class MotionPlanner:
    def __init__(self, obj, sensor, interval=0.05) -> None:
        self._loop_thread = LoopThread(interval, self.update)
        self.obj = obj
        self.sensor = sensor
        self.origin_pose = self.obj.get_base_pose()
        pos, ori = self.obj.get_base_pose()
        self.cid = p.createConstraint(
            self.obj.id,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            childFramePosition=pos,
            childFrameOrientation=ori,
        )
        self.georeg = GeometryRegister(self.obj)
        self.motion_generator = self.motion_generator_func()
        self.max_force = 100
        self.step_size = 1e-4
        self.move = 0
    
    def start(self):
        self._loop_thread.start()

    def add_object(self, obj):
        self.obj = obj

    def update(self):
        pos, ori = next(self.motion_generator)
        p.changeConstraint(self.cid, pos, ori, maxForce=self.max_force)

    def update_onpose(self, pose):
        pos, ori = pose
        p.changeConstraint(self.cid, pos, ori, maxForce=self.max_force)

    def motion_generator_func(self):
        # init contact
        color, depth = self.sensor.render()
        depth = depth[0]
        size = mask_size(depth)
        height = self.origin_pose[0][2]
        estimated_steps = int(3*height / self.step_size)
        for _ in range(estimated_steps):
            pos, ori = self.georeg.translate_object_with([0,0,-1],self.step_size)
            color, depth = self.sensor.render()
            cur_size = mask_size(depth)
            depth = depth[0]
            if cur_size == size and cur_size > 0:
                imageio.imwrite("output_tmp1/output_tmp1.png", color[0])
                self.record(depth, "output_tmp1")
                break
            else:
                size = cur_size
            yield pos, ori
        # for _ in range(estimated_steps):
        #     pos, ori = self.georeg.translate_object_with([0,0,1],self.step_size)
        #     if pos[2] >= height:
        #         break
        #     yield pos, ori

        # get next edge
        # ipdb.set_trace()
        edge = self.georeg.new_edges.pop(0)
        log.info(f"new edge: {edge[0].pixel_loc} {edge[1].pixel_loc}")
        center = 0.5*(edge[0].pixel_loc+edge[1].pixel_loc)
        image_center = self.georeg.image_center()
        move_vector = 2*(image_center - center.astype(np.int))
        edge[0].pixel_loc += move_vector
        edge[1].pixel_loc += move_vector
        edge[2].pixel_loc += move_vector
        center = self.georeg.pixel_to_world(center)
        stride = 2*np.linalg.norm(center-np.array([0,0,center[2]]))
        direction = (np.array([0,0,center[2]]) - center)/np.linalg.norm(np.array([0,0,center[2]]) - center)
        # direction = np.array([direction[0], -direction[1], direction[2]])
        # horizontal motion
        pos, ori = self.georeg.translate_object_with(direction, stride)
        yield pos, ori
        # pos, ori = self.georeg.translate_object_with([0,0,1], 0.006)
        # yield pos, ori
        # rotate around the edge
        estimated_steps = int(1e5)#int(3.14/2 / (self.step_size*1e2))
        edge1 = self.georeg.pixel_to_world(np.array(edge[0].pixel_loc))
        edge2 = self.georeg.pixel_to_world(np.array(edge[1].pixel_loc))
        log.info(f"edge1: {edge1} edge2: {edge2}")
        for _ in range(estimated_steps):
            pos, ori = self.georeg.rotate_object_with(edge1,edge2,(self.step_size*-2e1))
            color, depth = self.sensor.render()
            cur_size = mask_size(depth)
            depth = depth[0]
            if cur_size == size and cur_size > 0:
                self.record(depth, "output_tmp1",edge, False)
                break
            else:
                size = cur_size
            yield pos, ori

        # running idle
        while True:
            yield pos, ori

    def motion_generator_test_func(self):
      
        estimated_steps = int(1e5)#int(3.14/2 / (self.step_size*1e2))
        edge1 = np.array([0,1,0.04])
        edge2 = np.array([0,-1,0.04])
        log.info(f"edge1: {edge1} edge2: {edge2}")
        for _ in range(estimated_steps):
        # while True:
            
            pos, ori = self.georeg.rotate_object_with(edge1,edge2,(self.step_size*-1e3))
            # color, depth = self.sensor.render()
            # cur_size = mask_size(depth)
            # depth = depth[0]
            # if cur_size == size and cur_size > 0:
            #     self.record(depth, "output_tmp1",edge, False)
            #     break
            # else:
            #     size = cur_size
            yield pos, ori

        # running idle
        while True:
            yield pos, ori


    def record(self, depth, path,edge=None, initial=True, render=True):
        if initial:
            reference_points = initial_process(depth)
            self.georeg.register_point(reference_points)
        else:
            reference_points = process(depth,edge)
            self.georeg.register_point(reference_points,(edge[0].id,edge[1].id))
        
        # indices = [0,1,2]
        # np.save(os.path.join(path,f"{indices[0]}_{indices[1]}_{indices[2]}_depth.npy"), depth)
        vertice = [self.georeg.faces[-1][0], self.georeg.faces[-1][1], self.georeg.faces[-1][2]]
        indices = [vertice[0].id, vertice[1].id, vertice[2].id]
        # np.save(os.path.join(path,f"{indices[0]}_{indices[1]}_{indices[2]}_depth.npy"), depth)
        np.savez(os.path.join(path,f"{indices[0]}_{indices[1]}_{indices[2]}_depth.npz"), depth=depth, \
            pixel0_loc=vertice[0].pixel_loc, pixel1_loc=vertice[1].pixel_loc, pixel2_loc=vertice[2].pixel_loc)
        if initial:
            vertice = [self.georeg.faces[-2][0], self.georeg.faces[-2][1], self.georeg.faces[-2][2]]
            indices = [vertice[0].id, vertice[1].id, vertice[2].id]
            # np.save(os.path.join(path,f"{indices[0]}_{indices[1]}_{indices[2]}_depth.npy"), depth)
            np.savez(os.path.join(path,f"{indices[0]}_{indices[1]}_{indices[2]}_depth.npz"), depth=depth, \
            pixel0_loc=vertice[0].pixel_loc, pixel1_loc=vertice[1].pixel_loc, pixel2_loc=vertice[2].pixel_loc)
        if render:
            self.georeg.point_cloud_generation(path)

