import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tacto
import pybullet as p
import pybulletX as px
import hydra
import time
import ipdb
import imageio
from motion import MotionPlanner
from geometry import GeometryRegister

import logging

log = logging.getLogger(__name__)

def config_scene(cfg):
    # Initialize digits
    bg = cv2.imread("conf/bg_digit_240_320.jpg")
    digits = tacto.Sensor(**cfg.tacto, background=bg)

    # Create and initialize DIGIT
    digit_body = px.Body(**cfg.digit)
    digits.add_camera(digit_body.id, [-1])

    # Add object to pybullet and tacto simulator
    obj = px.Body(**cfg.object)
    digits.add_body(obj)
    return digits, obj

@hydra.main(config_path="./data/conf", config_name="digit")
def main(cfg):
    # Initialize World
    log.info("Initializing world")
    px.init()
    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)
    digits, obj = config_scene(cfg)
    # Create control panel to control the 6DoF pose of the object
    planner = MotionPlanner(obj, digits)
    # planner.start()
    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    current_time = time.time()
    while True:
        planner.update()
        color, depth = digits.render()
        digits.updateGUI(color, depth)
        if time.time() - current_time > 1:
            
            current_time = time.time()

    t.stop()

@hydra.main(config_path="./data/conf", config_name="digit")
def test(cfg):
    # Initialize World
    log.info("Initializing world")
    px.init()
    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)
    digits, obj = config_scene(cfg)
    # Create control panel to control the 6DoF pose of the object
    planner = MotionPlanner(obj, digits)
    # planner.start()
    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    depth = np.load("output_tmp/0_1_2_depth.npy")
    planner.record(depth, "output_tmp")

def visualize_result(path):
    point_cloud_data = np.load(os.path.join(path, "point_cloud_data.npy"))
    point_cloud_ref = np.load(os.path.join(path, "point_cloud_ref.npy"))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(point_cloud_data[:,0], point_cloud_data[:,1], point_cloud_data[:,2], s=1)
    ax.scatter(point_cloud_ref[:,0], point_cloud_ref[:,1], point_cloud_ref[:,2], s=10)
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    main()