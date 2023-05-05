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

import logging
from geometry import GeometryRegister

log = logging.getLogger(__name__)


# Load the config YAML file from examples/conf/digit.yaml
@hydra.main(config_path="./data/conf", config_name="digit_test")
def main(cfg):
    # Initialize digits
    bg = cv2.imread("conf/bg_digit_240_320.jpg")
    digits = tacto.Sensor(**cfg.tacto, background=bg)

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Create and initialize DIGIT
    digit_body = px.Body(**cfg.digit)
    digits.add_camera(digit_body.id, [-1])

    # Add object to pybullet and tacto simulator
    obj = px.Body(**cfg.object)
    digits.add_body(obj)

    # Create control panel to control the 6DoF pose of the object
    panel = px.gui.PoseControlPanel(obj, 100,**cfg.object_control_panel)
    panel.start()
    # georeg = GeometryRegister(obj,digits)
    log.info("Use the slides to move the object until in contact with the DIGIT")

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    current_time = time.time()
    index = 0
    while True:
        color, depth = digits.render()
        digits.updateGUI(color, depth)
        # ipdb.set_trace()
        # georeg.rotate_object_with([1,1,0],[-1,-1,0],30)
        # if time.time() - current_time > 1:
        #     # imageio.imsave(f"output_tmp/color{index}.png", color[0])
        #     for obj_name in digits.objects.keys():
        #         log.info(f"object id: {obj_name}, object pose: {digits.object_poses[obj_name]}")
        #     current_time = time.time()
        #     index += 1

    t.stop()


if __name__ == "__main__":
    main()

