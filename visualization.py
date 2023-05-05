import os
import matplotlib.pyplot as plt
import numpy as np

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
    visualize_result("output_tmp1")