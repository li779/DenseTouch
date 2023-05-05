import numpy as np
import os
import matplotlib.pyplot as plt
import tacto
import pybullet as p
import pybulletX as px
from scipy.spatial import transform
import time
import ipdb
from utils import render


def barycentric_coord(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u,v,w])

def face_normal(a, b, c):
    v1 = b - a
    v2 = c - a
    c = np.cross(v1, v2)
    return c/np.linalg.norm(c)

class ReferencePoint:
    # pos in object frame
    def __init__(self, id, pos, pixel_loc) -> None:
        self.id = id
        self.pos = pos
        self.pixel_loc = pixel_loc
        self.neighbors = []
        self.edge = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
        self.edge.append(False)
        neighbor.neighbors.append(self)
        neighbor.edge.append(False)
    
    def mark_edge(self, id):
        index1 = 0
        for i in range(len(self.neighbors)):
            if self.neighbors[i].id == id:
                index1 = i
                break
        self.edge[index1] = True
        index2 = 0
        for i in range(len(self.neighbors[index1].neighbors)):
            if self.neighbors[index1].neighbors[i].id == self.id:
                index2 = i
                break
        self.edge[index2] = True


class GeometryRegister:
    def __init__(self, obj) -> None:
        self.obj = obj
        self.sensor_z = 0.021
        self.image_size = (160, 120)
        self.sensor_scale = 0.004/30
        self.points = []
        self.faces = []
        self.new_edges = []

    def world_to_obj(self, wld_pos):
        obj_pos, obj_orn = self.obj.get_base_pose()
        obj_pos = np.array(obj_pos)
        obj_orn = np.array(obj_orn)
        wld_pos = np.array(wld_pos)
        rot_matrix = np.array(p.getMatrixFromQuaternion(obj_orn)).reshape(3, 3)
        point_obj = np.linalg.inv(rot_matrix).dot(wld_pos-obj_pos)
        return point_obj
    
    def pixel_to_world(self,pixel):
        center = (self.image_size[0]//2, self.image_size[1]//2)
        return [(pixel[0]-center[0])*self.sensor_scale,-(pixel[1]-center[1])*self.sensor_scale,self.sensor_z]
    
    def image_center(self):
        return (self.image_size[0]//2, self.image_size[1]//2)
    
    def rotate_object_with(self, wld_pos1, wld_pos2, angle):
        # ipdb.set_trace()
        wld_pos1 = np.array(wld_pos1)
        wld_pos2 = np.array(wld_pos2)
        axis = (wld_pos2 - wld_pos1)/np.linalg.norm(wld_pos2 - wld_pos1)
        center = (wld_pos1+wld_pos2)/2
        rot = transform.Rotation.from_rotvec(angle*axis)
        obj_pos, obj_orn = self.obj.get_base_pose()
        new_pos = rot.as_matrix().dot(obj_pos-center) + center
        new_orn = (transform.Rotation.from_quat(obj_orn)*rot).as_quat()
        return new_pos, new_orn
    
    def translate_object_with(self, direction, length):
        obj_pos, obj_orn = self.obj.get_base_pose()
        new_pos = np.array(obj_pos) + length * np.array(direction)
        return new_pos, obj_orn
    
    def register_point(self, reference_points, ids=None):
        # ipdb.set_trace()
        index = len(self.points)
        center = (self.image_size[0]//2, self.image_size[1]//2)
        ref_world = [[(p[0]-center[0])*self.sensor_scale,(p[1]-center[1])*self.sensor_scale,self.sensor_z] for p in reference_points]
        if len(reference_points) == 4:
            p = [ReferencePoint(index+i, self.world_to_obj(ref_world[i]), reference_points[i]) for i in range(len(reference_points))]
            self.points.extend(p)
        else:
            p = [self.points[i] for i in ids]
            p.append(ReferencePoint(index, self.world_to_obj(ref_world[-1]), reference_points[-1]))
            self.points.append(ReferencePoint(index, self.world_to_obj(ref_world[-1]), reference_points[-1]))
        self.faces.append([p[0], p[1], p[2]])
        self.new_edges.append((p[0], p[2], p[1]))
        self.new_edges.append((p[2], p[1], p[0]))
        if len(p) == 4:
            self.faces.append([p[0], p[3], p[1]])
            self.new_edges.append((p[3], p[0], p[1]))
            self.new_edges.append((p[1], p[3], p[0]))

    def point_cloud_generation(self, path):
        # ipdb.set_trace()
        point_cloud_ref = [p.pos for p in self.points]
        point_cloud_data = []
        for i,f in enumerate(self.faces):
            data = np.load(os.path.join(path,f"{f[0].id}_{f[1].id}_{f[2].id}_depth.npz"))
            depth, pixel0_loc, pixel1_loc, pixel2_loc = data['depth'], data['pixel0_loc'], data['pixel1_loc'], data['pixel2_loc']
            pixel = np.array([pixel0_loc, pixel1_loc, pixel2_loc])
            Pos = np.array([f[0].pos, f[1].pos, f[2].pos])
            assert pixel.shape == (3,2)
            assert Pos.shape == (3,3)
            mask = np.zeros((self.image_size[0],self.image_size[1]))
            coord = np.zeros((self.image_size[0],self.image_size[1],3))
            for j in range(self.image_size[0]):
                for k in range(self.image_size[1]):
                    bary = barycentric_coord([j,k], pixel[0], pixel[1], pixel[2])
                    if np.all(bary >= 0) and np.all(bary <= 1):
                        mask[j,k] = 1
                        coord[j,k,:] = bary
            
            d = np.array(depth)[mask==1].reshape(-1,1)
            mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
            point_cloud = ((coord@Pos)[mask==1]).reshape(-1,3)+d@face_normal(Pos[0],Pos[1],Pos[2]).reshape(1,3)
            point_cloud_data.append(point_cloud)
        point_cloud_data = np.concatenate(point_cloud_data, axis=0)
        point_cloud_ref = np.array(point_cloud_ref)
        np.save(f"{path}/point_cloud_data.npy", point_cloud_data)
        np.save(f"{path}/point_cloud_ref.npy", point_cloud_ref)
        # ipdb.set_trace()
        # render(point_cloud_data, "output_tmp/point_cloud_data.gif")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(point_cloud_data[:,0], point_cloud_data[:,1], point_cloud_data[:,2], s=1)
        ax.scatter(point_cloud_ref[:,0], point_cloud_ref[:,1], point_cloud_ref[:,2], s=10)
        ax.set_aspect('equal')
        plt.show()
        fig.savefig(f"{path}/point_cloud_data.png")
