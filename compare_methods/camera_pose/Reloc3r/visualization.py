import open3d as o3d
import numpy as np 
import os
from glob import glob
from pdb import set_trace as bb


class Viewpoint(object):
    def __init__(self, name, w, h, intrinsics, pose, color=(1,0,0), scale=0.1):
        super(Viewpoint, self).__init__()
        self.name = name
        self.w = w
        self.h = h
        self.intrinsics = intrinsics
        self.pose = pose
        self.scale = scale
        self.color = color

def get_viewpoint_geometry_in_o3d(viewpoint):
    v = o3d.geometry.LineSet.create_camera_visualization(
        viewpoint.w, viewpoint.h, 
        viewpoint.intrinsics, viewpoint.pose, 
        scale=viewpoint.scale) 
    v.paint_uniform_color(viewpoint.color)
    return v

def visualize_views(viewpoints, linesets=None):
    geometry_list = []
    for viewpoint in viewpoints:
        geometry_list.append(get_viewpoint_geometry_in_o3d(viewpoint))
    if linesets is not None:  # axes for the first database view
        geometry_list += linesets
    vis = o3d.visualization.Visualizer()
    
    vis.create_window()
    for geo in geometry_list:
        vis.add_geometry(geo)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='visualize poses')
    parser.add_argument('--mode', type=str, required=True)  
    parser.add_argument('--pose_path', type=str, default='./data/wild_images/pose2to1.txt')
    parser.add_argument('--pose_folder', type=str, default='./data/wild_video/poses')
    args = parser.parse_args()

    assert args.mode in ['relpose', 'visloc']

    # configure as desired...  
    image_width, image_height = 640, 480  
    intrinsics = np.array([[525, 0, 320],[0, 525, 240],[0, 0, 1]], dtype=float)  

    color_red = (245/255., 67/255., 62/255.)  # RGB
    color_green = (136/255., 176/255., 109/255.)
    color_blue = (93/255., 141/255., 253/255.)
    color_orange = (245/255., 168/255., 61/255.)

    # lines for x-axis
    points = [np.array([0,0,0]), np.array([1,0,0])]
    lines = [[0,1]]
    points = np.array(points)
    lines = np.array(lines)
    lineset_x = o3d.geometry.LineSet()
    lineset_x.points = o3d.utility.Vector3dVector(points)
    lineset_x.lines = o3d.utility.Vector2iVector(lines)
    lineset_x.paint_uniform_color((255,0,0)) 
    # lines for y-axis
    points = [np.array([0,0,0]), np.array([0,1,0])]
    lines = [[0,1]]
    points = np.array(points)
    lines = np.array(lines)
    lineset_y = o3d.geometry.LineSet()
    lineset_y.points = o3d.utility.Vector3dVector(points)
    lineset_y.lines = o3d.utility.Vector2iVector(lines)
    lineset_y.paint_uniform_color((0,255,0)) 
    # lines for z-axis
    points = [np.array([0,0,0]), np.array([0,0,1])]
    lines = [[0,1]]
    points = np.array(points)
    lines = np.array(lines)
    lineset_z = o3d.geometry.LineSet()
    lineset_z.points = o3d.utility.Vector3dVector(points)
    lineset_z.lines = o3d.utility.Vector2iVector(lines)
    lineset_z.paint_uniform_color((0,0,255)) 

    vps = []
    if args.mode == 'relpose':
        scale = 0.2
        relpose = np.loadtxt(args.pose_path)
        pose0 = np.identity(4)
        pose1 = pose0 @ relpose
        vps.append(Viewpoint('v1', image_width, image_height, intrinsics, np.linalg.inv(pose0), color=color_green, scale=scale))
        vps.append(Viewpoint('v2', image_width, image_height, intrinsics, np.linalg.inv(pose1), color=color_orange, scale=scale))
    elif args.mode == 'visloc':
        scale = 0.03
        pose_files = sorted(glob('{}/pose*.txt'.format(args.pose_folder)))
        pose_beg = np.loadtxt(pose_files[0])
        pose_end = np.loadtxt(pose_files[-1])
        pp = np.linalg.inv(pose_beg)  # relative transformation to align the first frame to identity
        pose_beg = pp @ pose_beg
        pose_end = pp @ pose_end
        vps.append(Viewpoint('beg', image_width, image_height, intrinsics, np.linalg.inv(pose_beg), color=color_green, scale=scale))
        vps.append(Viewpoint('end', image_width, image_height, intrinsics, np.linalg.inv(pose_end), color=color_green, scale=scale))
        for pid in range(1, len(pose_files)-1):
            pose = np.loadtxt(pose_files[pid])
            pose = pp @ pose
            vps.append(Viewpoint('pose{}'.format(pid), image_width, image_height, intrinsics, np.linalg.inv(pose), color=color_blue, scale=scale))


    visualize_views(vps, [lineset_x, lineset_y, lineset_z])

