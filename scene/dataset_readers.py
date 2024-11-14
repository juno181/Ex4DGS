#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from typing import NamedTuple
import re
import json
import glob
from pathlib import Path
from itertools import compress

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
import numpy as np
from plyfile import PlyData, PlyElement
import natsort

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.sh_utils import SH2RGB
from scene.c_gaussian_model import BasicPointCloud, BasicPointCloud_pc


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array #
    FovX: np.array #
    image: np.array
    image_path: str
    image_name: str
    opticalflow_path: tuple
    depth_path: str
    width: int #
    height: int #
    near: float
    far: float
    timestamp: float
    pose: np.array 
    hpdirecitons: np.array
    cxr: float
    cyr: float
    

class CameraInfo2():
    def __init__(self, uid, R, T, FovY, FovX, image_path, image_name, width, height, near, far, timestamp, pose, hpdirecitons, cxr, cyr):
        self.uid = uid
        self.R = R
        self.T = T
        self.FovY = FovY
        self.FovX = FovX
        self.image_path = image_path
        self.image_name = image_name
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        self.timestamp = timestamp
        self.pose = pose
        self.hpdirecitons = hpdirecitons
        self.cxr = cxr
        self.cyr = cyr
    
        self.image = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    
    return cam_infos


def readColmapCamerasTechnicolor(cam_extrinsics, cam_intrinsics, source_path, near, far, args, startime=0, endtime=-1):
    scene_name = os.path.basename(source_path)
    if scene_name =="Painter":
        start_idx = 0
    else:
        start_idx = 1
    cam_infos = []
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[key]
        totalcamname.append(extr.name)
    
    sortedtotalcamelist =  natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i # map each cam with a number

    tot_image_paths = sorted(glob.glob(source_path + "/*.png"))
    img_dict = {}
    for i in  tot_image_paths:
        int_matches = re.findall('[0-9]+', i)
        timestamp, cam_id = int(int_matches[-2]), int(int_matches[-1])
        if cam_id not in img_dict:
            img_dict[cam_id] = []
        img_dict[cam_id].append((i, timestamp))
    assert len(set([len(img_dict[l]) for l in img_dict.keys()])) == 1
            
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        id = int(extr.name[3:5])
        cam_name = os.path.basename(extr.name).split(".")[0]

        cam_image_paths = img_dict[id]
        
        for image_path, timestamp in cam_image_paths:
            if timestamp < startime or (endtime != -1 and timestamp >= endtime):
                continue
            
            cxr = ((intr.params[2] )/  width - 0.5) 
            cyr = ((intr.params[3] ) / height - 0.5) 
            # cxr = 0
            # cyr = 0

            K = np.eye(3)
            K[0, 0] = focal_length_x #* 0.5
            K[0, 2] = intr.params[2] #* 0.5 
            K[1, 1] = focal_length_y #* 0.5
            K[1, 2] = intr.params[3] #* 0.5
            
            cam_info = CameraInfo2(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image_path=image_path, 
                                   image_name=os.path.basename(image_path), width=width, height=height, near=near, far=far, timestamp=(timestamp-startime), 
                                   pose=None, hpdirecitons=None,  cxr=cxr, cyr=cyr)
   
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    
    return cam_infos


def readN3VCameras(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, endtime=-1):
    cam_infos = []

    # pose in llff. pipeline by hypereel 
    originnumpy = os.path.join(images_folder, "poses_bounds.npy")
    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        
        # bounds = poses_bounds[:, -2:]
        # near = bounds.min() * 0.95
        # far = bounds.max() * 1.05
        
        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # 19, 3, 5

        H, W, focal = poses[0, :, -1]
        cx, cy = W / 2.0, H / 2.0

        K = np.eye(3)
        K[0, 0] = focal * W / W / 2.0
        K[0, 2] = cx * W / W / 2.0
        K[1, 1] = focal * H / H / 2.0
        K[1, 2] = cy * H / H / 2.0
      
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)
    
    sortedtotalcamelist = natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in  range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i # map each cam with a number
     
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        cam_img_dir = os.path.join(images_folder, extr.name[:-4])
        tot_image_paths = sorted(glob.glob(cam_img_dir + "/*.png"), key=lambda x: int(os.path.basename(x)[:-4]))
        
        for j, image_path in enumerate(tot_image_paths):
            
            if j < startime or (endtime != -1 and j >= endtime):
                continue
                
            cam_info = CameraInfo2(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image_path=image_path, image_name=os.path.basename(image_path), width=width, height=height, near=near, far=far, timestamp=(j-startime), pose=None, hpdirecitons=None, cxr=0.0, cyr=0.0)
            cam_infos.append(cam_info)
            
    sys.stdout.write('\n')
    
    return cam_infos


def fetchPly_wt(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    times = np.vstack([vertices['t']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T6
    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return BasicPointCloud_pc(points=positions, colors=colors)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info


def readCamerasFromTransforms(path, transformsfile, args, near=0.2, far=300, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            print( frame["file_path"] )
            cam_name = os.path.join(path, frame["file_path"] + extension)
            
            fx = frame["fx"]
            fy = frame["fy"]
            cx = frame["cx"]
            cy = frame["cy"]
            timestamp = frame["timestamp"]
            
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem + extension
            image = Image.open(image_path)

            FovX = focal2fov(fx, image.size[0])
            FovY = focal2fov(fy, image.size[1])
            cxr = ((cx)/  image.size[0] - 0.5) 
            cyr = ((cy) / image.size[1] - 0.5) 

            cam_info = CameraInfo2(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image_path=image_path, image_name=image_name, 
                                   width=image.size[0], height=image.size[1], near=near, far=far, timestamp=timestamp, pose=None, hpdirecitons=None,  cxr=cxr, cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfoTechnicolor(path, images, eval, args):
    colmap_path = os.path.join(path, "colmap_" + str(int(args.start_timestamp)))

    try:
        cameras_extrinsic_file = os.path.join(colmap_path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(colmap_path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    near = 0.01
    far = 100
    
    cam_infos_unsorted = readColmapCamerasTechnicolor(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, source_path=path, 
                                                      near=near, far=far, args=args, startime=args.start_timestamp, endtime=args.end_timestamp)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [_ for _ in cam_infos if "_10.png" not in _.image_name]
        test_cam_infos = [_ for _ in cam_infos if "_10.png" in _.image_name]
        uniquecheck = []
        for cam_info in test_cam_infos:
            if cam_info.uid not in uniquecheck:
                uniquecheck.append(cam_info.uid)
        assert len(uniquecheck) == 1 
        
        sanitycheck = []
        for cam_info in train_cam_infos:
            if cam_info.uid not in sanitycheck:
                sanitycheck.append(cam_info.uid)
        for testname in uniquecheck:
            assert testname not in sanitycheck
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos[:4]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # normalization
    for c in train_cam_infos:
        c.T = c.T / nerf_normalization['radius']
    for c in test_cam_infos:
        c.T = c.T / nerf_normalization['radius']

    ply_path = os.path.join(colmap_path, "sparse", "0", "points3D.ply")
    bin_path = os.path.join(colmap_path, "sparse", "0", "points3D.bin")
    txt_path = os.path.join(colmap_path, "sparse", "0", "points3D.txt")
    
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        xyz = xyz / nerf_normalization['radius']
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    nerf_normalization['radius'] = 1

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info


def readColmapSceneInfoNeural3DVideo(path, images, eval, args):
    colmap_path = os.path.join(path, "colmap_" + str(int(args.start_timestamp)))
    try:
        cameras_extrinsic_file = os.path.join(colmap_path, "sparse", "0", "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_path, "sparse", "0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(colmap_path, "sparse", "0", "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_path, "sparse", "0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    near = 0.01
    far = 300

    cam_infos_unsorted = readN3VCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=path, near=near, far=far, 
                                        startime=args.start_timestamp, endtime=args.end_timestamp)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
     
    # if eval:
    train_cam_infos = [_ for _ in cam_infos if "cam00" not in _.image_path]
    test_cam_infos = [_ for _ in cam_infos if "cam00" in _.image_path]
    uniquecheck = []
    for cam_info in test_cam_infos:
        if cam_info.image_path not in uniquecheck:
            uniquecheck.append(cam_info.image_path)
    # assert len(uniquecheck) == len(cam_extrinsics) - 1
    
    sanitycheck = []
    for cam_info in train_cam_infos:
        if cam_info.image_path not in sanitycheck:
            sanitycheck.append(cam_info.image_path)
    for testname in uniquecheck:
        assert testname not in sanitycheck

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(colmap_path, "sparse", "0", "points3D.ply")
    bin_path = os.path.join(colmap_path, "sparse", "0", "points3D.bin")
    txt_path = os.path.join(colmap_path, "sparse", "0", "points3D.txt")
    
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Technicolor": readColmapSceneInfoTechnicolor,
    "Neural3DVideo": readColmapSceneInfoNeural3DVideo,
}