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

import torch
from torch import nn
import numpy as np
from kornia import create_meshgrid
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCV, pix2ndc, ndc2pix, fov2focal
from utils.general_utils import PILtoTorch
from PIL import Image


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


class Cameravideo():
    def __init__(self, colmap_id, R, T, FoVx, FoVy, gt_alpha_mask, image,
                 image_name, image_path, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", 
                 near=0.01, far=100.0, timestamp=0.0, 
                 rayo=None, rayd=None, rays=None, cxr=0.0, cyr=0.0, resolution=(1., 1.),
                 opticalflow_path=None, depth_path=None, im_scale=1.0
                 ):
        super(Cameravideo, self).__init__()
        
        self.uid = uid
        self.colmap_id = colmap_id
        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.timestamp = timestamp
        self.fisheyemapper = None
        self.resolution = resolution
        self.image_path = image_path
        self.opticalflow_path = opticalflow_path
        self.depth_path = depth_path
        self.im_scale = im_scale

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.zfar = far
        self.znear = near
        self.trans = trans
        self.scale = scale

        # w2c 
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale), dtype=torch.float32).transpose(0, 1).cuda()
        if cyr != 0.0 :
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if rayd is not None:
            projectinverse = self.projection_matrix.T.inverse()
            camera2wold = self.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
            
            ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

            projected = ndccamera @ projectinverse.T 
            diretioninlocal = projected / projected[:,:,3:] # 

            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)
            
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
            self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)                                                                          #rayd.permute(2, 0, 1).unsqueeze(0)
        else :
            self.rayo = None
            self.rayd = None
            
        self.image = image
        
        
WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    

def loadCamVideo(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)),  round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 
     
    if camerapose is not None:
        rays_o, rays_d = 1, cameradirect
    else :
        rays_o = None
        rays_d = None
    return Cameravideo(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, gt_alpha_mask=None, 
                  image=cam_info.image,
                  image_name=cam_info.image_name, image_path=cam_info.image_path, uid=id, data_device=args.data_device, 
                  near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, 
                  rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr, resolution=resolution)
    

def loadCamVideoss(args, id, cam_info, resolution_scale, nogt=False):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)),  round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

    resolution = (int(orig_w / 2), int(orig_h / 2))
    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 

    im_scale = 1
    # load gt image 
    if nogt == False :
        if "01_Welder" in args.source_path:
            if "camera_0009" in cam_info.image_name:
                im_scale = 1.15
                
        if "12_Cave" in args.source_path:
            if "camera_0009" in cam_info.image_name:
                im_scale = 1.15
        
        if "04_Truck" in args.source_path:
            if "camera_0008" in cam_info.image_name:
                im_scale = 1.2
        
        if camerapose is not None:
            rays_o, rays_d = 1, cameradirect
        else :
            rays_o = None
            rays_d = None
            
        return Cameravideo(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, gt_alpha_mask=None, image=cam_info.image if not args.lazy_loader else None,
                    image_name=cam_info.image_name, image_path=cam_info.image_path, uid=id, data_device=args.data_device, 
                    near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, 
                    rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr, resolution=resolution, im_scale=im_scale)
    else:
        if camerapose is not None:
            rays_o, rays_d = 1, cameradirect
        else :
            rays_o = None
            rays_d = None
            
        return Cameravideo(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, gt_alpha_mask=None, image=cam_info.image if not args.lazy_loader else None,
                    image_name=cam_info.image_name, image_path=cam_info.image_path, uid=id, data_device=args.data_device, 
                    near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, 
                    rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr, resolution=resolution)
        

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def cameraList_from_camInfosVideo(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCamVideo(args, id, c, resolution_scale))

    return camera_list


def cameraList_from_camInfosVideo2(cam_infos, resolution_scale, args, ss=False):
    camera_list = []

    if not ss: #
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCamVideo(args, id, c, resolution_scale))
    else:
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCamVideoss(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
