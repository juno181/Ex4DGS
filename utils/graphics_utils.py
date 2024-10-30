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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    times: np.array

class BasicPointCloud_pc(NamedTuple):
    points : np.array
    colors : np.array
    # normals : np.array
    # times: np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return (Rt).astype(np.float32)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return (Rt).astype(np.float32)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrixCV(znear, zfar, fovX, fovY, cx=0.0, cy=0.0):
    '''
    cx and cy range is -0.5 to 0.5  
    we use cx cy range -0.5 * 0.5
    
    '''
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # Adjust for off-center projection
    deltax = (2 * tanHalfFovX * znear) * cx  # 
    deltay = (2 * tanHalfFovY * znear) * cy  #
    
    left += deltax
    right += deltax
    top += deltay
    bottom += deltay

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
#    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 2] = z_sign * (zfar+znear) / (zfar - znear)

    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0


def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def unproject(depth, image_width, image_height, full_proj_transform):
    # warning this is not a batch operation
    x_pix = (torch.arange(0, image_width, device=depth.device) * 2  + 1.0) / image_width  - 1 
    y_pix = (torch.arange(0, image_height, device=depth.device) * 2 + 1.0) / image_height - 1
    xx, yy = torch.meshgrid(x_pix, 
                            y_pix)
    xx = xx.transpose(0, 1).flatten()
    yy = yy.transpose(0, 1).flatten()
        
    cam_hom = torch.stack([xx, yy, torch.ones_like(xx)* 0.998, torch.ones_like(xx)], dim=-1)
        
    cam_hom = cam_hom * depth.view(-1, 1)
    # cam_hom = torch.matmul(K_inv, cam_hom.view(-1, 4, 1))
    world_points = torch.matmul(cam_hom.view(-1, 1, 4), full_proj_transform.inverse())
    world_points = world_points.squeeze()[..., :3]

    return world_points.view(*depth.shape[1:], 3)


def project(inp_points, image_width, image_height, full_proj_transform):
    # warning this is not a batch operation
    points = inp_points.view(-1, 3)
    
    points_hom = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    points_hom = torch.matmul(points_hom.view(-1, 1, 4), full_proj_transform)
    points_hom = points_hom / points_hom[..., -2:-1]
    uv = points_hom.squeeze()[..., :3]
        
    uv[:, 0] = ((uv[:, 0] + 1) * image_width ) * 0.5
    uv[:, 1] = ((uv[:, 1] + 1) * image_height) * 0.5
    uv = uv[:, :2]
    
    f_x = torch.arange(0, image_width, device=points.device)
    f_y = torch.arange(0, image_height, device=points.device)
    grid_x, grid_y = torch.meshgrid(f_x, f_y)
    img_coords = torch.stack([grid_x.transpose(0, 1), grid_y.transpose(0, 1)], dim=-1)
        
    return uv.reshape(*inp_points.shape[:-1], 2) - img_coords


def project_depth(inp_points, full_proj_transform):
    # warning this is not a batch operation
    points = inp_points.view(-1, 3)
    
    points_hom = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    points_hom = torch.matmul(points_hom.view(-1, 1, 4), full_proj_transform)

    return points_hom[..., -2:-1]


def get_warped_image(ref_img, optflow):
    f_x = torch.arange(0, ref_img.shape[2], device="cuda")
    f_y = torch.arange(0, ref_img.shape[1], device="cuda")
    grid_x, grid_y = torch.meshgrid(f_x, f_y)
    grid = torch.stack([grid_x.transpose(0, 1), grid_y.transpose(0, 1)], dim=-1)
    
    warp_grid = grid + optflow
    warp_grid[..., 0] = warp_grid[..., 0] / ref_img.shape[2] * 2 - 1
    warp_grid[..., 1] = warp_grid[..., 1] / ref_img.shape[1] * 2 - 1
    
    warp_img = torch.nn.functional.grid_sample(ref_img.unsqueeze(0), warp_grid.unsqueeze(0))
    
    return warp_img.squeeze(0)
    
    
def get_mostsim_neiidxs(src_img, tgt_img, inp_idx, win_size=3, thres=0.1):
    H, W = src_img.shape[1], src_img.shape[2]
    src_img = src_img.mean(dim=0, keepdim=True)
    tgt_img = tgt_img.mean(dim=0, keepdim=True)
    
    unfolded_src = torch.nn.functional.unfold(src_img.view(1, 1, H, W), kernel_size=win_size, stride=1, padding=win_size//2)
    idxs = torch.arange(H * W, device=src_img.device, dtype=torch.float).view(1, H, W)
    unfolded_idxs = torch.nn.functional.unfold(idxs.view(1, 1, H, W), kernel_size=win_size, stride=1, padding=win_size//2)
    # l1 errors
    # unfolded_src[:, (win_size * win_size) // 2, :] = 0
    unfolded_l1 = (unfolded_src.view(win_size * win_size, H, W) - tgt_img).abs()
    min_mask = unfolded_l1 < thres
    # idx_diff = (min_idx // win_size - win_size // 2) * W + (min_idx % win_size - win_size // 2)
    # diff = idxs + idx_diff
    
    idx_from = inp_idx.repeat(win_size*win_size, 1, 1)
    idx_to = torch.index_select(inp_idx.view(-1), 0, unfolded_idxs.flatten()[min_mask.flatten()].long())

    return idx_from.flatten()[min_mask.flatten()], idx_to
    

def get_avg_depth(depth, win_size=3):
    # dynamic to dynamic
    H, W = depth.shape[1], depth.shape[2]
    weight = torch.ones_like(depth)
    
    unfolded_src = torch.nn.functional.unfold(depth.view(1, 1, H, W), kernel_size=win_size, stride=1, padding=win_size//2)
    unfolded_weight = torch.nn.functional.unfold(weight.view(1, 1, H, W), kernel_size=win_size, stride=1, padding=win_size//2)
    
    avg_depth = (unfolded_src.view(win_size * win_size, H, W).sum(dim=0) / unfolded_weight.view(win_size * win_size, H, W).sum(dim=0))
    
    return avg_depth.unsqueeze(0)
