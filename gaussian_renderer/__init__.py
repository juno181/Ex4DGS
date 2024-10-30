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
import math

import torch
from torch.nn import functional as F
from diff_gaussian_rasterization_df import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, timestamp=None, scaling_modifier=1.0, override_color=None, subpixel_offset=None, mode=0, training=False, near=0.2, far=100.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    timestamp = timestamp if timestamp is not None else viewpoint_camera.timestamp
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz_at_t(timestamp, mode=mode), dtype=pc._xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
        
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=pc.kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        min_depth=near,
        max_depth=far,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz_at_t(timestamp, mode=mode, training=training)
    means2D = screenspace_points
    opacity = pc.get_opacity_at_t(timestamp, mode=mode, training=training)
    
    flow = torch.zeros_like(means3D, requires_grad=True) + 0
    try:
        flow.retain_grad()
    except:
        pass
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # warning: this may not work correctly with the video model
        cov3D_precomp = pc.get_covariance_at_t(scaling_modifier, mode=mode)
    else:
        scales = pc.get_scaling(mode=mode)
        rotations = pc.get_rotation_at_t(timestamp, mode=mode)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features(motion=mode).transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz_at_t(timestamp, motion=mode) - viewpoint_camera.camera_center.repeat(pc.get_features(mode=mode).shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features(mode=mode)
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, out_flow, acc, idxs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dir3D = flow,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    torch.cuda.synchronize()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
            "render": rendered_image,
            "depth": rendered_depth,
            "opticalflow": out_flow,
            "acc": acc,
            "viewspace_points": screenspace_points,
            "viewspace_l1points": flow,
            "dominent_idxs": idxs,
            "visibility_filter" : radii > 0,
            "radii": radii
            }
