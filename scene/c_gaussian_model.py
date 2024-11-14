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
import os

import numpy as np
import torch
from torch import nn
from plyfile import PlyData, PlyElement

from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.graphics_utils import BasicPointCloud, BasicPointCloud_pc
from utils.general_utils import strip_symmetric, build_scaling_rotation, MultipleOptimizer
from utils.system_utils import mkdir_p
from utils.interpolations import linear_interp_uniiterval, quat_slerp_interp_uniiterval, time_bigaussian, pchip_interpolate, cube_interpolate, quad_diff_interpolate


class CGaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, duration : int, interval : int, time_pad : int = 1, interp_type='linear', rot_interp_type='slerp', time_pad_type=0, var_pad=3, kernel_size=0.1, **kwargs):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0).cuda()
        self._opacity = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0).cuda()
        self.min_radii2D = torch.empty(0).cuda()
        self.xyz_gradient_accum = torch.empty(0).cuda()
        self.denom = torch.empty(0).cuda()
        self.xyz_error_accum = torch.empty(0).cuda()
        self.xyz_error_min = torch.empty(0).cuda()
        self.xyz_error_min_timestamp = torch.empty(0).cuda()
        self.xyz_ssim_error_accum = torch.empty(0).cuda()
        self.error_denom = torch.empty(0).cuda()
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.var_pad = var_pad
        self.setup_functions()
        
        # properties for motion handling
        self.kernel_size = kernel_size
        self.duration = max(duration, 1) # prevent zero div
        self.interval = interval
        self.time_pad = time_pad
        self.time_pad_type = time_pad_type
        self.time_shift = time_pad 
        self.keyframe_num = 0
        self._xyz_disp = torch.empty(0).cuda()
        self._xyz_motion = torch.empty(0).cuda() # [b, t, 3]
        self._features_dc_motion = torch.empty(0).cuda()
        self._features_rest_motion = torch.empty(0).cuda()
        self._scaling_motion = torch.empty(0).cuda()
        self._opacity_motion = torch.empty(0).cuda() # [b, 1]
        self._opacity_duration_center = torch.empty(0).cuda() # [b, 1]
        self._opacity_duration_var = torch.empty(0).cuda() # [b, 1]
        self.opacity_degree = 2
        self._rotation_motion = torch.empty(0).cuda() # [b, t, 4]
        self.motion_max_radii2D = torch.empty(0).cuda()
        self.motion_min_radii2D = torch.empty(0).cuda()
        self.motion_xyz_gradient_accum = torch.empty(0).cuda()
        self.motion_denom = torch.empty(0).cuda()
        self.motion_xyz_error_min = torch.empty(0).cuda()
        self.motion_xyz_error_mean = torch.empty(0).cuda()
        self.motion_xyz_error_min_timestamp = torch.empty(0).cuda()
        self.motion_xyz_ssim_error_accum = torch.empty(0).cuda()
        self.motion_error_denom = torch.empty(0).cuda()
        
        self.interp_type = interp_type
        # set interpolations
        if interp_type == "linear":
            self.motion_degree = 1 # points
            def interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    return linear_interp_uniiterval(torch.gather(y, 1, t_idx.repeat(1, 1, 3)), 
                                                torch.gather(y, 1, t_idx.repeat(1, 1, 3) + 1), delta_t).squeeze(1)
                else:
                    return linear_interp_uniiterval(y[...,t_idx, :], y[...,t_idx+1, :], delta_t).squeeze(1)
        elif interp_type == "cube":
            self.motion_degree = 1 # points, diff
            def interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    y0 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) - 1)
                    y1 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree))
                    y2 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 1)
                    y3 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 2)
                    return cube_interpolate(y0[..., :3], y1[...,:3],y2[..., :3], y3[...,:3], delta_t).squeeze(1)
                else:
                    return cube_interpolate(y[:, t_idx-1, :3], y[:, t_idx, :3],y[:, t_idx+1, :3], y[:, t_idx+2, :3], delta_t).squeeze(1)
            self.time_shift += interval
            
        elif interp_type == "cubic_diff":
            self.motion_degree = 1 # points, diff
            def interpolation(y, y_d, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    y1 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree))
                    y2 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 1)
                    y_d1 = torch.gather(y_d, 1, t_idx.repeat(1, 1, 3 * self.motion_degree))
                    y_d2 = torch.gather(y_d, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 1)
                    return quad_diff_interpolate(y1, y2, y_d1, y_d2, delta_t).squeeze(1)
                else:
                    return quad_diff_interpolate(y[:,t_idx], y[:,t_idx+1], y_d[:,t_idx], y_d[:,t_idx+1], delta_t).squeeze(1)
                
        elif interp_type == "pchip":
            self.motion_degree = 1 # points
            def interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    y0 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) - 1)
                    y1 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree))
                    y2 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 1)
                    y3 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 2)
                    return pchip_interpolate(y0[..., :3], y1[...,:3],y2[..., :3], y3[...,:3], delta_t).squeeze(1)
                else:
                    return pchip_interpolate(y[...,t_idx-1, :3], y[...,t_idx, :3],y[...,t_idx+1, :3], y[...,t_idx+2, :3], delta_t).squeeze(1)
            self.time_shift += interval
        else:
            raise NotImplementedError
        
        self.interpolator = interpolation
        
        if rot_interp_type == 'lerp':
            def interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    return linear_interp_uniiterval(torch.gather(y, 1, t_idx.repeat(1, 1, 4)), 
                                                torch.gather(y, 1, t_idx.repeat(1, 1, 4) + 1), delta_t).squeeze(1)
                else:
                    return linear_interp_uniiterval(y[...,t_idx, :], y[...,t_idx+1, :], delta_t).squeeze(1)
        elif rot_interp_type == 'slerp':
            # rotation interpolation
            def quat_interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    return quat_slerp_interp_uniiterval(torch.gather(y, 1, t_idx.repeat(1, 1, 4)), 
                                                        torch.gather(y, 1, t_idx.repeat(1, 1, 4) + 1), delta_t).squeeze(1)
                else:
                    return quat_slerp_interp_uniiterval(y[...,t_idx, :], y[...,t_idx+1, :], delta_t).squeeze(1)
        else:
            raise NotImplementedError
        self.quat_interpolator = quat_interpolation
        self.error_dict = {}
        
    def get_xyz_at_t(self, t, mode=0, training=True):
        assert t >= -self.time_shift and t <= self.duration+self.time_shift
        if self._xyz_motion.shape[0] == 0 or mode == 1:
            return self.get_static_xyz_at_t(t)
        elif mode == 2:
            return self.get_dynamic_xyz_at_t(t)
        return torch.cat([self.get_static_xyz_at_t(t, training), self.get_dynamic_xyz_at_t(t)], dim=0).contiguous()
       
    def get_static_xyz_at_t(self, t, training=True):
        # static points 
        return self._xyz + self._xyz_disp * t / self.duration
    
    def get_dynamic_xyz_at_t(self, t):
        # dynamic points
        t = t + self.time_shift
        t_idx = t // self.interval
        delta_t = (t % self.interval) / self.interval
        
        if type(t) == torch.Tensor:
            t_idx = t_idx.view(-1, 1, 1).long()
            delta_t = delta_t.view(-1, 1, 1)
        else:
            t_idx = int(t_idx)
        return self.interpolator(self._xyz_motion, t_idx, delta_t)
       
    def get_rotation_at_t(self, t, mode=0):
        assert t >= -self.time_shift and t <= self.duration+self.time_shift
        if self._rotation_motion.shape[0] == 0 or mode == 1:
            return self._rotation
        elif mode == 2:
            return self.get_dynamic_rotation_at_t(t)
        return torch.cat([self._rotation, self.get_dynamic_rotation_at_t(t)], dim=0).contiguous()
    
    def get_dynamic_rotation_at_t(self, t):
        # dynamic points
        t = t + self.time_shift
        t_idx = t // self.interval
        delta_t = (t % self.interval) / self.interval
        
        if type(t) == torch.Tensor:
            t_idx = t_idx.view(-1, 1, 1).long()
            delta_t = delta_t.view(-1, 1, 1)
        else:
            t_idx = int(t_idx)
            
        return self.quat_interpolator(self._rotation_motion, t_idx, delta_t)
        
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.min_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.xyz_error_accum,
            self.xyz_error_min,
            self.xyz_error_min_timestamp,
            self.xyz_ssim_error_accum,
            self.error_denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            
            self._xyz_disp,
            self.duration,
            self.interval,
            self.time_shift,
            self.keyframe_num,
            self._xyz_motion,
            self._features_dc_motion,
            self._features_rest_motion,
            self._scaling_motion,
            self._opacity_motion,
            self._opacity_duration_center,
            self._opacity_duration_var,
            self._rotation_motion,
            self.motion_max_radii2D,
            self.motion_min_radii2D,
            self.motion_xyz_gradient_accum,
            self.motion_denom,
            self.motion_xyz_error_min,
            self.motion_xyz_error_mean,
            self.motion_xyz_error_min_timestamp,
            self.motion_xyz_ssim_error_accum,
            self.motion_error_denom,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self.min_radii2D, 
        xyz_gradient_accum, 
        denom,
        xyz_error_accum, 
        xyz_error_min, 
        xyz_error_min_timestamp,
        xyz_ssim_error_accum, 
        error_denom,
        opt_dict, 
        self.spatial_lr_scale,
        
        self._xyz_disp,
        self.duration,
        self.interval,
        self.time_shift,
        self.keyframe_num,
        self._xyz_motion,
        self._features_dc_motion,
        self._features_rest_motion,
        self._scaling_motion,
        self._opacity_motion,
        self._opacity_duration_center,
        self._opacity_duration_var,
        self._rotation_motion,
        self.motion_max_radii2D, 
        self.motion_min_radii2D, 
        motion_xyz_gradient_accum, 
        motion_denom,
        motion_xyz_error_min, 
        motion_xyz_error_mean, 
        motion_xyz_error_min_timestamp, 
        motion_xyz_ssim_error_accum, 
        motion_error_denom
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.motion_xyz_gradient_accum = motion_xyz_gradient_accum
        self.motion_denom = motion_denom
        self.xyz_error_accum = xyz_error_accum
        self.xyz_error_min = xyz_error_min
        self.xyz_error_min_timestamp = xyz_error_min_timestamp
        self.xyz_ssim_error_accum = xyz_ssim_error_accum
        self.error_denom = error_denom
        self.motion_xyz_error_min = motion_xyz_error_min
        self.motion_xyz_error_mean = motion_xyz_error_mean
        self.motion_xyz_error_min_timestamp = motion_xyz_error_min_timestamp
        self.motion_xyz_ssim_error_accum = motion_xyz_ssim_error_accum
        self.motion_error_denom = motion_error_denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_static_scaling(self):
        return  self.scaling_activation(self._scaling)

    @property
    def get_motion_scaling(self):
        return self.scaling_activation(self._scaling_motion)
    
    def get_scaling(self, mode=0):
        if self._scaling_motion.shape[0] == 0 or mode == 1:
            return self.get_static_scaling
        elif mode == 2:
            return self.get_motion_scaling
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_motion], dim=0)) 

    def get_features(self, mode=0):
        if self._features_dc_motion.shape[0] == 0 or mode == 1:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)
        elif mode == 2:
            features_dc_motion = self._features_dc_motion
            features_rest_motion = self._features_rest_motion
            return torch.cat((features_dc_motion, features_rest_motion), dim=1)
        
        features_dc = self._features_dc
        features_rest = self._features_rest
        features_dc_motion = self._features_dc_motion
        features_rest_motion = self._features_rest_motion
        return torch.cat((
                torch.cat((features_dc, features_rest), dim=1),
                torch.cat((features_dc_motion, features_rest_motion), dim=1)), dim=0).contiguous()
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_motion_opacity(self):
        return self.opacity_activation(self._opacity_motion)
    
    def get_motion_opacity_at_t(self, t, training=False):
        t = (t + self.time_shift) / self.interval
        return (time_bigaussian(self._opacity_duration_center, self._opacity_duration_var, t, training=training, var_min=self.var_pad/self.interval) \
                            * self.get_motion_opacity)
       
    def get_opacity_at_t(self, t, mode=0, training=False):
        assert t >= -self.time_shift and t <= self.duration+self.time_shift
        if self._opacity_motion.shape[0] == 0 or mode == 1:
            return self.get_opacity
        elif mode == 2:
            return self.get_motion_opacity_at_t(t, training=training)
        return torch.cat([self.get_opacity, 
                          self.get_motion_opacity_at_t(t, training=training)], dim=0).contiguous()
    
    def get_covariance_at_t(self, t, scaling_modifier = 1, mode=0):
        # pad not required here
        return self.covariance_activation(self.get_scaling(mode=mode), scaling_modifier, self.get_rotation_at_t(t, mode=mode)).contiguous()

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0 # why?

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.ones((self._xyz.shape[0]), device="cuda") * 1000
        self._xyz_disp = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_error_min = torch.ones((self._xyz.shape[0], 1), device="cuda") * 1000
        self.xyz_error_min_timestamp = torch.ones((self._xyz.shape[0], 1), device="cuda") * -1
        self.xyz_ssim_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.error_denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        self.motion_xyz_gradient_accum = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_denom = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_xyz_error_min = torch.ones((self._xyz_motion.shape[0], 1), device="cuda") * 1000
        self.motion_xyz_error_mean = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_xyz_error_min_timestamp = torch.ones((self._xyz_motion.shape[0], 1), device="cuda") * -1
        self.motion_xyz_ssim_error_accum = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_error_denom = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        #######################################################################################################
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._xyz_disp], 'lr': training_args.disp_lr, "name": "xyz_disp"},
            
            {'params': [self._xyz_motion], 'lr': training_args.dynamic_position_lr_init * self.spatial_lr_scale, "name": "motion_xyz"},
            {'params': [self._features_dc_motion], 'lr': training_args.feature_motion_lr, "name": "motion_f_dc"}, # use the same lr as static points
            {'params': [self._features_rest_motion], 'lr': training_args.feature_motion_lr / 20.0, "name": "motion_f_rest"},
            {'params': [self._scaling_motion], 'lr': training_args.scaling_lr, "name": "motion_scaling"},
            {'params': [self._opacity_motion], 'lr': training_args.opacity_motion_lr, "name": "motion_opacity"},
            {'params': [self._opacity_duration_center], 'lr': training_args.opacity_motion_center_lr, "name": "motion_opacity_center"},
            {'params': [self._opacity_duration_var], 'lr': training_args.opacity_motion_var_lr, "name": "motion_opacity_var"},
            {'params': [self._rotation_motion], 'lr': training_args.rotation_motion_lr, "name": "motion_rotation"}
        ]

        self.optimizer = torch.optim.RAdam(l, lr=0.001)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.xyz_motion_scheduler_args = get_expon_lr_func(lr_init=training_args.dynamic_position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.dynamic_position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.dynamic_position_lr_delay_mult,
                                                    max_steps=training_args.dynamic_position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            elif param_group["name"] == "motion_xyz":
                lr = self.xyz_motion_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_static_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._xyz_disp.shape[1]):
            l.append('xyz_disp_{}'.format(i))
            
        return l
    
    def construct_list_of_dynamic_attributes(self):
        l2 = []
        # attributes for dynamic points.
        for i in range(self._xyz_motion.shape[1]):
            for j in range(self._xyz_motion.shape[2]):
                l2.append('motion_xyz_{}_{}'.format(i, j))
        for i in range(self._features_dc_motion.shape[1]*self._features_dc_motion.shape[2]):
            l2.append('motion_f_dc_{}'.format(i))
        for i in range(self._features_rest_motion.shape[1]*self._features_rest_motion.shape[2]):
            l2.append('motion_f_rest_{}'.format(i))
        for i in range(self._scaling_motion.shape[1]):
            l2.append('motion_scale_{}'.format(i))
        l2.append('motion_opacity')
        for i in range(self._opacity_duration_center.shape[1]):
            l2.append('motion_opacity_c_{}'.format(i))
        for i in range(self._opacity_duration_var.shape[1]):
            l2.append('motion_opacity_v_{}'.format(i))
        for i in range(self._rotation_motion.shape[1]):
            for j in range(self._rotation_motion.shape[2]):
                l2.append('motion_rot_{}_{}'.format(i, j))
        
        return l2

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        xyz_disp = self._xyz_disp.detach().cpu().numpy()
            
        static_dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_static_attributes()]

        static_elements = np.empty(xyz.shape[0], dtype=static_dtype_full)
        static_attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, xyz_disp), axis=1)
        static_elements[:] = list(map(tuple, static_attributes))
        s_el = PlyElement.describe(static_elements, 'vertex')
        PlyData([s_el]).write(path)
        
        xyz_motion = self._xyz_motion.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc_motion = self._features_dc_motion.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_motion = self._features_rest_motion.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        scaling_motion = self._scaling_motion.detach().cpu().numpy()
        opacity_motion = self._opacity_motion.detach().cpu().numpy()
        opacity_duration_center = self._opacity_duration_center.detach().flatten(start_dim=1).cpu().numpy()
        opacity_duration_var = self._opacity_duration_var.detach().flatten(start_dim=1).cpu().numpy()
        rotation_motion = self._rotation_motion.detach().flatten(start_dim=1).contiguous().cpu().numpy()

        dynamic_dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_dynamic_attributes()]
        dynamic_elements = np.empty(xyz_motion.shape[0], dtype=dynamic_dtype_full)
        dynamic_attributes = np.concatenate((xyz_motion, f_dc_motion, f_rest_motion, scaling_motion, opacity_motion, opacity_duration_center, opacity_duration_var, rotation_motion), axis=1)
        dynamic_elements[:] = list(map(tuple, dynamic_attributes))
        d_el = PlyElement.describe(dynamic_elements, 'vertex')
        PlyData([d_el]).write(path.replace('point_cloud.ply', 'dynamic_point_cloud.ply'))
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.85))
        optimizable_tensors = self.replace_tensor_to_optimizer({"opacity": opacities_new})
        self._opacity = optimizable_tensors["opacity"]
        
        if self._opacity_motion.shape[0] == 0:
            return
        motion_opacities_new = inverse_sigmoid(torch.min(self.get_motion_opacity, torch.ones_like(self.get_motion_opacity)*0.95))
        optimizable_tensors = self.replace_tensor_to_optimizer({"motion_opacity": motion_opacities_new})
        self._opacity_motion = optimizable_tensors["motion_opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz_disp = np.zeros((xyz.shape[0], 3))
        xyz_disp[:, 0] = np.asarray(plydata.elements[0]["xyz_disp_0"])
        xyz_disp[:, 1] = np.asarray(plydata.elements[0]["xyz_disp_1"])
        xyz_disp[:, 2] = np.asarray(plydata.elements[0]["xyz_disp_2"])
        
        # dynamic points
        plydata_dynamic = PlyData.read(path.replace('point_cloud.ply', 'dynamic_point_cloud.ply'))
        motion_opacities = np.asarray(plydata_dynamic.elements[0]["motion_opacity"])[..., np.newaxis]
        num_points = motion_opacities.shape[0]
        self.keyframe_num = math.ceil((self.duration + self.time_shift + self.time_pad*2 + 1) / self.interval) + 1 + 4
        
        motion_xyz_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_xyz_")]
        motion_xyz_names = sorted(motion_xyz_names, key = lambda x: ( int(x.split('_')[-2]), int(x.split('_')[-1]) ))
        
        motion_xyz = np.zeros((num_points, self.keyframe_num * 3 * self.motion_degree))
        for idx, attr_name in enumerate(motion_xyz_names):
            motion_xyz[:, idx] = np.asarray(plydata_dynamic.elements[0][attr_name])
        motion_xyz = motion_xyz.reshape((num_points, self.keyframe_num, 3 * self.motion_degree))
        
        motion_features_dc = np.zeros((num_points, 3, 1))
        motion_features_dc[:, 0, 0] = np.asarray(plydata_dynamic.elements[0]["motion_f_dc_0"])
        motion_features_dc[:, 1, 0] = np.asarray(plydata_dynamic.elements[0]["motion_f_dc_1"])
        motion_features_dc[:, 2, 0] = np.asarray(plydata_dynamic.elements[0]["motion_f_dc_2"])
        
        motion_f_rest_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_f_rest_")]
        motion_f_rest_names = sorted(motion_f_rest_names, key = lambda x: int(x.split('_')[-1]))
        assert len(motion_f_rest_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        motion_features_extra = np.zeros((num_points, len(motion_f_rest_names)))
        for idx, attr_name in enumerate(motion_f_rest_names):
            motion_features_extra[:, idx] = np.asarray(plydata_dynamic.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        motion_features_extra = motion_features_extra.reshape((motion_features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        motion_scale_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_scale_")]
        motion_scale_names = sorted(motion_scale_names, key = lambda x: int(x.split('_')[-1]))
        motion_scales = np.zeros((num_points, 3))
        for idx, attr_name in enumerate(motion_scale_names):
            motion_scales[:, idx] = np.asarray(plydata_dynamic.elements[0][attr_name])
        
        motion_opacity_c_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_opacity_c_")]
        motion_opacity_c_names = sorted(motion_opacity_c_names, key = lambda x: int(x.split('_')[-1] ))
        motion_opacities_c = np.zeros((num_points, self.opacity_degree, 1))
        for idx, attr_name in enumerate(motion_opacity_c_names):
            motion_opacities_c[:, idx, 0] = np.asarray(plydata_dynamic.elements[0][attr_name])
        
        motion_opacity_v_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_opacity_v_")]
        motion_opacity_v_names = sorted(motion_opacity_v_names, key = lambda x: int(x.split('_')[-1] ))
        motion_opacities_v = np.zeros((num_points, self.opacity_degree, 1))
        for idx, attr_name in enumerate(motion_opacity_v_names):
            motion_opacities_v[:, idx, 0] = np.asarray(plydata_dynamic.elements[0][attr_name])
        
        motion_rot_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_rot_")]
        motion_rot_names = sorted(motion_rot_names, key = lambda x: ( int(x.split('_')[-2]), int(x.split('_')[-1]) ))
        motion_rots = np.zeros((num_points, self.keyframe_num * 4))
        for idx, attr_name in enumerate(motion_rot_names):
            motion_rots[:, idx] = np.asarray(plydata_dynamic.elements[0][attr_name])
        motion_rots = motion_rots.reshape((num_points, self.keyframe_num, 4))
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._xyz_disp = nn.Parameter(torch.tensor(xyz_disp, dtype=torch.float, device="cuda").requires_grad_(True))

        # dynamic points
        self._xyz_motion = nn.Parameter(torch.tensor(motion_xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc_motion = nn.Parameter(torch.tensor(motion_features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_motion = nn.Parameter(torch.tensor(motion_features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling_motion = nn.Parameter(torch.tensor(motion_scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_motion = nn.Parameter(torch.tensor(motion_opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_duration_center = nn.Parameter(torch.tensor(motion_opacities_c, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_duration_var = nn.Parameter(torch.tensor(motion_opacities_v, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation_motion = nn.Parameter(torch.tensor(motion_rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor_dict):
        optimizable_tensors = {}
        for name, tensor in tensor_dict.items():
            for group in self.optimizer.param_groups:
                if group["name"] == name:
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.zeros_like(tensor)
                        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                        del self.optimizer.state[group['params'][0]]
                        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                        self.optimizer.state[group['params'][0]] = stored_state

                        optimizable_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(group["params"][0].requires_grad_(True))
                        optimizable_tensors[group["name"]] = group["params"][0]
                
        return optimizable_tensors

    def _prune_optimizer(self, static_mask, dynamic_mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"].startswith("motion_"):
                mask = dynamic_mask
            else:
                mask = static_mask
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, static_mask, dynamic_mask):
        
        vaild_static_mask = ~static_mask
        if dynamic_mask.shape[0] == 0:
            dynamic_mask = torch.empty(0, dtype=torch.bool, device=static_mask.device)
        vaild_dynamic_mask = ~dynamic_mask
        
        optimizable_tensors = self._prune_optimizer(vaild_static_mask, vaild_dynamic_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._xyz_disp = optimizable_tensors["xyz_disp"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[vaild_static_mask]
        self.denom = self.denom[vaild_static_mask]
        self.xyz_error_accum = self.xyz_error_accum[vaild_static_mask]
        self.xyz_error_min = self.xyz_error_min[vaild_static_mask]
        self.xyz_error_min_timestamp = self.xyz_error_min_timestamp[vaild_static_mask]
        self.xyz_ssim_error_accum = self.xyz_ssim_error_accum[vaild_static_mask]
        self.error_denom = self.error_denom[vaild_static_mask]
        self.max_radii2D = self.max_radii2D[vaild_static_mask]
        self.min_radii2D = self.min_radii2D[vaild_static_mask]
        
        # dynamic points
        if vaild_dynamic_mask.shape[0] == 0:
            return
        
        self._xyz_motion = optimizable_tensors["motion_xyz"]
        self._features_dc_motion = optimizable_tensors["motion_f_dc"]
        self._features_rest_motion = optimizable_tensors["motion_f_rest"]
        self._scaling_motion = optimizable_tensors["motion_scaling"]
        self._opacity_motion = optimizable_tensors["motion_opacity"]
        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]
        self._rotation_motion = optimizable_tensors["motion_rotation"]
        
        self.motion_xyz_gradient_accum = self.motion_xyz_gradient_accum[vaild_dynamic_mask]
        self.motion_denom = self.motion_denom[vaild_dynamic_mask]
        self.motion_xyz_error_min = self.motion_xyz_error_min[vaild_dynamic_mask]
        self.motion_xyz_error_mean = self.motion_xyz_error_mean[vaild_dynamic_mask]
        self.motion_xyz_error_min_timestamp = self.motion_xyz_error_min_timestamp[vaild_dynamic_mask]
        self.motion_xyz_ssim_error_accum = self.motion_xyz_ssim_error_accum[vaild_dynamic_mask]
        self.motion_error_denom = self.motion_error_denom[vaild_dynamic_mask]
        self.motion_max_radii2D = self.motion_max_radii2D[vaild_dynamic_mask]
        self.motion_min_radii2D = self.motion_min_radii2D[vaild_dynamic_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # warning this may skip invaild paramgroups
            if not group["name"] in tensors_dict.keys():
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_xyz_disp,
                              new_xyz_motion, new_features_dc_motion, new_features_rest_motion, new_scaling_motion, new_opacity_motion, 
                              new_opacity_duration_center, new_opacity_duration_var, new_rotation_motion
                              ):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "xyz_disp" : new_xyz_disp,
        
        "motion_xyz" : new_xyz_motion,
        "motion_f_dc" : new_features_dc_motion,
        "motion_f_rest" : new_features_rest_motion,
        "motion_scaling" : new_scaling_motion,
        "motion_opacity" : new_opacity_motion,
        "motion_opacity_center" : new_opacity_duration_center,
        "motion_opacity_var" : new_opacity_duration_var,
        "motion_rotation" : new_rotation_motion
        }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._xyz_disp = optimizable_tensors["xyz_disp"]
        
        # dynamic points
        self._xyz_motion = optimizable_tensors["motion_xyz"]
        self._features_dc_motion = optimizable_tensors["motion_f_dc"]
        self._features_rest_motion = optimizable_tensors["motion_f_rest"]
        self._scaling_motion = optimizable_tensors["motion_scaling"]
        self._opacity_motion = optimizable_tensors["motion_opacity"]
        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]
        self._rotation_motion = optimizable_tensors["motion_rotation"]
        
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_ssim_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.error_denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.ones((self._xyz.shape[0]), device="cuda") * 1000
        
        self.motion_xyz_gradient_accum = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_denom = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_xyz_error_mean = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_xyz_ssim_error_accum = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_error_denom = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_max_radii2D = torch.zeros((self._xyz_motion.shape[0]), device="cuda")
        self.motion_min_radii2D = torch.ones((self._xyz_motion.shape[0]), device="cuda") * 1000

    def densification_postfix_onlystatic(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_xyz_disp,
                              ):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "xyz_disp" : new_xyz_disp,
        }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._xyz_disp = optimizable_tensors["xyz_disp"]
        
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_ssim_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.error_denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.ones((self._xyz.shape[0]), device="cuda") * 1000
        
    def densify_and_split(self, static_grads, dynamic_grads, dynamic_ssim_errors, grad_threshold, dynamic_grad_threshold, scene_extent, max_screen_size, max_dynamic_screen_size, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_static_grad = torch.zeros((n_init_points), device="cuda")
        padded_static_grad[:static_grads.shape[0]] = static_grads.squeeze()
        selected_static_pts_mask = torch.where(padded_static_grad >= grad_threshold, True, False)
        selected_static_pts_mask = torch.logical_and(selected_static_pts_mask,
                                              torch.max(self.get_static_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_static_scaling.max(dim=1).values > 0.1 * scene_extent
            selected_static_pts_mask = torch.logical_or(torch.logical_or(selected_static_pts_mask, big_points_vs), big_points_ws)
        
        stds = self.get_static_scaling[selected_static_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_static_pts_mask]).repeat(N,1,1)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_static_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_static_scaling[selected_static_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_static_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_static_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_static_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_static_pts_mask].repeat(N,1)
        new_xyz_disp = self._xyz_disp[selected_static_pts_mask].repeat(N,1)
        
        static_prune_filter = torch.cat((selected_static_pts_mask, torch.zeros(N * selected_static_pts_mask.sum(), device="cuda", dtype=bool)))
        
        new_xyz_error_min = torch.ones((selected_static_pts_mask.sum() * N, 1), device="cuda") * 1000
        new_xyz_error_min_timestamp = torch.ones((selected_static_pts_mask.sum() * N, 1), device="cuda") * -1
        
        self.xyz_error_min = torch.cat([self.xyz_error_min, new_xyz_error_min])
        self.xyz_error_min_timestamp = torch.cat([self.xyz_error_min_timestamp, new_xyz_error_min_timestamp])
            
        if self._xyz_motion.shape[0] != 0:
            # dynamic points
            n_init_points = self._xyz_motion.shape[0]
            # Extract points that satisfy the gradient condition
            padded_dynamic_grad = torch.zeros((n_init_points), device="cuda")
            padded_dynamic_grad[:dynamic_grads.shape[0]] = dynamic_grads.squeeze()
            selected_dynamic_pts_mask = torch.where(padded_dynamic_grad >= dynamic_grad_threshold, True, False)
            selected_dynamic_pts_mask = torch.logical_and(selected_dynamic_pts_mask,
                                                torch.max(self.get_motion_scaling, dim=1).values > self.percent_dense*scene_extent)

            if max_dynamic_screen_size:
                big_points_vs = self.motion_max_radii2D > max_dynamic_screen_size
                big_points_ws = self.get_motion_scaling.max(dim=1).values > 0.1 * scene_extent
                selected_dynamic_pts_mask = torch.logical_or(torch.logical_or(selected_dynamic_pts_mask, big_points_vs), big_points_ws)
            
            # we select rot matrix where median point of opacity
            stds = self.get_motion_scaling[selected_dynamic_pts_mask].repeat(N,1)*2
            means = torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds).unsqueeze(1).repeat(1,self.keyframe_num,1).view(-1,3)
            rots_motion = build_rotation(self._rotation_motion[selected_dynamic_pts_mask].view(-1, 4)).reshape(-1,self.keyframe_num,3,3).repeat(N,1,1,1).view(-1,3,3)
            if self.motion_degree == 1:
                new_xyz_motion = torch.bmm(rots_motion, samples.unsqueeze(-1)).squeeze(-1).view(-1,self.keyframe_num,3) + self._xyz_motion[selected_dynamic_pts_mask].repeat(N,1,1)
            else:
                new_xyz_motion = torch.zeros_like(self._xyz_motion[selected_dynamic_pts_mask]).repeat(N,1,1)
                new_xyz_motion[..., :3] += torch.bmm(rots_motion, samples.unsqueeze(-1)).squeeze(-1).view(-1,self.keyframe_num,3) + self._xyz_motion[selected_dynamic_pts_mask][..., :3].repeat(N,1,1)

            new_scaling_motion = self.scaling_inverse_activation(self.get_motion_scaling[selected_dynamic_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation_motion = self._rotation_motion[selected_dynamic_pts_mask].repeat(N,1,1)
            new_features_dc_motion = self._features_dc_motion[selected_dynamic_pts_mask].repeat(N,1,1)
            new_features_rest_motion = self._features_rest_motion[selected_dynamic_pts_mask].repeat(N,1,1)
            new_opacity_motion = self._opacity_motion[selected_dynamic_pts_mask].repeat(N,1)
            new_opacity_duration_center = self._opacity_duration_center[selected_dynamic_pts_mask].repeat(N,1,1)
            new_opacity_duration_center_len = ((new_opacity_duration_center[:, 1] - new_opacity_duration_center[:, 0]).abs() / 3).clamp_min(2/self.interval)
            new_opacity_duration_center[:, 1] = new_opacity_duration_center[:, 1] + (new_opacity_duration_center_len * (torch.randn_like(new_opacity_duration_center[:, 1])))
            new_opacity_duration_center[:, 0] = new_opacity_duration_center[:, 0] + (new_opacity_duration_center_len * (torch.randn_like(new_opacity_duration_center[:, 0])))
            new_opacity_duration_center = new_opacity_duration_center.clamp((self.time_shift + 1)/self.interval, (self.time_shift+self.duration - 1)/self.interval)
            new_opacity_duration_var = torch.ones_like(self._opacity_duration_var[selected_dynamic_pts_mask].repeat(N,1,1)) * 2
            dynamic_prune_filter = torch.cat((selected_dynamic_pts_mask, torch.zeros(N * selected_dynamic_pts_mask.sum(), device="cuda", dtype=bool)))
            
            new_motion_xyz_error_min = torch.ones((selected_dynamic_pts_mask.sum() * N, 1), device="cuda") * 1000
            new_motion_xyz_error_min_timestamp = torch.ones((selected_dynamic_pts_mask.sum() * N, 1), device="cuda") * -1
            
            self.motion_xyz_error_min = torch.cat([self.motion_xyz_error_min, new_motion_xyz_error_min])
            self.motion_xyz_error_min_timestamp = torch.cat([self.motion_xyz_error_min_timestamp, new_motion_xyz_error_min_timestamp])
        
            self.densification_postfix(
                new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_xyz_disp, 
                new_xyz_motion, new_features_dc_motion, new_features_rest_motion, new_scaling_motion, new_opacity_motion, new_opacity_duration_center, new_opacity_duration_var, new_rotation_motion
                )
        else:
            self.densification_postfix_onlystatic(
                new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_xyz_disp, 
                )
            dynamic_prune_filter = torch.empty(0).cuda()
        
        self.prune_points(static_prune_filter, dynamic_prune_filter)

    def densify_and_clone(self, static_grads, dynamic_grads, dynamic_ssim_errors, 
                          grad_threshold, dynamic_grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_static_pts_mask = torch.where(torch.norm(static_grads, dim=-1) >= grad_threshold, True, False)
        selected_static_pts_mask = torch.logical_and(selected_static_pts_mask,
                                              torch.max(self.get_static_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_static_pts_mask]
        new_features_dc = self._features_dc[selected_static_pts_mask]
        new_features_rest = self._features_rest[selected_static_pts_mask]
        new_opacities = self._opacity[selected_static_pts_mask]
        new_scaling = self._scaling[selected_static_pts_mask]
        new_rotation = self._rotation[selected_static_pts_mask]
        new_xyz_disp = self._xyz_disp[selected_static_pts_mask]
        
        new_xyz_error_min = self.xyz_error_min[selected_static_pts_mask]
        new_xyz_error_min_timestamp = self.xyz_error_min_timestamp[selected_static_pts_mask]
        self.xyz_error_min = torch.cat([self.xyz_error_min, new_xyz_error_min])
        self.xyz_error_min_timestamp = torch.cat([self.xyz_error_min_timestamp, new_xyz_error_min_timestamp])
        
        if self._xyz_motion.shape[0] == 0:
            self.densification_postfix_onlystatic(
                new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_xyz_disp, 
                )
            return
        # dynamic points
        selected_dynamic_pts_mask = torch.where(torch.norm(dynamic_grads, dim=-1) >= dynamic_grad_threshold, True, False)
        selected_dynamic_pts_mask = torch.logical_and(selected_dynamic_pts_mask,
                                              torch.max(self.get_motion_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz_motion = self._xyz_motion[selected_dynamic_pts_mask]
        new_features_dc_motion = self._features_dc_motion[selected_dynamic_pts_mask]
        new_features_rest_motion = self._features_rest_motion[selected_dynamic_pts_mask]
        new_scaling_motion = self._scaling_motion[selected_dynamic_pts_mask]
        new_rotation_motion = self._rotation_motion[selected_dynamic_pts_mask]
        new_opacity_motion = self._opacity_motion[selected_dynamic_pts_mask]
        new_opacity_duration_center = self._opacity_duration_center[selected_dynamic_pts_mask]
        new_opacity_duration_center_len = ((new_opacity_duration_center[:, 1] - new_opacity_duration_center[:, 0]).abs() / 3).clamp_min(2/self.interval)
        new_opacity_duration_center[:, 1] = new_opacity_duration_center[:, 1] + (new_opacity_duration_center_len * (torch.randn_like(new_opacity_duration_center[:, 1])))
        new_opacity_duration_center[:, 0] = new_opacity_duration_center[:, 0] + (new_opacity_duration_center_len * (torch.randn_like(new_opacity_duration_center[:, 0])))
        new_opacity_duration_center = new_opacity_duration_center.clamp((self.time_shift + 1)/self.interval, (self.time_shift+self.duration - 1)/self.interval)
        new_opacity_duration_var = torch.ones_like(self._opacity_duration_var[selected_dynamic_pts_mask]) * 2
        
        new_motion_xyz_error_min = self.motion_xyz_error_min[selected_dynamic_pts_mask]
        new_motion_xyz_error_min_timestamp = self.motion_xyz_error_min_timestamp[selected_dynamic_pts_mask]
        self.motion_xyz_error_min = torch.cat([self.motion_xyz_error_min, new_motion_xyz_error_min])
        self.motion_xyz_error_min_timestamp = torch.cat([self.motion_xyz_error_min_timestamp, new_motion_xyz_error_min_timestamp])

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_xyz_disp, 
            new_xyz_motion, new_features_dc_motion, new_features_rest_motion, new_scaling_motion, new_opacity_motion, new_opacity_duration_center, new_opacity_duration_var, new_rotation_motion
            )

    def densify_and_prune(self, max_grad, max_dgrad, min_opacity, min_motion_opacity, extent,
                          max_screen_size, max_dynamic_screen_size, duration_thres=-5.0, 
                          s_max_ssim=0.5, s_l1_thres=0.1, d_max_ssim=0.5, d_l1_thres=0.1):
        static_grads = self.xyz_gradient_accum / self.denom
        static_grads[static_grads.isnan()] = 0.0
        
        dynamic_grads = self.motion_xyz_gradient_accum / self.motion_denom
        dynamic_grads[dynamic_grads.isnan()] = 0.0
        
        dynamic_ssim_errors = self.motion_xyz_ssim_error_accum / (self.motion_error_denom).clamp_min(1e-4)
        dynamic_ssim_errors[dynamic_ssim_errors.isnan()] = 0.0

        self.densify_and_clone(static_grads, dynamic_grads, dynamic_ssim_errors, max_grad, max_dgrad, extent)
        self.densify_and_split(static_grads, dynamic_grads, dynamic_ssim_errors, max_grad, max_dgrad, extent, max_screen_size, max_dynamic_screen_size)

        static_prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_static_scaling.max(dim=1).values > 0.1 * extent
            static_prune_mask = torch.logical_or(torch.logical_or(static_prune_mask, big_points_vs), big_points_ws)
            
        static_l1_mask = self.xyz_error_accum / self.error_denom.clamp(1e-4)
        static_l1_mask = static_l1_mask > s_l1_thres
        static_prune_mask = torch.logical_or(static_prune_mask, static_l1_mask.squeeze())
        
        static_ssim_mask = self.xyz_ssim_error_accum / self.error_denom.clamp(1e-4)
        static_ssim_mask = (static_ssim_mask < s_max_ssim) * (static_ssim_mask > 0)
        static_prune_mask = torch.logical_or(static_prune_mask, static_ssim_mask.squeeze())
            
        if self._xyz_motion.shape[0] == 0:
            dynamic_prune_mask = torch.empty(0).cuda()
            self.prune_points(static_prune_mask, dynamic_prune_mask)
            torch.cuda.empty_cache()
            return
        
        # dynamic points
        dynamic_prune_mask = (self.get_motion_opacity < min_motion_opacity).squeeze()
        
        dynamic_l1_mask = self.motion_xyz_error_mean / self.motion_error_denom.clamp(1e-4)
        dynamic_l1_mask = dynamic_l1_mask > d_l1_thres
        dynamic_prune_mask = torch.logical_or(dynamic_prune_mask, dynamic_l1_mask.squeeze())
        
        dynamic_ssim_mask = self.motion_xyz_ssim_error_accum / self.motion_error_denom.clamp(1e-4)
        dynamic_ssim_mask = (dynamic_ssim_mask < d_max_ssim) * (dynamic_ssim_mask > 0)
        dynamic_prune_mask = torch.logical_or(dynamic_prune_mask, dynamic_ssim_mask.squeeze())
        
        if max_dynamic_screen_size:
            big_points_vs = self.motion_max_radii2D > max_dynamic_screen_size
            big_points_ws = self.get_motion_scaling.max(dim=1).values > 0.1 * extent
            dynamic_prune_mask = torch.logical_or(torch.logical_or(dynamic_prune_mask, big_points_vs), big_points_ws)
            
        self.prune_points(static_prune_mask, dynamic_prune_mask)

        torch.cuda.empty_cache()

    def prune_invisible(self):
        static_prune_mask = (self.xyz_error_min_timestamp < 0).squeeze()
        dynamic_prune_mask = (self.motion_xyz_error_min_timestamp < 0).squeeze()
        
        torch.logical_or(
            dynamic_prune_mask,
            ((self._opacity_duration_center[:, 1] - self._opacity_duration_center[:, 0]).abs() < 0.5 / self.interval).squeeze()
        )
        
        self.prune_points(static_prune_mask, dynamic_prune_mask)

        torch.cuda.empty_cache()

    def prune_small(self):
        static_prune_mask = (self.min_radii2D < 5).squeeze()
        dynamic_prune_mask = (self.motion_min_radii2D < 5).squeeze()
        
        self.prune_points(static_prune_mask, dynamic_prune_mask)

        torch.cuda.empty_cache()
        
    def add_densification_stats(self, viewspace_point_tensor, static_update_filter, dynamic_update_filter, static_num):
        self.xyz_gradient_accum[static_update_filter] += torch.norm(viewspace_point_tensor.grad[:static_num][static_update_filter,:2], dim=-1, keepdim=True)
        self.denom[static_update_filter] += 1

        if self.motion_xyz_gradient_accum.shape[0] == 0:
            return
        
        self.motion_xyz_gradient_accum[dynamic_update_filter] += torch.norm(viewspace_point_tensor.grad[static_num:][dynamic_update_filter,:2], dim=-1, keepdim=True)
        self.motion_denom[dynamic_update_filter] += 1
        
    def mark_prune_stats(self, radii, viewspace_point_error_tensor):
        static_num = self._xyz.shape[0]
        
        static_radii = radii[:static_num]
        static_vis_filter = viewspace_point_error_tensor.grad[:static_num, 0] > 0
        self.min_radii2D[static_vis_filter] = torch.min(self.min_radii2D[static_vis_filter], static_radii[static_vis_filter])
                        
        if self.motion_xyz_gradient_accum.shape[0] == 0:
            return
        
        dynamic_radii = radii[static_num:]
        dynamic_vis_filter = viewspace_point_error_tensor.grad[static_num:, 0] > 0
        self.motion_min_radii2D[dynamic_vis_filter] = torch.min(self.motion_min_radii2D[dynamic_vis_filter], dynamic_radii[dynamic_vis_filter])
        
    def add_l1_ssim_stats(self, viewspace_point_error_tensor, static_update_filter, dynamic_update_filter, static_num, timestamp):
        static_errors = viewspace_point_error_tensor.grad[:static_num]
        static_l1_error = static_errors[static_update_filter, 1:2] / static_errors[static_update_filter, 0:1].clamp_min(1e-4)
        self.xyz_error_accum[static_update_filter] += static_l1_error
        self.xyz_error_min_timestamp[static_update_filter] = torch.where(torch.logical_and(self.xyz_error_min[static_update_filter] > static_l1_error, static_errors[static_update_filter, 0:1] > 0.01), 
                                                                                 timestamp * torch.ones_like(static_l1_error), 
                                                                                 self.xyz_error_min_timestamp[static_update_filter])
        self.xyz_error_min[static_update_filter] = torch.where(torch.logical_and(self.xyz_error_min[static_update_filter] > static_l1_error, static_errors[static_update_filter, 0:1] > 0.01), 
                                                                       static_l1_error, self.xyz_error_min[static_update_filter])
        self.xyz_ssim_error_accum[static_update_filter] += static_errors[static_update_filter, 2:3] / static_errors[static_update_filter, 0:1].clamp_min(1e-4)
        self.error_denom[static_update_filter] += (static_errors[static_update_filter, 0:1] > 0).float()
        
        if self.motion_xyz_error_min.shape[0] == 0:
            return
        
        dynamic_errors = viewspace_point_error_tensor.grad[static_num:]
        
        dynamic_l1_error = dynamic_errors[dynamic_update_filter, 1:2] / dynamic_errors[dynamic_update_filter, 0:1].clamp_min(1e-4)
        self.motion_xyz_error_min_timestamp[dynamic_update_filter] = torch.where(torch.logical_and(self.motion_xyz_error_min[dynamic_update_filter] > dynamic_l1_error, dynamic_errors[dynamic_update_filter, 0:1] > 0.01),
                                                                                 timestamp * torch.ones_like(dynamic_l1_error), 
                                                                                 self.motion_xyz_error_min_timestamp[dynamic_update_filter])
        self.motion_xyz_error_min[dynamic_update_filter] = torch.where(torch.logical_and(self.motion_xyz_error_min[dynamic_update_filter] > dynamic_l1_error, dynamic_errors[dynamic_update_filter, 0:1] > 0.01),
                                                                       dynamic_l1_error, self.motion_xyz_error_min[dynamic_update_filter])
        
        self.motion_xyz_error_mean[dynamic_update_filter] += dynamic_l1_error
        self.motion_xyz_ssim_error_accum[dynamic_update_filter] += dynamic_errors[dynamic_update_filter, 2:3] / dynamic_errors[dynamic_update_filter, 0:1].clamp_min(1e-4)
        self.motion_error_denom[dynamic_update_filter] += (dynamic_errors[dynamic_update_filter, 0:1] > 0).float()

    def extract_dynamic_points_from_static(self, viewpoint_loc, timestamp, vis_filter, extent, percentile=0.98, motion_thres=1000.0, min_motion_thres=1e-6, win_size=0, max_dur=None):
        if max_dur is None:
            max_dur = self.duration
        else:
            max_dur = max(float(max_dur), self.interval)
            
        disp = self._xyz_disp[vis_filter].norm(dim=-1)
        disp_denorm = (self._xyz[vis_filter] - viewpoint_loc.to(self._xyz.device)).norm(dim=-1) ** 2
        disp = disp / (disp_denorm+0.000001)
        disp = disp / (disp.max()+0.000001)
        mv_thresh = torch.quantile(disp, percentile)
        
        dynamic_mask = (disp > mv_thresh) | (self._xyz_disp.norm(dim=-1)[vis_filter] > motion_thres * extent)
        dynamic_mask = dynamic_mask & (self._xyz_disp.norm(dim=-1)[vis_filter] > min_motion_thres * extent)
        static_prune_mask = vis_filter.clone()
        static_prune_mask[vis_filter] = dynamic_mask
        static_prune_mask = torch.logical_and(static_prune_mask, self.xyz_error_min_timestamp.squeeze() >= 0)

        if self.keyframe_num == 0:
            self.keyframe_num = math.ceil((max_dur + self.time_shift*2 + 1) / self.interval) + 1 + 2
        new_xyz_motion = nn.functional.interpolate(torch.stack([
                self._xyz[static_prune_mask] - self._xyz_disp[static_prune_mask] * self.interval / max_dur,
                self._xyz[static_prune_mask] + self._xyz_disp[static_prune_mask] * (1 + self.interval / max_dur)],dim=2).unsqueeze(-1),
                size=(self.keyframe_num, 1), mode='bilinear').squeeze(-1).transpose(1, 2)
        if self.motion_degree > 1:
            new_xyz_motion = torch.cat([new_xyz_motion, torch.zeros_like(new_xyz_motion).repeat(1, 1, self.motion_degree-1)], dim=-1)

        new_features_dc_motion = self._features_dc[static_prune_mask]
        new_features_rest_motion = self._features_rest[static_prune_mask]
        new_scaling_motion = self._scaling[static_prune_mask]
        new_opacity_motion = self._opacity[static_prune_mask]
        t = self.xyz_error_min_timestamp[static_prune_mask]
        min_time = 0
        new_opacity_duration_center = torch.stack([
            torch.ones_like(new_opacity_motion) * (t * 1 / 2 + self.time_shift ) / self.interval, 
            torch.ones_like(new_opacity_motion) * ( (max_dur + t.clamp_min(min_time) * 1) / 2 + self.time_shift) / self.interval,
            ], dim=1).clamp((min_time+self.time_shift+1)/self.interval, (self.time_shift+max_dur-1)/self.interval)
        new_opacity_duration_var = torch.stack([
            torch.ones_like(new_opacity_motion) * (t + self.time_pad) , 
            torch.ones_like(new_opacity_motion) * (max_dur - t + self.time_pad) ,
            ], dim=1)

        new_rotation_motion = self._rotation[static_prune_mask].unsqueeze(1).repeat(1, self.keyframe_num, 1)
        
        d = {
        "motion_xyz" : new_xyz_motion,
        "motion_f_dc" : new_features_dc_motion,
        "motion_f_rest" : new_features_rest_motion,
        "motion_scaling" : new_scaling_motion,
        "motion_opacity" : new_opacity_motion,
        "motion_opacity_center" : new_opacity_duration_center,
        "motion_opacity_var" : new_opacity_duration_var,
        "motion_rotation" : new_rotation_motion
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # dynamic points 
        self._xyz_motion = optimizable_tensors["motion_xyz"]
        self._features_dc_motion = optimizable_tensors["motion_f_dc"]
        self._features_rest_motion = optimizable_tensors["motion_f_rest"]
        self._scaling_motion = optimizable_tensors["motion_scaling"]
        self._opacity_motion = optimizable_tensors["motion_opacity"]
        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]
        self._rotation_motion = optimizable_tensors["motion_rotation"]
        
        # reset grad anyway
        self.motion_xyz_gradient_accum = torch.zeros((self._xyz_motion.size(0), 1), device="cuda")
        self.motion_denom = torch.zeros((self._xyz_motion.size(0), 1), device="cuda")
        self.motion_xyz_error_mean = torch.zeros((self._xyz_motion.size(0), 1), device="cuda")
        self.motion_xyz_ssim_error_accum = torch.zeros((self._xyz_motion.size(0), 1), device="cuda")
        self.motion_error_denom = torch.zeros((self._xyz_motion.size(0), 1), device="cuda")
        self.motion_max_radii2D = torch.zeros((self._xyz_motion.size(0)), device="cuda")
        self.motion_min_radii2D = torch.ones((self._xyz_motion.size(0)), device="cuda") * 1000
        
        new_motion_xyz_error_min = torch.ones((static_prune_mask.sum(), 1), device="cuda") * 1000
        new_motion_xyz_error_min_timestamp = torch.ones((static_prune_mask.sum(), 1), device="cuda") * -1
        self.motion_xyz_error_min = torch.cat([self.motion_xyz_error_min, new_motion_xyz_error_min])
        self.motion_xyz_error_min_timestamp = torch.cat([self.motion_xyz_error_min_timestamp, new_motion_xyz_error_min_timestamp])
    
        self.prune_points(static_prune_mask, torch.zeros(self._xyz_motion.size(0), device=self._xyz_motion.device, dtype=torch.bool))

    def prune_nan_points(self):
        if self._xyz.shape[0] > 0 and self._xyz.isnan().any():
            static_prune_mask = self._xyz.isnan().any(dim=-1)
        else:
            static_prune_mask =  torch.zeros(self._xyz.size(0), device=self._xyz.device, dtype=torch.bool)
        if self._xyz_motion.shape[0] > 0 and self._xyz_motion.isnan().any():
            dynamic_prune_mask = self._xyz_motion.isnan().flatten(start_dim=1).any(dim=-1)
        else:
            dynamic_prune_mask =  torch.zeros(self._xyz_motion.size(0), device=self._xyz_motion.device, dtype=torch.bool)
        
        if static_prune_mask.sum() + dynamic_prune_mask.sum() > 0:
            print("Prune {} static and {} dynamic points".format(static_prune_mask.sum(), dynamic_prune_mask.sum()))
            self.prune_points(static_prune_mask, dynamic_prune_mask)

    def expand_duration(self, duration):
        duration = int(duration)+1
        if duration <= self.duration:
            return False
        
        # no dynamic points
        if self._xyz_motion.shape[0] == 0:
            # print("No dynamic points to expand. static num: {} dynamic num: {}".format(self._xyz.shape[0], self._xyz_motion.shape[0]))
            self.duration = duration
            return False
        
        require_dim = math.ceil((duration + self.time_shift + self.time_pad*2 + 1) / self.interval) + 1 + 2
        cur_dim = self._xyz_motion.shape[1]
        num_expand = require_dim - cur_dim
        
        if num_expand < 1:
            # print("Not expanded: {}".format(duration))
            self.duration = duration
            return False
            
        # linear linterpolation of lastest frame
        def lin_interp_last(x, n, zero_init=False, average=1):
            diff = (x[:, -average:] - x[:, -average-1:-average]).mean(dim=1, keepdim=True) * 1.0
            new_frames = torch.arange(1, n+1, device="cuda").view(1, -1, * [1] * len(diff.shape[2:])) * diff + x[:, -1:]
            if self.motion_degree > 1 and zero_init:
                new_frames[..., 3:] = 0.
            return torch.cat([x, new_frames], dim=1)

        num_avg = min(self.keyframe_num-2, 4)
        new_xyz_motion = lin_interp_last(self._xyz_motion, num_expand, zero_init=True, average=num_avg)
        new_rotation_motion = lin_interp_last(self._rotation_motion, num_expand, average=num_avg)
        
        new_opacity_duration_var = self._opacity_duration_var.detach().clone()
        new_opacity_duration_var[:, 1] = torch.where((self._opacity_duration_center + self.time_shift / self.interval > (duration + self.time_shift) / self.interval - 0.5).any(dim=1),
                                                   torch.ones_like(self._opacity_duration_var[:, 1]),
                                                   self._opacity_duration_var[:, 1])
        new_opacity_duration_center = self._opacity_duration_center.clamp_max((self.time_shift+self.duration - 1)/self.interval)
        
        d = {
        "motion_xyz" : new_xyz_motion,
        "motion_opacity_center" : new_opacity_duration_center,
        "motion_opacity_var" : new_opacity_duration_var,
        "motion_rotation" : new_rotation_motion,
        }
            
        optimizable_tensors = self.replace_tensor_to_optimizer(d)
        
        self._xyz_motion = optimizable_tensors["motion_xyz"]
        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]
        self._rotation_motion = optimizable_tensors["motion_rotation"]
        self.keyframe_num = require_dim
        self.duration = duration
        
        return True
    
    def mark_error(self, loss, timestamp):
        t_idx = timestamp // self.interval
        
        if t_idx in self.error_dict.keys():
            self.error_dict[t_idx] = (self.error_dict[t_idx][0] + loss, self.error_dict[t_idx][1] + 1)
        else:
            self.error_dict[t_idx] = (loss, 1)
    
    def get_errorneous_timestamp(self):
        if len(self.error_dict) == 0:
            return None
        
        max_loss = 0
        max_idx = 0
        max_count = 0
        
        for t_idx in self.error_dict.keys():
            loss, count = self.error_dict[t_idx]
            max_count = max(max_count, count)
            
            if loss / count > max_loss and count > max_count * 0.1:
                max_loss = loss / count
                max_idx = t_idx
        
        if max_loss == 0:
            return None
        
        del self.error_dict[max_idx]
        
        return (max_idx + 0.5) * self.interval
 
    def adjust_temp_opa(self, max_dur=None):
        if max_dur is None:
            max_dur = self.duration
        else:
            max_dur = float(max_dur)
        if self._xyz_motion.shape[0] == 0:
            return
        new_opacity_duration_var = self._opacity_duration_var.detach().clone()
        new_opacity_duration_var[:, 1] = torch.where((self._opacity_duration_center > (max_dur + self.time_shift) / self.interval - 0.2).any(dim=1),
                                                   self._opacity_duration_var[:, 1].clamp_min(1)*2,
                                                   self._opacity_duration_var[:, 1])
        new_opacity_duration_var[:, 0] = torch.where((self._opacity_duration_center < (self.time_shift) / self.interval + 0.2).any(dim=1),
                                                   self._opacity_duration_var[:, 0].clamp_min(1)*2,
                                                   self._opacity_duration_var[:, 0])
        new_opacity_duration_center = self._opacity_duration_center.clamp((self.time_shift) / self.interval + 0.2, (max_dur + self.time_shift) / self.interval - 0.2)
        
        new_opacity_duration_var = torch.where(self._opacity_duration_var < 0.5,
                                                   torch.ones_like(self._opacity_duration_var)*0.5,
                                                   new_opacity_duration_var)
        
        d = {
        "motion_opacity_center" : new_opacity_duration_center,
        "motion_opacity_var" : new_opacity_duration_var,
        }
            
        optimizable_tensors = self.replace_tensor_to_optimizer(d)
        
        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]