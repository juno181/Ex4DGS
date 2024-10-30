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
import json
import sys
import uuid
import math
from random import randint
import gc

import torch
from tqdm import tqdm
from PIL import Image
import joblib

from gaussian_renderer import render, network_gui
from scene import Scene, getmodel
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state, PILtoTorch
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
TENSORBOARD_FOUND = False
torch.set_default_dtype(torch.float32)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    GaussianModel = getmodel(dataset.model) # gmodel, gmodelrgbonly
    gaussians = GaussianModel(dataset.sh_degree, dataset.start_duration, dataset.time_interval, dataset.time_pad, 
                              interp_type=dataset.interp_type, rot_interp_type=dataset.rot_interp_type, 
                              time_pad_type=dataset.time_pad_type, var_pad=dataset.var_pad, kernel_size=dataset.kernel_size)
    scene = Scene(dataset, gaussians, use_timepad=True)
    gaussians.training_setup(opt)
    args.duration = dataset.duration
        
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # variables for progressive training
    scene.set_sampling_len(dataset.start_duration, sample_every=dataset.sample_every)
    expanded = gaussians.expand_duration(dataset.start_duration)
    
    sample_len = dataset.start_duration
    g_sample_len = dataset.start_duration

    mark_extract = False
    need_extract = True
    mark_last = False
    viewpoint_stack = None
    prune_inv = False
    train_images = None
    e_count = args.extract_every
    
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, far=dataset.far)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack, train_images = scene.getTrainCameras(return_as='generator', shuffle=True)
            viewpoint_stack = viewpoint_stack.copy()
                        
            if iteration > opt.prune_invisible_interval:
                prune_inv = True
            
        viewpoint_cam = viewpoint_stack.pop(0)
        gt_image = next(train_images).cuda()
        
        if mark_last:
            if viewpoint_cam.timestamp >= scene.sample_len - gaussians.interval:
                mark_extract = True
                mark_last = False

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, near=dataset.near, far=dataset.far)
        image, viewspace_point_tensor, viewspace_point_error_tensor, visibility_filter, radii, depth, flow, acc, idxs = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["viewspace_l1points"], render_pkg["visibility_filter"], \
            render_pkg["radii"], render_pkg["depth"], render_pkg["opticalflow"], render_pkg["acc"], render_pkg["dominent_idxs"]

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # backtrack register
        if opt.l1_accum:
            l1_errors = (image - gt_image).abs().mean(dim=0)
            ssim_errors = ssim(image, gt_image, reduce=False).mean(dim=0)
            hook_tensor = torch.stack([acc[0], l1_errors, ssim_errors])
            flow_h = flow.register_hook(lambda grad: hook_tensor)
            loss += flow.mean() * 0
                
        # Regularization
        if opt.static_reg > 0 and iteration > opt.progressive_growing_steps + opt.make_dynamic_interval:
            loss += opt.static_reg * torch.log(gaussians._xyz_disp.norm(dim=-1)+0.001).mean()

        if opt.motion_reg > 0 and iteration > opt.progressive_growing_steps * opt.extract_every + opt.make_dynamic_interval and gaussians._xyz_motion.shape[0] > 0:
            diff1 = (gaussians._xyz_motion[:, :1] - gaussians._xyz_motion[:, 1:])
            loss += opt.motion_reg * diff1.norm(dim=-1).mean()
            
        if opt.rot_reg > 0 and iteration > opt.progressive_growing_steps * opt.extract_every + opt.make_dynamic_interval and gaussians._xyz_motion.shape[0] > 0:
            r1 = gaussians._rotation_motion[:, 1:] 
            r2 = gaussians._rotation_motion[:, :-1]
            
            r_i = 1 - (r1 * r2).sum(dim=-1) / r1.norm(dim=-1).clamp_min(1e-6) / r2.norm(dim=-1).clamp_min(1e-6)
            loss += opt.rot_reg * r_i.mean()
            
        loss.backward()
        
        if opt.l1_accum:
            flow_h.remove()
        
        iter_end.record()

        gaussians.mark_error(loss.item(), viewpoint_cam.timestamp)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = loss.item()
            psnr_log = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_log:.{2}f}", "Loss": f"{ema_loss_for_log:.{6}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, (pipe, background), dataset.near, dataset.far)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if opt.l1_accum:
                gaussians.mark_prune_stats(radii, viewspace_point_error_tensor)
            
            # Densification
            if iteration < opt.densify_until_iter:
                static_num = gaussians._xyz.shape[0]
                static_vis_filter = visibility_filter[:static_num]
                static_radii = radii[:static_num]
                dynamic_vis_filter = visibility_filter[static_num:]
                dynamic_radii = radii[static_num:]
                
                gaussians.max_radii2D[static_vis_filter] = torch.max(gaussians.max_radii2D[static_vis_filter], static_radii[static_vis_filter])
                gaussians.motion_max_radii2D[dynamic_vis_filter] = torch.max(gaussians.motion_max_radii2D[dynamic_vis_filter], dynamic_radii[dynamic_vis_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, static_vis_filter, dynamic_vis_filter, static_num)

                if opt.l1_accum:
                    gaussians.add_l1_ssim_stats(viewspace_point_error_tensor, static_vis_filter, dynamic_vis_filter, static_num, viewpoint_cam.timestamp)
                    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold, dynamic_size_threshold = None, None
                    s_max_ssim = opt.s_max_ssim  if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.ssim_prune_every) == 0 else 0
                    s_l1_thres = opt.s_l1_thres if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.l1_prune_every) == 0 else 100
                    
                    d_max_ssim = opt.d_max_ssim  if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.ssim_prune_every) == 0 else 0
                    d_l1_thres = opt.d_l1_thres if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.l1_prune_every) == 0 else 100
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 
                                                opt.densify_dgrad_threshold, 
                                                0.01, 0.01, scene.cameras_extent, size_threshold, dynamic_size_threshold,
                                                s_max_ssim=s_max_ssim, s_l1_thres=s_l1_thres, d_max_ssim=d_max_ssim, d_l1_thres=d_l1_thres)
                elif iteration > opt.extract_from_iter and iteration % opt.extracton_interval == 0:
                    static_num = gaussians._xyz.shape[0]
                    candidate = gaussians.get_errorneous_timestamp()
                    if not candidate is None:
                        gaussians.extract_dynamic_points_from_static(torch.tensor(viewpoint_cam.T).unsqueeze(0), candidate, static_vis_filter, scene.cameras_extent, percentile=opt.extract_percentile, max_dur=sample_len)
            if iteration % (opt.densification_interval*4) == 0 and iteration < opt.densify_until_iter - 3000:
                gaussians.adjust_temp_opa(max_dur=sample_len)

            # if prune_inv and iteration < opt.iterations - 5000:
            if prune_inv and iteration < opt.iterations and iteration > 3000:
                gaussians.prune_invisible()
                if opt.l1_accum:
                    gaussians.prune_small()
                prune_inv = False
                
            # Optimizer step
            if iteration < opt.iterations:
                # prevent nan grad of dynamic opacity
                if gaussians._opacity_duration_var.shape[0] != 0:
                    if not gaussians._opacity_duration_var.grad is None:
                        gaussians._opacity_duration_var.grad = gaussians._opacity_duration_var.grad.nan_to_num()
                        
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                
                gaussians.prune_nan_points()
                
                torch.cuda.empty_cache()
                
            if iteration > opt.extract_from_iter and (iteration % opt.progressive_growing_steps == opt.make_dynamic_interval) and need_extract :
                mark_last = True
                need_extract = False
                
            # increase duration
            if iteration > opt.extract_from_iter and iteration % opt.progressive_growing_steps == 0 and iteration > opt.progressive_growing_steps and ~need_extract:
                sample_len = min(dataset.duration + gaussians.time_shift, int(dataset.time_interval * dataset.progressive_step) + scene.sample_len)

                scene.set_sampling_len(sample_len, sample_every=dataset.sample_every)
                g_sample_len = min(dataset.duration + gaussians.time_shift, sample_len)
                expanded = gaussians.expand_duration(g_sample_len)
                
                if expanded:
                    e_count += 1
                    if e_count >= opt.extract_every:
                        mark_last = True
                        need_extract = True
                        e_count = 0

            # create dynamic points from static points
            if mark_extract:
                static_num = gaussians._xyz.shape[0]
                static_vis_filter = visibility_filter[:static_num]
                gaussians.extract_dynamic_points_from_static(torch.tensor(viewpoint_cam.T).unsqueeze(0), viewpoint_cam.timestamp, 
                                                             static_vis_filter, scene.cameras_extent, percentile=opt.extract_percentile, max_dur=sample_len)
                mark_extract = False


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderArgs, near, far):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        
        test_viewpoint_stack, test_images = scene.getTestCameras(shuffle=False, return_as='generator',  n_job=1)
        test_viewpoint_stack = test_viewpoint_stack.copy()
        
        train_viewpoint_stack, train_images = scene.getTrainCameras(shuffle=False, return_as='generator', n_job=1)
        train_viewpoint_stack = train_viewpoint_stack.copy()
            
        validation_configs = ({'name': 'test', 'cameras': test_viewpoint_stack, 
                                                'images': test_images}, 
                              {'name': 'train', 'cameras': train_viewpoint_stack, 
                                                 'images': train_images})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                count = 0
                
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name'] == 'train':
                        sampled_idx_list = [idx % len(train_viewpoint_stack) for idx in range(5, 30, 5)]
                        if not idx in sampled_idx_list:
                            _ = next(config['images'])
                            continue
                    
                    gt_image = next(config['images']).cuda()
                    rend_pkg = render(viewpoint, scene.gaussians, near=near, far=far, *renderArgs)
                    image = torch.clamp(rend_pkg["render"], 0.0, 1.0).cuda()
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double().detach().item()
                    psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double().detach().item()
                    count += 1
                    
                    del(rend_pkg)
                    del(image)
                    del(gt_image)
                    del(viewpoint)
                    torch.cuda.empty_cache()
                    
                psnr_test /= count
                l1_test /= count
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/motion_opacity_histogram", scene.gaussians.get_motion_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians._xyz.shape[0]+scene.gaussians._xyz_motion.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[60000, 80000, 7000_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[60000, 80000, 7000_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--configpath", type=str, default = "None")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # incase we provide config file not directly pass to the file
    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k) 
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
    else:
        raise ValueError("config file not exist or not provided")

    # Start GUI server, configure and run training
    while True:
        try:
            network_gui.init(args.ip, args.port)
            break
        except:
            args.port += 1
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
