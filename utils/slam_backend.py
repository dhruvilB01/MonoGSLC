import os
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image, ImageDraw
from evo.core import metrics
from evo.core.trajectory import PosePath3D
from tqdm import tqdm

import wandb

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
from utils.pose_graph import PoseGraph


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.loop_events = []

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]
        self.use_wandb = self.config["Results"].get("use_wandb", False)

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        pose_graph_cfg = self.config.get("PoseGraph", {})
        self.use_pose_graph = pose_graph_cfg.get("enabled", False)
        self.pose_graph_optimize_on_loop = pose_graph_cfg.get(
            "optimize_on_loop", True
        )
        self.pose_graph_optimize_every_n_loops = pose_graph_cfg.get(
            "optimize_every_n_loops", 1
        )
        self.pose_graph = PoseGraph() if self.use_pose_graph else None
        self.pose_graph_max_translation_delta = pose_graph_cfg.get(
            "max_translation_delta", None
        )
        rot_deg = pose_graph_cfg.get("max_rotation_delta_deg", None)
        self.pose_graph_max_rotation_delta = (
            np.deg2rad(rot_deg) if rot_deg is not None else None
        )
        self.pose_graph_freeze_active_window = pose_graph_cfg.get(
            "freeze_active_window", False
        )
        self.pose_graph_loop_counter = 0
        save_dir = self.config["Results"].get("save_dir")
        self.loop_closure_viz_dir = None
        if save_dir:
            self.loop_closure_viz_dir = os.path.join(save_dir, "loop_closures")
            mkdir_p(self.loop_closure_viz_dir)
        loop_cfg = self.config.get("LoopClosure", {})
        self.loop_max_ate_increase = loop_cfg.get("max_ate_increase", 0.0)
        self.loop_min_ate_improvement = loop_cfg.get("min_ate_improvement", None)

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def _estimate_current_ate(self):
        if len(self.viewpoints) < 2:
            return None
        kf_ids = sorted(self.viewpoints.keys())

        def pose_from_rt(R, T):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = R.detach().cpu().numpy()
            pose[:3, 3] = T.detach().cpu().numpy()
            return pose

        try:
            poses_est, poses_gt = [], []
            for kf_id in kf_ids:
                kf = self.viewpoints[kf_id]
                if not hasattr(kf, "R_gt") or not hasattr(kf, "T_gt"):
                    return None
                pose_est = np.linalg.inv(pose_from_rt(kf.R, kf.T))
                pose_gt = np.linalg.inv(pose_from_rt(kf.R_gt, kf.T_gt))
                poses_est.append(pose_est)
                poses_gt.append(pose_gt)

            traj_ref = PosePath3D(poses_se3=poses_gt)
            traj_est = PosePath3D(poses_se3=poses_est)
            traj_est_aligned = PosePath3D(
                poses_se3=[pose.copy() for pose in traj_est.poses_se3]
            )
            traj_est_aligned.align(traj_ref, correct_scale=self.monocular)
            ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
            ape_metric.process_data((traj_ref, traj_est_aligned))
            ate = ape_metric.get_statistic(metrics.StatisticsType.rmse)
            return float(ate)
        except Exception as exc:
            Log(f"Failed to compute loop-closure ATE: {exc}")
            return None

    def _rt_to_matrix(self, R, T):
        mat = torch.eye(4, device=R.device, dtype=R.dtype)
        mat[:3, :3] = R
        mat[:3, 3] = T
        return mat

    def _compose_query_pose(self, rel_pose, T_anchor):
        T_cw_anchor = torch.linalg.inv(T_anchor)
        T_cw_query = rel_pose @ T_cw_anchor
        return torch.linalg.inv(T_cw_query)

    def _compute_loop_sim3(self, anchor_view, T_query_prev, T_query_target):
        Tc_anchor = torch.linalg.inv(self._rt_to_matrix(anchor_view.R, anchor_view.T))
        Tc_prev = torch.linalg.inv(T_query_prev)
        Tc_target = torch.linalg.inv(T_query_target)

        c_anchor = Tc_anchor[:3, 3]
        c_prev = Tc_prev[:3, 3]
        c_target = Tc_target[:3, 3]

        dist_prev = torch.norm(c_prev - c_anchor)
        dist_target = torch.norm(c_target - c_anchor)
        if not torch.isfinite(dist_prev) or dist_prev.item() < 1e-6:
            scale = torch.tensor(1.0, device=self.device)
        else:
            raw_scale = dist_target / torch.clamp(dist_prev, min=1e-6)
            scale = torch.clamp(raw_scale, min=0.5, max=2.0)

        R_prev = Tc_prev[:3, :3]
        R_target = Tc_target[:3, :3]
        R_delta = R_target @ R_prev.transpose(0, 1)
        t_delta = c_target - scale * (R_delta @ c_prev)

        sim3 = torch.eye(4, device=self.device, dtype=torch.float32)
        sim3[:3, :3] = scale * R_delta.to(dtype=torch.float32)
        sim3[:3, 3] = t_delta.to(dtype=torch.float32)

        if torch.isnan(sim3).any() or torch.isinf(sim3).any():
            return None
        return sim3

    def _select_loop_submap(self, anchor_idx, query_idx):
        if len(self.viewpoints) == 0:
            return []
        ids = sorted(self.viewpoints.keys())
        if query_idx >= anchor_idx:
            return [idx for idx in ids if idx >= query_idx]
        return [idx for idx in ids if idx <= query_idx]

    def _snapshot_viewpoints(self, kf_ids):
        snapshot = {}
        for kf_id in kf_ids:
            viewpoint = self.viewpoints.get(kf_id)
            if viewpoint is None:
                continue
            snapshot[kf_id] = (viewpoint.R.clone(), viewpoint.T.clone())
        return snapshot

    def _restore_viewpoints(self, snapshot):
        if not snapshot:
            return
        for kf_id, (R, T) in snapshot.items():
            viewpoint = self.viewpoints.get(kf_id)
            if viewpoint is None:
                continue
            viewpoint.update_RT(R.clone(), T.clone())

    def _gaussian_mask_for_ids(self, kf_ids):
        if (
            not kf_ids
            or self.gaussians is None
            or self.gaussians._xyz.shape[0] == 0
            or self.gaussians.unique_kfIDs.shape[0] == 0
        ):
            return None
        ids_tensor = torch.tensor(
            kf_ids,
            device=self.gaussians.unique_kfIDs.device,
            dtype=self.gaussians.unique_kfIDs.dtype,
        )
        mask_cpu = torch.isin(self.gaussians.unique_kfIDs, ids_tensor)
        if not mask_cpu.any():
            return None
        return mask_cpu.to(device=self.gaussians._xyz.device)

    def _snapshot_gaussians(self, mask):
        if mask is None:
            return None
        snapshot = self.gaussians._xyz[mask].clone().detach()
        return mask.clone(), snapshot

    def _restore_gaussians(self, snapshot):
        if snapshot is None:
            return
        mask, xyz = snapshot
        if mask.shape[0] != self.gaussians._xyz.shape[0]:
            # mask corresponds to previous tensor size, skip restore
            return
        with torch.no_grad():
            self.gaussians._xyz[mask] = xyz.clone()

    def _apply_sim3_to_viewpoints(self, sim3, kf_ids):
        if not kf_ids:
            return
        with torch.no_grad():
            for kf_id in kf_ids:
                viewpoint = self.viewpoints.get(kf_id)
                if viewpoint is None:
                    continue
                Twc = self._rt_to_matrix(viewpoint.R, viewpoint.T)
                Tc = torch.linalg.inv(Twc)
                Tc_new = sim3 @ Tc
                Twc_new = torch.linalg.inv(Tc_new)
                viewpoint.update_RT(
                    Twc_new[:3, :3].clone().to(device=self.device),
                    Twc_new[:3, 3].clone().to(device=self.device),
                )

    def _apply_sim3_to_gaussians(self, sim3, kf_ids, mask=None):
        if mask is None:
            mask = self._gaussian_mask_for_ids(kf_ids)
        if mask is None:
            return
        with torch.no_grad():
            pts = self.gaussians._xyz[mask]
            ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
            pts_h = torch.cat([pts, ones], dim=1)
            pts_trans = (sim3 @ pts_h.transpose(0, 1)).transpose(0, 1)
            self.gaussians._xyz[mask] = pts_trans[:, :3]

    def _viewpoint_to_image(self, viewpoint):
        if (
            viewpoint is None
            or getattr(viewpoint, "original_image", None) is None
            or viewpoint.original_image is None
        ):
            return None
        tensor = viewpoint.original_image
        if not torch.is_tensor(tensor):
            return None
        img = tensor.detach().clone().to("cpu")
        if img.dim() != 3:
            return None
        img = img.permute(1, 2, 0).contiguous().numpy()
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
        return img

    def _save_loop_closure_images(
        self,
        anchor_idx,
        query_idx,
        anchor_view,
        query_view,
        event,
        ate_before,
        ate_after,
        accepted,
    ):
        if self.loop_closure_viz_dir is None:
            return
        try:
            anchor_img = self._viewpoint_to_image(anchor_view)
            query_img = self._viewpoint_to_image(query_view)
            if anchor_img is None or query_img is None:
                return
            anchor_pil = Image.fromarray(anchor_img)
            query_pil = Image.fromarray(query_img)

            target_h = max(anchor_pil.height, query_pil.height)

            def resize_to_height(img):
                if img.height == target_h or img.height == 0:
                    return img
                new_w = max(1, int(round(img.width * target_h / max(1, img.height))))
                return img.resize((new_w, target_h))

            anchor_pil = resize_to_height(anchor_pil)
            query_pil = resize_to_height(query_pil)

            spacing = 10
            combined_w = anchor_pil.width + spacing + query_pil.width
            combined = Image.new("RGB", (combined_w, target_h), color=(0, 0, 0))
            combined.paste(anchor_pil, (0, 0))
            combined.paste(query_pil, (anchor_pil.width + spacing, 0))

            draw = ImageDraw.Draw(combined)
            status = "accepted" if accepted else "rejected"
            score = event.get("score", 0.0)
            inliers = event.get("inliers", 0)
            ate_text = ""
            if ate_before is not None and ate_after is not None:
                ate_text = f"ATE {ate_before:.3f}->{ate_after:.3f}"
            lines = [
                f"anchor {anchor_idx} | query {query_idx}",
                f"score={score:.3f} inliers={inliers} status={status}",
            ]
            if ate_text:
                lines.append(ate_text)

            y = 5
            for line in lines:
                draw.text((6, y + 1), line, fill=(0, 0, 0))
                draw.text((5, y), line, fill=(255, 255, 255))
                y += 14

            filename = (
                f"loop_{status}_anchor{anchor_idx:05d}_query{query_idx:05d}"
                f"_score{score:.3f}_in{inliers}.png"
            )
            combined.save(os.path.join(self.loop_closure_viz_dir, filename))
        except Exception as exc:
            Log(f"Failed to save loop closure visualization: {exc}")

    def _apply_global_pose_corrections(self, optimized_nodes, exclude_ids=None):
        if not optimized_nodes:
            return
        if self.gaussians is None or self.gaussians._xyz.shape[0] == 0:
            return
        old_nodes = self.pose_graph.nodes if self.pose_graph else {}
        if not old_nodes:
            return
        exclude_ids = set(exclude_ids or [])
        for kf_id, T_new in optimized_nodes.items():
            if kf_id in exclude_ids:
                continue
            viewpoint = self.viewpoints.get(kf_id)
            if viewpoint is None:
                continue
            viewpoint.update_RT(
                torch.from_numpy(T_new[:3, :3])
                .to(device=self.device, dtype=torch.float32)
                .clone(),
                torch.from_numpy(T_new[:3, 3])
                .to(device=self.device, dtype=torch.float32)
                .clone(),
            )
        unique_ids = self.gaussians.unique_kfIDs.detach().cpu().numpy()
        new_map = {kf_id: T_new for kf_id, T_new in optimized_nodes.items()}
        transform_per_kf = {}
        for kf_id, T_old in old_nodes.items():
            if kf_id not in new_map:
                continue
            if kf_id in exclude_ids:
                continue
            T_new = new_map[kf_id]
            T_delta = T_new @ np.linalg.inv(T_old)
            transform_per_kf[kf_id] = T_delta
        if not transform_per_kf:
            return
        xyz = self.gaussians._xyz
        xyz_np = xyz.detach().cpu().numpy()
        for kf_id, T_delta in transform_per_kf.items():
            pts_idx = unique_ids == kf_id
            if not np.any(pts_idx):
                continue
            translation = np.linalg.norm(T_delta[:3, 3])
            if (
                self.pose_graph_max_translation_delta is not None
                and translation > self.pose_graph_max_translation_delta
            ):
                Log(
                    f"Skipping pose update for kf {kf_id} (translation delta {translation:.3f} m)"
                )
                continue
            rot_trace = (np.trace(T_delta[:3, :3]) - 1) / 2.0
            rot_trace = np.clip(rot_trace, -1.0, 1.0)
            rotation = np.arccos(rot_trace)
            if (
                self.pose_graph_max_rotation_delta is not None
                and rotation > self.pose_graph_max_rotation_delta
            ):
                Log(
                    f"Skipping pose update for kf {kf_id} (rotation delta {np.rad2deg(rotation):.2f} deg)"
                )
                continue
            pts = xyz_np[pts_idx]
            ones = np.ones((pts.shape[0], 1))
            pts_h = np.concatenate([pts, ones], axis=1)
            pts_trans = (T_delta @ pts_h.T).T[:, :3]
            xyz_np[pts_idx] = pts_trans
        self.gaussians._xyz = (
            torch.from_numpy(xyz_np)
            .to(device=self.gaussians._xyz.device, dtype=self.gaussians._xyz.dtype)
            .clone()
        )

    def _apply_sim3_to_submap(self, sim3, kf_ids):
        if sim3 is None or not kf_ids:
            return
        mask = self._gaussian_mask_for_ids(kf_ids)
        self._apply_sim3_to_viewpoints(sim3, kf_ids)
        self._apply_sim3_to_gaussians(sim3, kf_ids, mask=mask)

    def _optimize_loop_window(self, ordered_ids):
        if not ordered_ids:
            return
        chunk = max(1, self.config["Training"]["window_size"])
        loop_iters = max(10, self.mapping_itr_num // 2)
        for start in range(0, len(ordered_ids), chunk):
            window = ordered_ids[start : start + chunk]
            self.map(window, iters=loop_iters)
        self.current_window = ordered_ids[-chunk:]

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def handle_loop_closure(self, event):
        anchor_idx = event.get("anchor_idx")
        query_idx = event.get("query_idx")
        if anchor_idx not in self.viewpoints or query_idx not in self.viewpoints:
            return
        ate_before = self._estimate_current_ate()
        anchor = self.viewpoints[anchor_idx]
        query = self.viewpoints[query_idx]
        rel_pose = torch.tensor(event["rel_pose"], dtype=torch.float32, device=self.device)
        metric_scale = event.get("metric_scale", True)
        if not metric_scale:
            rel_pose = rel_pose.clone()
            rel_pose[:3, 3] = 0.0

        T_anchor = self._rt_to_matrix(anchor.R, anchor.T)
        T_query_prev = self._rt_to_matrix(query.R.clone(), query.T.clone())
        T_query_target = self._compose_query_pose(rel_pose, T_anchor)

        affected_ids = (
            self._select_loop_submap(anchor_idx, query_idx) if metric_scale else []
        )
        pose_snapshot_ids = affected_ids if metric_scale else [query_idx]
        viewpoint_snapshot = self._snapshot_viewpoints(pose_snapshot_ids)
        gaussian_snapshot = None
        if metric_scale and affected_ids:
            mask = self._gaussian_mask_for_ids(affected_ids)
            gaussian_snapshot = self._snapshot_gaussians(mask)

        sim3 = None
        if metric_scale:
            sim3 = self._compute_loop_sim3(anchor, T_query_prev, T_query_target)
        if sim3 is not None and affected_ids:
            mask = (
                gaussian_snapshot[0]
                if gaussian_snapshot is not None
                else self._gaussian_mask_for_ids(affected_ids)
            )
            self._apply_sim3_to_viewpoints(sim3, affected_ids)
            self._apply_sim3_to_gaussians(sim3, affected_ids, mask=mask)
        else:
            query.update_RT(
                T_query_target[:3, :3].clone(), T_query_target[:3, 3].clone()
            )

        self.loop_events.append(event)

        ate_after_transform = self._estimate_current_ate()
        if not self._loop_accepts(ate_before, ate_after_transform):
            self._restore_viewpoints(viewpoint_snapshot)
            self._restore_gaussians(gaussian_snapshot)
            ate_after_str = (
                f"{ate_after_transform:.3f}"
                if ate_after_transform is not None
                else "nan"
            )
            Log(
                f"Loop closure rejected anchor={anchor_idx} query={query_idx} "
                f"(ATE {ate_before:.3f}->{ate_after_str})"
            )
            self._save_loop_closure_images(
                anchor_idx,
                query_idx,
                anchor,
                query,
                event,
                ate_before,
                ate_after_transform,
                accepted=False,
            )
            return
        if self.use_pose_graph:
            self.pose_graph.add_loop_edge(anchor_idx, query_idx, T_query_target)

        window = []
        seen = set()
        for idx in self.current_window + [anchor_idx, query_idx]:
            if idx in seen:
                continue
            seen.add(idx)
            window.append(idx)
        max_window = self.config["Training"]["window_size"]
        fallback_window = window[-max_window:]

        if affected_ids:
            self._optimize_loop_window(affected_ids)
        else:
            self.current_window = fallback_window
            self.map(self.current_window, iters=self.mapping_itr_num)

        if len(self.current_window) == 0 and fallback_window:
            self.current_window = fallback_window

        if len(self.current_window) > 0:
            self.map(self.current_window, prune=True, iters=10)

        ate_after = self._estimate_current_ate()
        ate_msg = ""
        resolved = None
        if ate_before is not None and ate_after is not None:
            resolved = ate_before - ate_after
            ate_msg = (
                f" ATE {ate_before:.3f}->{ate_after:.3f} (resolved {resolved:+.3f} m)"
            )
        elif ate_before is not None:
            ate_msg = f" ATE before={ate_before:.3f} (post-update unavailable)"
        elif ate_after is not None:
            ate_msg = f" ATE after={ate_after:.3f}"
        Log(
            f"Loop closure detected anchor={anchor_idx} query={query_idx} "
            f"score={event.get('score', 0.0):.3f} "
            f"inliers={event.get('inliers', 0)}{ate_msg}"
        )
        if self.use_wandb and wandb.run is not None:
            wandb.log(
                {
                    "loop/anchor_idx": anchor_idx,
                    "loop/query_idx": query_idx,
                    "loop/score": float(event.get("score", 0.0)),
                    "loop/inliers": int(event.get("inliers", 0)),
                    "loop/ate_before": ate_before,
                    "loop/ate_after": ate_after,
                    "loop/ate_resolved": resolved,
                    "loop/metric_scale": metric_scale,
                }
            )
        self._save_loop_closure_images(
            anchor_idx,
            query_idx,
            anchor,
            query,
            event,
            ate_before,
            ate_after,
            accepted=True,
        )
        self.push_to_frontend("loop_closure")
        if self.use_pose_graph and self.pose_graph_optimize_on_loop:
            self.pose_graph_loop_counter += 1
            should_optimize = (
                self.pose_graph_optimize_every_n_loops <= 1
                or self.pose_graph_loop_counter
                % max(1, self.pose_graph_optimize_every_n_loops)
                == 0
            )
            if should_optimize:
                fixed_ids = (
                    set(int(k) for k in self.current_window)
                    if self.pose_graph_freeze_active_window
                    else None
                )
                optimized = self.pose_graph.optimize(fixed_ids=fixed_ids)
                if optimized:
                    self._apply_global_pose_corrections(
                        optimized, exclude_ids=fixed_ids
                    )

    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
                    if self.use_pose_graph:
                        T_cur = self._rt_to_matrix(viewpoint.R, viewpoint.T)
                        self.pose_graph.add_node(cur_frame_idx, T_cur)
                        prev_kf = (
                            current_window[-2] if len(current_window) > 1 else None
                        )
                        if prev_kf is not None:
                            prev_view = self.viewpoints[prev_kf]
                            T_prev = self._rt_to_matrix(prev_view.R, prev_view.T)
                            T_rel = torch.linalg.inv(T_prev) @ T_cur
                            self.pose_graph.add_odometry_edge(prev_kf, cur_frame_idx, T_rel)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                elif data[0] == "loop_closure":
                    event = data[1]
                    self.handle_loop_closure(event)
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
    def _loop_accepts(self, ate_before, ate_after):
        if ate_before is None or ate_after is None:
            return True
        improvement = ate_before - ate_after
        if (
            self.loop_min_ate_improvement is not None
            and improvement < self.loop_min_ate_improvement
        ):
            return False
        if (
            self.loop_max_ate_increase is not None
            and (ate_after - ate_before) > self.loop_max_ate_increase
        ):
            return False
        return True
