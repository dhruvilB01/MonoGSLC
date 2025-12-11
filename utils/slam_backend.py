import os
import random
import time
import math

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
        self.pose_graph_request_id_counter = 0
        self.pose_graph_pending_request = None

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
        self.pose_graph_optimize_every_n_keyframes = pose_graph_cfg.get(
            "optimize_every_n_keyframes", None
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
        self.pose_graph_request_id = 0
        self.pose_graph_pending = False
        self.pose_graph_keyframe_counter = 0
        self.pose_graph_extra_odom_offsets = pose_graph_cfg.get(
            "extra_odometry_offsets", []
        )
        self.motion_prior_weight_base = self.config["Training"].get(
            "motion_prior_weight", 0.0
        )
        self.motion_prior_weight = self.motion_prior_weight_base
        self._motion_log_last = 0.0
        self.motion_vel_weight = self.config["Training"].get(
            "motion_vel_weight", 0.05
        )
        self.motion_yaw_weight = self.config["Training"].get(
            "motion_yaw_weight", 0.05
        )
        self.motion_rot_weight = self.config["Training"].get(
            "motion_rot_weight", 0.0
        )
        self.scale_prior_weight = self.config["Training"].get(
            "scale_prior_weight", 0.0
        )
        self.scale_prior_target = self.config["Training"].get(
            "scale_prior_target", 0.1
        )
        self._scale_log_last = 0.0
        self.loop_sim3_max_span = self.config["Training"].get(
            "loop_sim3_max_span", 60
        )
        self.epi_weight = self.config["Training"].get("epi_weight", 0.0)
        self._epi_points = torch.tensor(
            [
                [-0.5, -0.5, 1.0],
                [-0.5, 0.0, 1.0],
                [-0.5, 0.5, 1.0],
                [0.0, -0.5, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.5, 1.0],
                [0.5, -0.5, 1.0],
                [0.5, 0.0, 1.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=torch.float32,
        )
        self.kf_accept = self.config["Training"].get(
            "kf_acceptance",
            {
                "scale_min": 0.5,
                "scale_max": 2.0,
                "yaw_deg_max": 45.0,
                "straightness_min": 0.6,
                "loss_growth_max": 1.1,
                "curvature_deg_max": 75.0,
            },
        )
        save_dir = self.config["Results"].get("save_dir")
        self.loop_closure_viz_dir = None
        if save_dir:
            self.loop_closure_viz_dir = os.path.join(save_dir, "loop_closures")
            mkdir_p(self.loop_closure_viz_dir)
        loop_cfg = self.config.get("LoopClosure", {})
        self.loop_max_ate_increase = loop_cfg.get("max_ate_increase", 0.0)
        self.loop_min_ate_improvement = loop_cfg.get("min_ate_improvement", None)
        self.loop_enable_rejection = loop_cfg.get("enable_rejection", True)

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
        Log(
            f"[SIM3] dist_prev={dist_prev.item():.3f} dist_target={dist_target.item():.3f} raw_scale={scale.item():.3f}"
        )
        return sim3

    def _select_loop_submap(self, anchor_idx, query_idx):
        if len(self.viewpoints) == 0:
            return []
        ids = sorted(self.viewpoints.keys())
        max_span = self.loop_sim3_max_span
        if query_idx >= anchor_idx:
            return [
                idx for idx in ids if idx >= query_idx and (idx - query_idx) <= max_span
            ]
        return [
            idx for idx in ids if idx <= query_idx and (query_idx - idx) <= max_span
        ]

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
            rot_trace = (np.trace(T_delta[:3, :3]) - 1) / 2.0
            rot_trace = np.clip(rot_trace, -1.0, 1.0)
            rotation_rad = np.arccos(rot_trace)
            rotation_deg = np.degrees(rotation_rad)
            Log(
                f"[PG-UPDATE] kf={kf_id} Δt={translation:.3f} m ΔR={rotation_deg:.2f} deg"
            )
            if (
                self.pose_graph_max_translation_delta is not None
                and translation > self.pose_graph_max_translation_delta
            ):
                Log(
                    f"Skipping pose update for kf {kf_id} (translation delta {translation:.3f} m)"
                )
                continue
            if (
                self.pose_graph_max_rotation_delta is not None
                and rotation_rad > self.pose_graph_max_rotation_delta
            ):
                Log(
                    f"Skipping pose update for kf {kf_id} (rotation delta {rotation_deg:.2f} deg)"
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

    def _angle_diff_deg(self, a, b):
        diff = a - b
        while diff > 180.0:
            diff -= 360.0
        while diff < -180.0:
            diff += 360.0
        return diff

    def _rotation_log(self, R):
        trace = torch.clamp((torch.trace(R) - 1.0) * 0.5, -0.999999, 0.999999)
        theta = torch.acos(trace)
        if torch.abs(theta) < 1e-6:
            return torch.zeros(3, device=R.device, dtype=R.dtype)
        skew = (R - R.transpose(0, 1)) / (2.0 * torch.sin(theta))
        return theta * torch.tensor(
            [skew[2, 1], skew[0, 2], skew[1, 0]], device=R.device, dtype=R.dtype
        )

    def _camera_center(self, view):
        R = view.R.to(dtype=self.dtype)
        T = view.T.to(dtype=self.dtype)
        return -R.transpose(0, 1) @ T

    def _skew(self, v):
        s = torch.zeros((3, 3), device=v.device, dtype=v.dtype)
        s[0, 1] = -v[2]
        s[0, 2] = v[1]
        s[1, 0] = v[2]
        s[1, 2] = -v[0]
        s[2, 0] = -v[1]
        s[2, 1] = v[0]
        return s

    def _epipolar_residual(self, prev_view, curr_view):
        if self.epi_weight <= 0:
            return None
        R_prev = prev_view.R.to(dtype=self.dtype)
        R_curr = curr_view.R.to(dtype=self.dtype)
        C_prev = self._camera_center(prev_view)
        C_curr = self._camera_center(curr_view)
        baseline_world = C_curr - C_prev
        t_rel = R_prev @ baseline_world
        norm_t = torch.norm(t_rel)
        if norm_t < 1e-6:
            return None
        t_rel_unit = t_rel / norm_t
        R_rel = R_curr @ R_prev.transpose(0, 1)
        E = self._skew(t_rel_unit) @ R_rel
        pts = self._epi_points.to(E.device)
        residuals = []
        for pt in pts:
            x1 = pt
            x2 = pt
            val = torch.matmul(x2, torch.matmul(E, x1))
            residuals.append(val * val)
        if not residuals:
            return None
        return torch.stack(residuals).mean()

    def _get_position_np(self, viewpoint, use_gt=False):
        tensor = None
        if use_gt and hasattr(viewpoint, "T_gt") and viewpoint.T_gt is not None:
            tensor = viewpoint.T_gt
        else:
            tensor = viewpoint.T
        if tensor is None:
            return None
        return tensor.detach().cpu().numpy()

    def _log_trajectory_diagnostics(self, cur_frame_idx, window=5):
        sorted_ids = sorted(self.viewpoints.keys())
        if cur_frame_idx not in sorted_ids:
            return
        idx = sorted_ids.index(cur_frame_idx)
        if idx < 1:
            return
        cur_view = self.viewpoints[cur_frame_idx]
        diag = {}
        prev_id = sorted_ids[idx - 1]
        prev_view = self.viewpoints.get(prev_id)
        if prev_view is None:
            return
        p_curr = self._get_position_np(cur_view)
        p_prev = self._get_position_np(prev_view)
        if p_curr is None or p_prev is None:
            return
        delta_curr = p_curr - p_prev
        dist_curr = np.linalg.norm(delta_curr)
        if idx >= 2:
            prevprev_id = sorted_ids[idx - 2]
            prevprev_view = self.viewpoints.get(prevprev_id)
            if prevprev_view is not None:
                p_prevprev = self._get_position_np(prevprev_view)
                if p_prevprev is not None:
                    delta_prev = p_prev - p_prevprev
                    dist_prev = np.linalg.norm(delta_prev)
                    if dist_prev > 1e-6 and dist_curr > 1e-6:
                        scale_ratio = dist_curr / dist_prev
                        cosang = np.dot(delta_prev, delta_curr) / (
                            dist_prev * dist_curr
                        )
                        cosang = float(np.clip(cosang, -1.0, 1.0))
                        curvature = math.degrees(math.acos(cosang))
                        Log(
                            f"[DIAG] scale_ratio={scale_ratio:.3f} curvature={curvature:.2f} deg"
                        )
                        diag["scale_ratio"] = scale_ratio
                        diag["curvature_deg"] = curvature
        heading = math.degrees(math.atan2(delta_curr[1], delta_curr[0]))
        p_prev_gt = self._get_position_np(prev_view, use_gt=True)
        p_curr_gt = self._get_position_np(cur_view, use_gt=True)
        if p_prev_gt is not None and p_curr_gt is not None:
            delta_gt = p_curr_gt - p_prev_gt
            if np.linalg.norm(delta_gt) > 1e-6:
                heading_gt = math.degrees(math.atan2(delta_gt[1], delta_gt[0]))
                yaw_drift = self._angle_diff_deg(heading, heading_gt)
                Log(f"[DIAG] yaw_drift={yaw_drift:.2f} deg")
                diag["yaw_drift"] = yaw_drift
        start_idx = max(0, idx - (window - 1))
        path_ids = sorted_ids[start_idx : idx + 1]
        path_len = 0.0
        for i in range(1, len(path_ids)):
            p_a = self._get_position_np(self.viewpoints[path_ids[i - 1]])
            p_b = self._get_position_np(self.viewpoints[path_ids[i]])
            if p_a is None or p_b is None:
                continue
            path_len += np.linalg.norm(p_b - p_a)
        if path_len > 1e-6:
            p_start = self._get_position_np(self.viewpoints[path_ids[0]])
            if p_start is not None:
                displacement = np.linalg.norm(p_curr - p_start)
                straightness = displacement / path_len
                Log(f"[DIAG] straightness={straightness:.3f}")
                diag["straightness"] = straightness
        diag["step_distance"] = float(dist_curr)
        cur_view.diag_metrics = diag

    def _metrics_allow_edge(self, metrics):
        if not metrics:
            return True
        cfg = self.kf_accept
        scale_ratio = metrics.get("scale_ratio")
        if scale_ratio is not None:
            if scale_ratio < cfg.get("scale_min", 0.5) or scale_ratio > cfg.get(
                "scale_max", 2.0
            ):
                return False
        straightness = metrics.get("straightness")
        if straightness is not None and straightness < cfg.get(
            "straightness_min", 0.6
        ):
            return False
        curvature = metrics.get("curvature_deg")
        curvature_max = cfg.get("curvature_deg_max", None)
        if curvature is not None and curvature_max is not None:
            if curvature > curvature_max:
                return False
        return True

    def _motion_prior_loss(self, window_ids):
        if self.motion_prior_weight <= 0 or len(window_ids) < 3:
            return None
        loss = 0.0
        vel_weight = self.motion_vel_weight
        yaw_weight = self.motion_yaw_weight
        scale_weight = 0.02
        latest_mag = None
        for idx in range(2, len(window_ids)):
            v0 = self.viewpoints.get(window_ids[idx - 2])
            v1 = self.viewpoints.get(window_ids[idx - 1])
            v2 = self.viewpoints.get(window_ids[idx])
            if v0 is None or v1 is None or v2 is None:
                continue
            p0 = v0.T
            p1 = v1.T
            p2 = v2.T
            delta1 = p1 - p0
            delta2 = p2 - p1
            acc = delta2 - delta1
            loss = loss + (acc * acc).sum()
            vel_res = delta2 - delta1.detach()
            loss = loss + vel_weight * (vel_res * vel_res).sum()
            mag_prev = torch.norm(delta1.detach())
            mag_curr = torch.norm(delta2)
            latest_mag = mag_curr
            loss = loss + scale_weight * (mag_curr - mag_prev).pow(2)
            if (
                torch.norm(delta1[:2]).item() > 1e-6
                and torch.norm(delta2[:2]).item() > 1e-6
            ):
                yaw1 = torch.atan2(delta1[1], delta1[0])
                yaw2 = torch.atan2(delta2[1], delta2[0])
                yaw_diff = torch.atan2(
                    torch.sin(yaw2 - yaw1), torch.cos(yaw2 - yaw1)
                )
                loss = loss + yaw_weight * yaw_diff.pow(2)
            if self.motion_rot_weight > 0:
                R_prev = v1.R.to(dtype=self.dtype)
                R_curr = v2.R.to(dtype=self.dtype)
                R_rel = R_curr @ R_prev.transpose(0, 1)
                rotvec = self._rotation_log(R_rel)
                loss = loss + self.motion_rot_weight * (rotvec * rotvec).sum()
            if self.epi_weight > 0:
                epi = self._epipolar_residual(v1, v2)
                if epi is not None:
                    loss = loss + self.epi_weight * epi
        if isinstance(loss, float):
            return None
        adaptive_weight = self.motion_prior_weight_base
        if latest_mag is not None and self.motion_prior_weight_base > 0:
            adaptive_weight = self.motion_prior_weight_base * (
                1.0 + min(latest_mag.item() / 0.05, 5.0)
            )
        value = adaptive_weight * loss
        now = time.time()
        if now - self._motion_log_last >= 5.0:
            Log(f"[MOTION] prior loss={value.item():.6f}")
            self._motion_log_last = now
        return value

    def _request_pose_graph_opt(self, fixed_ids=None):
        if self.frontend_queue is None or self.pose_graph is None:
            return
        has_loop = any(edge["type"] == "loop" for edge in self.pose_graph.edges)
        if not has_loop:
            Log("PoseGraph: skipping optimization (no loop edges)")
            return
        nodes_payload = {
            int(k): np.asarray(v).copy() for k, v in self.pose_graph.nodes.items()
        }
        edges_payload = []
        for edge in self.pose_graph.edges:
            edges_payload.append(
                {
                    "type": edge["type"],
                    "i": int(edge["i"]),
                    "j": int(edge["j"]),
                    "measurement": np.asarray(edge["measurement"]).copy(),
                }
            )
        request_id = self.pose_graph_request_id_counter + 1
        self.pose_graph_request_id_counter = request_id
        self.pose_graph_pending_request = request_id
        self.pose_graph_pending = True
        payload = [
            "pose_graph_opt",
            request_id,
            nodes_payload,
            edges_payload,
            list(fixed_ids or []),
        ]
        self.frontend_queue.put(payload)

    def _scale_regularization_loss(self, window_ids):
        if self.scale_prior_weight <= 0 or len(window_ids) < 2:
            return None
        loss = 0.0
        count = 0
        prev = None
        for kf_id in window_ids:
            viewpoint = self.viewpoints.get(kf_id)
            if viewpoint is None:
                continue
            position = viewpoint.T
            if prev is not None:
                baseline = torch.norm(position - prev)
                target = torch.tensor(
                    self.scale_prior_target,
                    dtype=baseline.dtype,
                    device=baseline.device,
                )
                loss = loss + (baseline - target).pow(2)
                count += 1
            prev = position
        if count == 0:
            return None
        value = self.scale_prior_weight * loss / count
        now = time.time()
        if now - self._scale_log_last >= 5.0:
            Log(f"[SCALE] prior loss={value.item():.6f}")
            self._scale_log_last = now
        return value

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
            motion_loss = self._motion_prior_loss(current_window)
            if motion_loss is not None:
                loss_mapping += motion_loss
            scale_loss = self._scale_regularization_loss(current_window)
            if scale_loss is not None:
                loss_mapping += scale_loss
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
        metric_scale = event.get("metric_scale", self.config.get("LoopClosure", {}).get("metric_scale", True))
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
            T_loop_np = T_query_target.detach().cpu().numpy()
            t_norm = np.linalg.norm(T_loop_np[:3, 3])
            rot_trace = (np.trace(T_loop_np[:3, :3]) - 1) / 2.0
            rot_trace = np.clip(rot_trace, -1.0, 1.0)
            rot_deg = np.degrees(np.arccos(rot_trace))
            Log(
                f"[LOOP] {anchor_idx}->{query_idx} Δt={t_norm:.3f} m ΔR={rot_deg:.2f} deg metric_scale={metric_scale}"
            )
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
            if should_optimize and not self.pose_graph_pending:
                fixed_ids = (
                    set(int(k) for k in self.current_window)
                    if self.pose_graph_freeze_active_window
                    else None
                )
                self._request_pose_graph_opt(fixed_ids=fixed_ids)

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
                    if self.initialized:
                        self._log_trajectory_diagnostics(cur_frame_idx)
                    if self.use_pose_graph:
                        T_cur = self._rt_to_matrix(viewpoint.R, viewpoint.T)
                        self.pose_graph.add_node(cur_frame_idx, T_cur)
                        prev_kf = None
                        if self.pose_graph.nodes:
                            ordered = sorted(int(k) for k in self.pose_graph.nodes.keys())
                            try:
                                idx = ordered.index(int(cur_frame_idx))
                                if idx > 0:
                                    prev_kf = ordered[idx - 1]
                            except ValueError:
                                if ordered:
                                    prev_kf = ordered[-1]
                        if prev_kf is not None and prev_kf in self.viewpoints:
                            prev_view = self.viewpoints[prev_kf]
                            T_prev = self._rt_to_matrix(prev_view.R, prev_view.T)
                            T_rel = torch.linalg.inv(T_prev) @ T_cur
                            metrics_ok = self._metrics_allow_edge(
                                getattr(viewpoint, "diag_metrics", None)
                            )
                            if metrics_ok:
                                self.pose_graph.add_odometry_edge(
                                    prev_kf, cur_frame_idx, T_rel
                                )
                            else:
                                Log(
                                    f"PoseGraph: skipping odom edge {prev_kf}->{cur_frame_idx} due to metrics"
                                )
                        for offset in self.pose_graph_extra_odom_offsets:
                            neighbor_idx = cur_frame_idx - offset
                            if neighbor_idx in self.viewpoints:
                                neighbor_view = self.viewpoints[neighbor_idx]
                                T_neigh = self._rt_to_matrix(
                                    neighbor_view.R, neighbor_view.T
                                )
                                T_rel = torch.linalg.inv(T_neigh) @ T_cur
                                metrics_ok = self._metrics_allow_edge(
                                    getattr(viewpoint, "diag_metrics", None)
                                )
                                if metrics_ok:
                                    self.pose_graph.add_odometry_edge(
                                        neighbor_idx, cur_frame_idx, T_rel
                                    )
                                else:
                                    Log(
                                        f"PoseGraph: skipping offset edge {neighbor_idx}->{cur_frame_idx} due to metrics"
                                    )
                        if self.pose_graph_optimize_every_n_keyframes is not None:
                            self.pose_graph_keyframe_counter += 1
                            every = max(1, self.pose_graph_optimize_every_n_keyframes)
                            if (
                                self.pose_graph_keyframe_counter % every == 0
                                and not self.pose_graph_pending
                            ):
                                fixed_ids = (
                                    set(int(k) for k in self.current_window)
                                    if self.pose_graph_freeze_active_window
                                    else None
                                )
                                self._request_pose_graph_opt(fixed_ids=fixed_ids)

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
                elif data[0] == "pose_graph_result":
                    request_id = data[1]
                    optimized = data[2]
                    fixed_ids = set(data[3]) if len(data) > 3 and data[3] is not None else None
                    if request_id == self.pose_graph_pending_request:
                        self.pose_graph_pending = False
                        self.pose_graph_pending_request = None
                    if optimized:
                        if self.pose_graph:
                            self.pose_graph.nodes = {
                                int(k): np.asarray(v) for k, v in optimized.items()
                            }
                        self._apply_global_pose_corrections(
                            optimized, exclude_ids=fixed_ids
                        )
                        ate_pg = self._estimate_current_ate()
                        if ate_pg is not None:
                            Log(f"[ATE] after pose graph = {ate_pg:.3f} m")
                    else:
                        Log("PoseGraph: received empty optimization result")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
    def _loop_accepts(self, ate_before, ate_after):
        if not self.loop_enable_rejection:
            return True
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
