import time
import math

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from loop_closure.dino_clip_detector import DinoClipLoopDetector
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.pose_graph import PoseGraph
from utils.slam_utils import get_loss_tracking, get_median_depth


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1
        self.kf_accept = self.config["Training"].get(
            "kf_acceptance",
            {
                "scale_min": 0.5,
                "scale_max": 2.0,
                "yaw_deg_max": 45.0,
                "straightness_min": 0.6,
                "loss_growth_max": 1.1,
            },
        )
        self.last_tracking_loss = None

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        self.loop_detector = None

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.use_every_n_frames = max(
            1, int(self.config["Training"].get("frame_stride", 1))
        )
        self.single_thread = self.config["Training"]["single_thread"]
        loop_cfg = self.config.get("LoopClosure", {"enabled": False})
        if loop_cfg.get("enabled", False):
            intrinsics = {
                "fx": self.dataset.fx,
                "fy": self.dataset.fy,
                "cx": self.dataset.cx,
                "cy": self.dataset.cy,
            }
            self.loop_detector = DinoClipLoopDetector(loop_cfg, intrinsics)
        else:
            self.loop_detector = None

    def _get_camera_position(self, cam, use_gt=False):
        if cam is None:
            return None
        tensor = None
        if use_gt and hasattr(cam, "T_gt") and cam.T_gt is not None:
            tensor = cam.T_gt
        else:
            tensor = cam.T
        if tensor is None:
            return None
        return tensor.detach().cpu().numpy()

    def _compute_kf_metrics(self, cur_frame_idx, window=5):
        sorted_ids = sorted(self.cameras.keys())
        if cur_frame_idx not in sorted_ids:
            return {}
        idx = sorted_ids.index(cur_frame_idx)
        if idx < 1:
            return {}
        cam_curr = self.cameras[cur_frame_idx]
        cam_prev = self.cameras.get(sorted_ids[idx - 1])
        if cam_prev is None:
            return {}
        p_curr = self._get_camera_position(cam_curr)
        p_prev = self._get_camera_position(cam_prev)
        if p_curr is None or p_prev is None:
            return {}
        delta_curr = p_curr - p_prev
        dist_curr = np.linalg.norm(delta_curr)
        metrics = {}
        if idx >= 2:
            cam_prevprev = self.cameras.get(sorted_ids[idx - 2])
            if cam_prevprev is not None:
                p_prevprev = self._get_camera_position(cam_prevprev)
                if p_prevprev is not None:
                    delta_prev = p_prev - p_prevprev
                    dist_prev = np.linalg.norm(delta_prev)
                    if dist_prev > 1e-6 and dist_curr > 1e-6:
                        metrics["scale_ratio"] = dist_curr / dist_prev
                        cosang = np.dot(delta_prev, delta_curr) / (
                            dist_prev * dist_curr
                        )
                        cosang = float(np.clip(cosang, -1.0, 1.0))
                        metrics["curvature_deg"] = math.degrees(math.acos(cosang))
        heading = math.degrees(math.atan2(delta_curr[1], delta_curr[0])) if dist_curr > 1e-6 else 0.0
        p_prev_gt = self._get_camera_position(cam_prev, use_gt=True)
        p_curr_gt = self._get_camera_position(cam_curr, use_gt=True)
        if p_prev_gt is not None and p_curr_gt is not None:
            delta_gt = p_curr_gt - p_prev_gt
            if np.linalg.norm(delta_gt) > 1e-6:
                heading_gt = math.degrees(math.atan2(delta_gt[1], delta_gt[0]))
                diff = heading - heading_gt
                while diff > 180.0:
                    diff -= 360.0
                while diff < -180.0:
                    diff += 360.0
                metrics["yaw_drift"] = diff
        start_idx = max(0, idx - (window - 1))
        path_ids = sorted_ids[start_idx : idx + 1]
        path_len = 0.0
        for i in range(1, len(path_ids)):
            a = self._get_camera_position(self.cameras[path_ids[i - 1]])
            b = self._get_camera_position(self.cameras[path_ids[i]])
            if a is None or b is None:
                continue
            path_len += np.linalg.norm(b - a)
        if path_len > 1e-6:
            p_start = self._get_camera_position(self.cameras[path_ids[0]])
            if p_start is not None:
                metrics["straightness"] = np.linalg.norm(p_curr - p_start) / path_len
        return metrics

    def _should_accept_keyframe(self, cur_frame_idx, viewpoint):
        metrics = self._compute_kf_metrics(cur_frame_idx)
        cfg = self.kf_accept
        scale_ratio = metrics.get("scale_ratio")
        if scale_ratio is not None:
            if scale_ratio < cfg.get("scale_min", 0.5) or scale_ratio > cfg.get(
                "scale_max", 2.0
            ):
                Log(
                    f"Rejecting KF {cur_frame_idx}: scale_ratio={scale_ratio:.3f}"
                )
                return False
        yaw_drift = metrics.get("yaw_drift")
        if yaw_drift is not None:
            if abs(yaw_drift) > cfg.get("yaw_deg_max", 45.0):
                Log(f"Rejecting KF {cur_frame_idx}: yaw_drift={yaw_drift:.2f} deg")
                return False
        straightness = metrics.get("straightness")
        if straightness is not None and straightness < cfg.get(
            "straightness_min", 0.6
        ):
            Log(
                f"Rejecting KF {cur_frame_idx}: straightness={straightness:.3f}"
            )
            return False
        loss_growth_max = cfg.get("loss_growth_max", 1.1)
        if (
            hasattr(viewpoint, "tracking_loss")
            and viewpoint.tracking_loss is not None
            and self.last_tracking_loss is not None
        ):
            if viewpoint.tracking_loss > self.last_tracking_loss * loss_growth_max:
                Log(
                    f"Rejecting KF {cur_frame_idx}: tracking loss {viewpoint.tracking_loss:.4f} vs {self.last_tracking_loss:.4f}"
                )
                return False
        self.last_tracking_loss = getattr(viewpoint, "tracking_loss", None)
        return True

    def _maybe_run_loop_closure(self, cur_frame_idx, viewpoint):
        if self.loop_detector is None or not self.loop_detector.is_enabled():
            return
        color = (
            viewpoint.original_image.detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        color = (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
        depth = viewpoint.depth
        if isinstance(depth, torch.Tensor):
            depth_np = depth.detach().cpu().numpy()
        elif depth is None:
            depth_np = None
        else:
            depth_np = np.array(depth)
        detection = self.loop_detector.register_keyframe(cur_frame_idx, color, depth_np)
        if detection is not None:
            self.backend_queue.put(["loop_closure", detection])

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()
            viewpoint.tracking_loss = float(
                loss_tracking.detach().cpu().item()
            )

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        curr_frame_WC = torch.linalg.inv(pose_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        R_prev = last_kf_WC[:3, :3]
        R_curr = curr_frame_WC[:3, :3]
        R_rel = R_curr @ R_prev.transpose(0, 1)
        delta_yaw = math.degrees(math.atan2(R_rel[1, 0], R_rel[0, 0]))
        rotation_threshold = self.config["Training"].get("kf_rotation_deg", 1.0)
        rotation_check = abs(delta_yaw) > rotation_threshold

        center_prev = last_kf_WC[:3, 3]
        center_curr = curr_frame_WC[:3, 3]
        delta_world = center_curr - center_prev
        forward_dir = R_prev[:, 2]
        forward_motion = torch.dot(delta_world, forward_dir)
        forward_ratio = self.config["Training"].get("kf_forward_ratio", 0.1)
        forward_check = False
        if self.median_depth is not None and self.median_depth > 1e-6:
            forward_check = (
                forward_motion > forward_ratio * float(self.median_depth)
            )

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (
            (point_ratio_2 < kf_overlap and dist_check2)
            or dist_check
            or rotation_check
            or forward_check
        )

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += self.use_every_n_frames
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += self.use_every_n_frames
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                    self._maybe_run_loop_closure(cur_frame_idx, viewpoint)
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += self.use_every_n_frames

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
                elif data[0] == "pose_graph_opt":
                    self._handle_pose_graph_opt(data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return

    def _handle_pose_graph_opt(self, data):
        _, request_id, nodes, edges, fixed_ids = data
        pose_graph = PoseGraph()
        pose_graph.nodes = {int(k): np.asarray(v) for k, v in nodes.items()}
        pose_graph.edges = edges
        optimized = pose_graph.optimize(fixed_ids=set(fixed_ids or []))
        self.backend_queue.put(
            ["pose_graph_result", request_id, optimized, fixed_ids]
        )
