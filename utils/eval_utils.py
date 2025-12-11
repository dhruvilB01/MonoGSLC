import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot

    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = PosePath3D(
        poses_se3=[pose.copy() for pose in traj_est.poses_se3]
    )
    traj_est_aligned.align(traj_ref, correct_scale=monocular)

    ## RMSEimport json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = PosePath3D(poses_se3=[pose.copy() for pose in traj_est.poses_se3])
    traj_est_aligned.align(traj_ref, correct_scale=monocular)

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
            os.path.join(plot_dir, "stats_{}.json".format(str(label))),
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig, ax = plt.subplots(figsize=(10, 8))  # Use subplots instead
    ax.set_title(f"ATE RMSE: {ape_stat}")
    ax.set_aspect('equal')

    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")

    # Plot trajectory with color mapping manually
    from matplotlib import cm
    import matplotlib.colors as mcolors

    # Get trajectory points
    xyz = traj_est_aligned.positions_xyz
    errors = ape_metric.error

    # Create colormap
    norm = mcolors.Normalize(vmin=ape_stats["min"], vmax=ape_stats["max"])
    cmap = cm.get_cmap('jet')

    # Plot with colors
    for i in range(len(xyz) - 1):
        color = cmap(norm(errors[i]))
        ax.plot(xyz[i:i + 2, 0], xyz[i:i + 2, 1], color=color, linewidth=2)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Error (m)')

    # Add legend entry for estimated trajectory
    ax.plot([], [], color='red', linewidth=2, label='estimated')
    ax.legend()

    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.close(fig)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    # Automatically handle full vs. partial GT
    results = evaluate_sequence_auto(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular
    )

    # Log whatever metrics were available
    wandb_data = {"frame_idx": latest_frame_idx}
    wandb_data.update({
        "ate": results.get("ate", None),
        "rpe": results.get("rpe", None),
        "start_end_alignment_error": results.get("start_end_alignment_error", None),
        "drift_ratio": results.get("drift_ratio", None),
    })
    wandb.log(wandb_data)

    return results


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in range(0, end_idx, interval):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    # Create plot without problematic colormap
    plot_mode = evo.tools.plot.PlotMode.xy
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "ground truth")
    evo.tools.plot.traj(ax, plot_mode, traj_est_aligned, "-", "blue", "estimated")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.close(fig)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)
        
    # Automatically handle full vs. partial GT
    results = evaluate_sequence_auto(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular
    )

    # Log whatever metrics were available
    wandb_data = {"frame_idx": latest_frame_idx}
    wandb_data.update({
        "ate": results.get("ate", None),
        "rpe": results.get("rpe", None),
        "start_end_alignment_error": results.get("start_end_alignment_error", None),
        "drift_ratio": results.get("drift_ratio", None),
    })
    wandb.log(wandb_data)

    return results


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in range(0, end_idx, interval):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


def evaluate_rpe(poses_gt, poses_est, plot_dir, label, delta=1):
    traj_ref = trajectory.PosePath3D(poses_se3=poses_gt)
    traj_est = trajectory.PosePath3D(poses_se3=poses_est)
    rpe_metric = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=delta,
        delta_unit=metrics.Unit.frames
    )
    rpe_metric.process_data((traj_ref, traj_est))
    rpe_rmse = rpe_metric.get_statistic(metrics.StatisticsType.rmse)
    Log(f"RMSE RPE [m]: {rpe_rmse}", tag="Eval")
    json.dump(
        rpe_metric.get_all_statistics(),
        open(os.path.join(plot_dir, f"rpe_stats_{label}.json"), "w"),
        indent=4
    )
    return rpe_rmse
    
def start_end_alignment_error(poses_gt, poses_est):
    start_est, end_est = poses_est[0][:3,3], poses_est[-1][:3,3]
    start_gt, end_gt   = poses_gt[0][:3,3], poses_gt[-1][:3,3]
    return float(np.linalg.norm((end_est - start_est) - (end_gt - start_gt)))

def drift_ratio(ate_rmse, poses_gt):
    total_length = sum(
        np.linalg.norm(poses_gt[i+1][:3,3] - poses_gt[i][:3,3])
        for i in range(len(poses_gt)-1)
    )
    return float(ate_rmse / total_length)

def evaluate_sequence_auto(poses_gt, poses_est, plot_dir, label, monocular=False):
    """
    Automatically select metrics based on ground truth availability.
    If full ground truth is provided, compute ATE + RPE.
    Otherwise, compute start-end alignment and drift ratio only.
    """
    # Require at least 2 GT poses
    if poses_gt is None or len(poses_gt) < 2:
        Log("No ground truth provided — skipping evaluation.", tag="Eval")
        return None

    # Check if we have full ground truth coverage (simple heuristic)
    full_gt = len(poses_gt) > len(poses_est) * 0.8
    results = {}

    if full_gt:
        # --- Full GT: ATE + RPE ---
        ate = evaluate_evo(
            poses_gt=poses_gt,
            poses_est=poses_est,
            plot_dir=plot_dir,
            label=label,
            monocular=monocular,
        )
        rpe = evaluate_rpe(poses_gt, poses_est, plot_dir, label)
        results.update({"ate": ate, "rpe": rpe})
    else:
        # --- Partial GT: Start-End + Drift Ratio ---
        start_end_err = start_end_alignment_error(poses_gt, poses_est)
        # Here we use start-end drift instead of ATE for ratio
        drift = start_end_err / sum(
            np.linalg.norm(poses_gt[i + 1][:3, 3] - poses_gt[i][:3, 3])
            for i in range(len(poses_gt) - 1)
        )
        results.update({
            "start_end_alignment_error": start_end_err,
            "drift_ratio": drift
        })
        Log(f"Start-End error [m]: {start_end_err:.4f}, Drift ratio: {drift:.6f}", tag="Eval")

    return results

