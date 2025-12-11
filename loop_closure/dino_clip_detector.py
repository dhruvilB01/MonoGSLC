import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from scripts.feature_encoder import get_model_and_transform


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / norm


def signed_power(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return np.sign(x) * (np.abs(x) ** alpha)


@dataclass
class LoopClosureDetection:
    anchor_idx: int
    query_idx: int
    score: float
    inliers: int
    inlier_ratio: float
    good_matches: int
    rel_pose: np.ndarray  # 4x4 camera->camera transform delta
    metric_scale: bool = True

    def as_message(self) -> Dict:
        return {
            "anchor_idx": int(self.anchor_idx),
            "query_idx": int(self.query_idx),
            "score": float(self.score),
            "inliers": int(self.inliers),
            "inlier_ratio": float(self.inlier_ratio),
            "good_matches": int(self.good_matches),
            "rel_pose": self.rel_pose.tolist(),
            "metric_scale": bool(self.metric_scale),
        }


class MultiModelBank:
    """
    Handles feature extraction and storage for multiple encoders (DINO, CLIP, ...).
    """

    def __init__(
        self,
        model_names: Sequence[str],
        weights: Sequence[float],
        device: Optional[str] = None,
        use_powerlaw: bool = True,
        tta_flip: bool = True,
    ) -> None:
        assert len(model_names) == len(weights)
        self.use_powerlaw = use_powerlaw
        self.tta_flip = tta_flip
        w = np.array(weights, dtype=np.float32)
        self.weights = w / (w.sum() + 1e-8)
        self.models = []
        self.transforms = []
        self.device = device
        for name in model_names:
            model, transform, dev, _ = get_model_and_transform(name, device=device)
            self.models.append(model)
            self.transforms.append(transform)
            self.device = dev

    def encode(self, image_rgb: np.ndarray) -> List[np.ndarray]:
        """
        Args:
            image_rgb: uint8 RGB array (H,W,3)
        Returns:
            List of L2-normalized feature vectors, one per model.
        """
        import torch

        pil_img = Image.fromarray(image_rgb)
        feats: List[np.ndarray] = []
        for model, transform in zip(self.models, self.transforms):
            x = transform(pil_img).unsqueeze(0).to(self.device)
            if self.tta_flip:
                x_flip = transform(pil_img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(self.device)
                x = torch.cat([x, x_flip], dim=0)
            with torch.no_grad():
                f = model(x)
            f = f.mean(dim=0, keepdim=True)
            vec = f.detach().cpu().numpy().squeeze(0)
            if self.use_powerlaw:
                vec = signed_power(vec)
            feats.append(l2_normalize(vec.astype(np.float32)))
        return feats


class DinoClipLoopDetector:
    """
    Online loop-closure detector that uses DINOv2/CLIP descriptors for retrieval
    and ORB + PnP for geometric verification to recover a relative pose.
    """

    def __init__(self, config: Dict, intrinsics: Dict[str, float]) -> None:
        self.enabled = config.get("enabled", False)
        if not self.enabled:
            return
        model_names = config.get("models", ["dinov2"])
        weights = config.get("weights", [1.0])
        assert len(model_names) == len(weights), "LoopClosure weights must match models"
        self.threshold = float(config.get("threshold", 0.9))
        self.min_gap = int(config.get("min_gap", 100))
        self.cooldown = int(config.get("cooldown", 40))
        self.max_db = config.get("max_db", None)
        self.detector_stride = int(config.get("detector_stride", 1))
        self.use_powerlaw = config.get("use_powerlaw", True)
        self.tta_flip = config.get("tta_flip", True)
        self.bank = MultiModelBank(
            model_names=model_names,
            weights=weights,
            device=config.get("device", None),
            use_powerlaw=self.use_powerlaw,
            tta_flip=self.tta_flip,
        )
        orb_cfg = config.get("orb", {})
        self.orb = cv2.ORB_create(
            nfeatures=int(orb_cfg.get("nfeatures", 2000)),
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=15,
        )
        self.knn_match_ratio = float(orb_cfg.get("ratio_thresh", 0.75))
        self.min_inliers = int(orb_cfg.get("min_inliers", 40))
        self.min_inlier_ratio = float(orb_cfg.get("min_inlier_ratio", 0.5))
        self.ransac_thresh = float(orb_cfg.get("ransac_thresh", 2.0))
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        validation_cfg = config.get("validation", {})
        self.cycle_consistency_thresh = float(validation_cfg.get("cycle_ratio", 0.5))
        self.max_reproj_error = float(validation_cfg.get("max_reproj_error", 3.0))
        self.max_pose_condition = float(validation_cfg.get("max_pose_condition", 1e4))

        self.fx = float(intrinsics["fx"])
        self.fy = float(intrinsics["fy"])
        self.cx = float(intrinsics["cx"])
        self.cy = float(intrinsics["cy"])
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        self.entries: List[Dict] = []
        self.weights = np.array(weights, dtype=np.float32) / (np.sum(weights) + 1e-8)
        self.last_detection_idx = -10**9
        self.frame_count = 0

    def is_enabled(self) -> bool:
        return self.enabled

    def _compute_orb(self, gray: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None:
            desc = np.zeros((0, 32), dtype=np.uint8)
        return kps, desc

    def _candidate_indices(self, query_idx: int) -> List[int]:
        idxs = []
        for i, entry in enumerate(self.entries):
            if query_idx - entry["kf_idx"] >= self.min_gap:
                idxs.append(i)
        return idxs

    def _cycle_consistency_ratio(
        self,
        matches: List[cv2.DMatch],
        anchor_desc: np.ndarray,
        query_desc: np.ndarray,
    ) -> float:
        if (
            len(matches) == 0
            or anchor_desc is None
            or query_desc is None
            or len(anchor_desc) == 0
            or len(query_desc) == 0
        ):
            return 0.0
        anchor_best = {}
        reverse = self.matcher.knnMatch(anchor_desc, query_desc, k=1)
        for pair in reverse:
            if not pair:
                continue
            best = pair[0]
            anchor_best[best.queryIdx] = best.trainIdx
        mutual = 0
        for m in matches:
            if anchor_best.get(m.trainIdx, -1) == m.queryIdx:
                mutual += 1
        return mutual / max(1, len(matches))

    def _best_match(self, feats: List[np.ndarray], candidates: List[int]) -> Tuple[int, float]:
        if not candidates:
            return -1, -1.0
        best_idx = -1
        best_score = -math.inf
        for local_idx in candidates:
            entry = self.entries[local_idx]
            score = 0.0
            for w, f_query, f_db in zip(self.weights, feats, entry["feats"]):
                score += w * float(np.dot(f_query, f_db))
            if score > best_score:
                best_score = score
                best_idx = local_idx
        return best_idx, best_score

    @staticmethod
    def _backproject(
        kp: cv2.KeyPoint, depth_map: Optional[np.ndarray], fx: float, fy: float, cx: float, cy: float
    ) -> Optional[np.ndarray]:
        if depth_map is None:
            return None
        if isinstance(depth_map, list):
            depth_map = np.asarray(depth_map)
        x, y = kp.pt
        ix, iy = int(round(x)), int(round(y))
        if iy < 0 or iy >= depth_map.shape[0] or ix < 0 or ix >= depth_map.shape[1]:
            return None
        z_val = depth_map[iy, ix]
        if isinstance(z_val, np.ndarray):
            if z_val.size == 0:
                return None
            z_val = z_val.reshape(-1)[0]
        z = float(z_val)
        if not np.isfinite(z) or z <= 0:
            return None
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        return np.array([X, Y, z], dtype=np.float32)

    def _verify_with_essential(
        self,
        matches: List[cv2.DMatch],
        anchor: Dict,
        query_kp: List[cv2.KeyPoint],
        score: float,
        query_idx: int,
    ) -> Optional[LoopClosureDetection]:
        if len(matches) < max(self.min_inliers, 12):
            return None
        pts_anchor = np.asarray(
            [anchor["kp"][m.trainIdx].pt for m in matches], dtype=np.float32
        )
        pts_query = np.asarray(
            [query_kp[m.queryIdx].pt for m in matches], dtype=np.float32
        )
        if pts_anchor.shape[0] < max(self.min_inliers, 12):
            return None
        E, mask = cv2.findEssentialMat(
            pts_anchor,
            pts_query,
            focal=self.fx,
            pp=(self.cx, self.cy),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.ransac_thresh,
        )
        if E is None or mask is None:
            return None
        inlier_count = int(mask.sum())
        inlier_ratio = inlier_count / max(1, len(matches))
        if inlier_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            return None
        _, R, t, mask_pose = cv2.recoverPose(
            E,
            pts_anchor,
            pts_query,
            self.K,
            mask=mask,
        )
        if mask_pose is None:
            return None
        inlier_count = int(mask_pose.sum())
        inlier_ratio = inlier_count / max(1, len(matches))
        if inlier_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            return None
        rel_pose = np.eye(4, dtype=np.float32)
        rel_pose[:3, :3] = R
        rel_pose[:3, 3] = t.reshape(3)
        return LoopClosureDetection(
            anchor_idx=anchor["kf_idx"],
            query_idx=query_idx,
            score=score,
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            good_matches=len(matches),
            rel_pose=rel_pose,
            metric_scale=False,
        )

    def _verify_with_pnp(
        self,
        anchor: Dict,
        query_gray: np.ndarray,
        query_kp: List[cv2.KeyPoint],
        query_desc: np.ndarray,
        score: float,
        query_idx: int,
    ) -> Optional[LoopClosureDetection]:
        matches = self.matcher.knnMatch(query_desc, anchor["desc"], k=2)
        good = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < self.knn_match_ratio * n.distance:
                good.append(m)
        if len(good) < self.min_inliers:
            return None
        cycle_ratio = self._cycle_consistency_ratio(good, anchor["desc"], query_desc)
        if cycle_ratio < self.cycle_consistency_thresh:
            return None

        obj_points = []
        img_points = []
        for m in good:
            kp_anchor = anchor["kp"][m.trainIdx]
            kp_query = query_kp[m.queryIdx]
            pt3d = self._backproject(kp_anchor, anchor["depth"], self.fx, self.fy, self.cx, self.cy)
            if pt3d is None:
                continue
            obj_points.append(pt3d)
            img_points.append(kp_query.pt)

        if len(obj_points) < max(self.min_inliers, 12):
            return self._verify_with_essential(good, anchor, query_kp, score, query_idx)

        obj = np.asarray(obj_points, dtype=np.float32)
        img = np.asarray(img_points, dtype=np.float32)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            self.K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self.ransac_thresh,
        )
        if not success or inliers is None:
            return self._verify_with_essential(good, anchor, query_kp, score, query_idx)
        inlier_count = int(len(inliers))
        inlier_ratio = inlier_count / max(1, len(obj_points))
        if inlier_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            return None
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, None)
        proj = proj.reshape(-1, 2)
        repro = np.linalg.norm(proj - img, axis=1)
        rms = float(np.sqrt(np.mean(repro**2)))
        if not np.isfinite(rms) or rms > self.max_reproj_error:
            return None
        centered = obj - obj.mean(axis=0, keepdims=True)
        cov = centered.T @ centered
        cond = np.linalg.cond(cov) if cov.size else np.inf
        if not np.isfinite(cond) or cond > self.max_pose_condition:
            return None
        rel_pose = np.eye(4, dtype=np.float32)
        rel_pose[:3, :3] = R
        rel_pose[:3, 3] = t
        return LoopClosureDetection(
            anchor_idx=anchor["kf_idx"],
            query_idx=query_idx,
            score=score,
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            good_matches=len(good),
            rel_pose=rel_pose,
            metric_scale=True,
        )

    def register_keyframe(
        self,
        kf_idx: int,
        image_rgb: np.ndarray,
        depth_map: Optional[np.ndarray],
    ) -> Optional[Dict]:
        """
        Process a new keyframe.

        Args:
            kf_idx: global frame index.
            image_rgb: uint8 RGB image (H,W,3).
            depth_map: optional depth (meters).
        Returns:
            Loop detection dict ready to send through queues, or None.
        """
        if not self.enabled:
            return None
        if self.frame_count % max(1, self.detector_stride) != 0:
            self.frame_count += 1
            return None
        self.frame_count += 1

        feats = self.bank.encode(image_rgb)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        kp, desc = self._compute_orb(gray)

        detection = None
        candidate_ids = self._candidate_indices(kf_idx)
        if candidate_ids and (kf_idx - self.last_detection_idx) >= self.cooldown:
            match_idx, score = self._best_match(feats, candidate_ids)
            if match_idx >= 0 and score >= self.threshold:
                anchor = self.entries[match_idx]
                detection_obj = self._verify_with_pnp(anchor, gray, kp, desc, score, kf_idx)
                if detection_obj is not None:
                    self.last_detection_idx = kf_idx
                    detection = detection_obj.as_message()

        entry = {
            "kf_idx": kf_idx,
            "feats": feats,
            "gray": gray,
            "kp": kp,
            "desc": desc,
            "depth": depth_map.copy() if depth_map is not None else None,
        }
        self.entries.append(entry)
        if self.max_db is not None and len(self.entries) > self.max_db:
            self.entries.pop(0)
        return detection
