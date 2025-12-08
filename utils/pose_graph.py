import numpy as np
import torch

from utils.gtsam_backend import GTSAMPoseGraph, HAS_GTSAM
from utils.logging_utils import Log


class PoseGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.optimized_nodes = {}
        self.backend = GTSAMPoseGraph() if HAS_GTSAM else None

    def add_node(self, kf_id, Twc):
        if Twc is None:
            return
        if isinstance(Twc, torch.Tensor):
            Twc_np = Twc.detach().cpu().numpy()
        else:
            Twc_np = np.asarray(Twc)
        self.nodes[int(kf_id)] = Twc_np

    def add_odometry_edge(self, i, j, Tij):
        if Tij is None:
            return
        if isinstance(Tij, torch.Tensor):
            Tij_np = Tij.detach().cpu().numpy()
        else:
            Tij_np = np.asarray(Tij)
        self.edges.append(
            {"type": "odom", "i": int(i), "j": int(j), "measurement": Tij_np}
        )

    def add_loop_edge(self, i, j, Tij):
        if Tij is None:
            return
        if isinstance(Tij, torch.Tensor):
            Tij_np = Tij.detach().cpu().numpy()
        else:
            Tij_np = np.asarray(Tij)
        self.edges.append(
            {"type": "loop", "i": int(i), "j": int(j), "measurement": Tij_np}
        )

    def optimize(self, fixed_ids=None):
        if not HAS_GTSAM or self.backend is None:
            Log("PoseGraph optimization skipped (gtsam unavailable)")
            return None
        if len(self.nodes) < 2:
            Log("PoseGraph optimization skipped (insufficient nodes)")
            return None
        self.backend.build_from_pose_graph(self.nodes, self.edges, fixed_ids=fixed_ids)
        success = self.backend.optimize()
        if not success:
            Log("PoseGraph optimization failed.")
            return None
        self.optimized_nodes = self.backend.get_optimized_poses()
        if self.optimized_nodes:
            self.nodes = {int(k): v for k, v in self.optimized_nodes.items()}
        return self.optimized_nodes
