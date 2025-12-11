import numpy as np
import torch

from utils.logging_utils import Log

try:
    from utils.g2o_backend import G2OPoseGraph, HAS_G2O
except Exception:
    G2OPoseGraph = None
    HAS_G2O = False

try:
    from utils.gtsam_backend import GTSAMPoseGraph, HAS_GTSAM
except Exception:
    GTSAMPoseGraph = None
    HAS_GTSAM = False


class PoseGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.optimized_nodes = {}
        self.backend = None
        self.backend_name = None
        self.backend_cls = None
        if HAS_G2O and G2OPoseGraph is not None:
            self.backend_cls = G2OPoseGraph
            self.backend_name = "g2o"
        elif HAS_GTSAM and GTSAMPoseGraph is not None:
            self.backend_cls = GTSAMPoseGraph
            self.backend_name = "gtsam"

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
        if int(i) == int(j):
            Log(f"[EDGE-REJECT] self-edge {i}->{j}")
            return
        trans = np.linalg.norm(Tij_np[:3, 3])
        Log(f"[PG] odom edge {i}->{j} | Δt={trans:.3f} m")
        MAX_EDGE_TRANS = 3.0
        if trans > MAX_EDGE_TRANS:
            Log(
                f"[EDGE-REJECT] odom edge {i}->{j} too long ({trans:.3f} m)"
            )
            return
        for edge in self.edges:
            if (
                edge["type"] == "odom"
                and int(edge["i"]) == int(i)
                and int(edge["j"]) == int(j)
            ):
                Log(f"[EDGE-DUP] duplicate odom {i}->{j}")
                return
        self.edges.append(
            {"type": "odom", "i": int(i), "j": int(j), "measurement": Tij_np}
        )

    def add_loop_edge(self, i, j, T_target):
        if T_target is None:
            return
        if isinstance(T_target, torch.Tensor):
            T_target_np = T_target.detach().cpu().numpy()
        else:
            T_target_np = np.asarray(T_target)
        anchor = self.nodes.get(int(i))
        if anchor is None:
            Log(f"PoseGraph: missing anchor {i} for loop edge")
            return
        measurement = np.linalg.inv(anchor) @ T_target_np
        self.edges.append(
            {"type": "loop", "i": int(i), "j": int(j), "measurement": measurement}
        )

    def optimize(self, fixed_ids=None):
        if not self._ensure_backend():
            Log("PoseGraph optimization skipped (no backend available)")
            return None
        if len(self.nodes) < 2:
            Log("PoseGraph optimization skipped (insufficient nodes)")
            return None
        Log(
            f"PoseGraph: optimizing with {self.backend_name} "
            f"(nodes={len(self.nodes)}, edges={len(self.edges)})"
        )
        valid_nodes = set(int(k) for k in self.nodes.keys())
        filtered_edges = []
        for e in self.edges:
            i, j = int(e["i"]), int(e["j"])
            if i not in valid_nodes or j not in valid_nodes:
                Log(
                    f"PoseGraph: dropping edge ({i}->{j}) missing node "
                    f"(i in graph? {i in valid_nodes}, j? {j in valid_nodes})"
                )
                continue
            measurement = e.get("measurement")
            if measurement is None or not np.all(np.isfinite(measurement)):
                Log(
                    f"PoseGraph: dropping edge ({i}->{j}) due to invalid measurement"
                )
                continue
            filtered_edges.append(e)
        removed = len(self.edges) - len(filtered_edges)
        if removed > 0:
            Log(f"PoseGraph: pruned {removed} dangling edges before optimization")
        self.edges = filtered_edges
        built = self.backend.build_from_pose_graph(
            self.nodes, self.edges, fixed_ids=fixed_ids
        )
        if not built:
            Log("PoseGraph optimization skipped (backend build failed)")
            return None
        success = False
        try:
            success = self.backend.optimize()
        except Exception as exc:
            Log(f"PoseGraph: backend optimize raised {exc}")
            self.backend = None
        if not success:
            Log("PoseGraph optimization failed.")
            return None
        self.optimized_nodes = self.backend.get_optimized_poses()
        if self.optimized_nodes:
            Log(
                f"PoseGraph: optimization updated {len(self.optimized_nodes)} poses"
            )
            for k, T_new in self.optimized_nodes.items():
                T_old = self.nodes.get(int(k))
                if T_old is None:
                    continue
                dT = T_new @ np.linalg.inv(T_old)
                trans = np.linalg.norm(dT[:3, 3])
                rot_trace = (np.trace(dT[:3, :3]) - 1) / 2.0
                rot_trace = np.clip(rot_trace, -1.0, 1.0)
                rot = np.degrees(np.arccos(rot_trace))
                Log(f"[PG-DELTA] kf={k} Δt={trans:.3f} m ΔR={rot:.2f} deg")
        if self.optimized_nodes:
            self.nodes = {int(k): v for k, v in self.optimized_nodes.items()}
        return self.optimized_nodes

    def _ensure_backend(self):
        if self.backend is not None:
            return True
        if self.backend_cls is None:
            return False
        try:
            self.backend = self.backend_cls()
            Log(f"PoseGraph: initialized backend {self.backend_name}")
            return True
        except Exception as exc:
            Log(f"Failed to initialize pose-graph backend: {exc}")
            self.backend = None
            return False
