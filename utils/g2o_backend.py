import numpy as np

try:
    import pyg2o as g2o

    HAS_G2O = True
except Exception:
    try:
        import g2o as g2o

        HAS_G2O = True
    except Exception:
        g2o = None
        HAS_G2O = False


def _mat_to_isometry(mat):
    """Convert 4x4 pose matrix to g2o.Isometry3d."""
    R = mat[:3, :3]
    t = mat[:3, 3]
    qw = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    qy = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    qz = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    qx = np.copysign(qx, R[2, 1] - R[1, 2])
    qy = np.copysign(qy, R[0, 2] - R[2, 0])
    qz = np.copysign(qz, R[1, 0] - R[0, 1])
    quat = np.array([qw, qx, qy, qz])
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        quat /= norm
    iso = g2o.Isometry3d()
    iso.set_rotation(R.astype(np.float64))
    iso.set_translation(t.astype(np.float64))
    return iso


if HAS_G2O:

    class G2OPoseGraph:
        def __init__(self):
            self.optimizer = None
            self.vertex_ids = []
            self.nodes = {}
            self.edges = []
            self.result = {}
            self.id_map = {}
            self.rev_id_map = {}

            self.base_algorithm = self._create_algorithm()
            if self.base_algorithm is None:
                raise RuntimeError("Failed to construct g2o optimization algorithm")
            self.odom_info = np.diag(
                [50.0, 50.0, 50.0, 100.0, 100.0, 100.0]
            )
            self.loop_info = np.diag(
                [500.0, 500.0, 500.0, 800.0, 800.0, 800.0]
            )
            self.odom_kernel_delta = 5.0
            self.loop_kernel_delta = 5.0

        def _create_algorithm(self):
            linear_solver = None
            for name in [
                "LinearSolverCholmodSE3",
                "LinearSolverEigenSE3",
                "LinearSolverDenseSE3",
                "LinearSolverPCGSE3",
            ]:
                solver_cls = getattr(g2o, name, None)
                if solver_cls is None:
                    continue
                try:
                    linear_solver = solver_cls()
                    break
                except Exception:
                    linear_solver = None
            if linear_solver is None:
                return None
            block_solver = g2o.BlockSolverSE3(linear_solver)
            return g2o.OptimizationAlgorithmLevenberg(block_solver)

        def build_from_pose_graph(self, nodes, edges, fixed_ids=None):
            self.optimizer = g2o.SparseOptimizer()
            self.optimizer.set_algorithm(self.base_algorithm)
            self.vertex_ids = []
            self.nodes = dict(nodes)
            self.edges = list(edges)
            fixed_ids = set(fixed_ids or [])
            if not self.nodes:
                return False

            ordered_ids = sorted(int(k) for k in self.nodes.keys())
            self.id_map = {node_id: idx for idx, node_id in enumerate(ordered_ids)}
            self.rev_id_map = {idx: node_id for node_id, idx in self.id_map.items()}
            mapped_fixed = {self.id_map[i] for i in fixed_ids if i in self.id_map}

            # Add vertices
            print(f"[g2o] Adding {len(self.nodes)} vertices")
            for node_id, pose in self.nodes.items():
                original_id = int(node_id)
                mapped_id = self.id_map.get(original_id)
                if mapped_id is None:
                    continue
                v = g2o.VertexSE3()
                v.set_id(mapped_id)
                iso = _mat_to_isometry(pose)
                if not np.all(np.isfinite(pose)):
                    print(f"[g2o] Skipping node {node_id} due to invalid pose")
                    continue
                v.set_estimate(iso)
                if mapped_id in mapped_fixed:
                    v.set_fixed(True)
                self.optimizer.add_vertex(v)
                self.vertex_ids.append(mapped_id)

            if not mapped_fixed and self.vertex_ids:
                first_id = min(self.vertex_ids)
                vertex = self.optimizer.vertex(first_id)
                if vertex is not None:
                    vertex.set_fixed(True)

            # Add edges
            print(f"[g2o] Adding {len(self.edges)} edges")
            for edge in self.edges:
                i = int(edge["i"])
                j = int(edge["j"])
                vi = self.id_map.get(i)
                vj = self.id_map.get(j)
                if vi is None or vj is None:
                    print(f"[g2o] Edge {i}->{j} dropped (missing vertex)")
                    continue
                if self.optimizer.vertex(vi) is None or self.optimizer.vertex(vj) is None:
                    print(f"[g2o] Edge {i}->{j} dropped (vertex object missing)")
                    continue
                measurement = edge["measurement"]
                e = g2o.EdgeSE3()
                e.set_vertex(0, self.optimizer.vertex(vi))
                e.set_vertex(1, self.optimizer.vertex(vj))
                if not np.all(np.isfinite(measurement)):
                    print(f"[g2o] Skipping edge {i}->{j} invalid measurement")
                    continue
                if measurement.shape != (4, 4):
                    print(f"[g2o] Skipping edge {i}->{j} bad shape {measurement.shape}")
                    continue
                detR = np.linalg.det(measurement[:3, :3])
                if detR <= 0:
                    print(f"[g2o] Skipping edge {i}->{j} det(R)={detR}")
                    continue
                print(f"[g2o] Edge {i}->{j} info: type={edge['type']}")
                e.set_measurement(_mat_to_isometry(measurement))
                if edge["type"] == "odom":
                    e.set_information(self.odom_info)
                    kernel = g2o.RobustKernelHuber()
                    kernel.set_delta(self.odom_kernel_delta)
                    e.set_robust_kernel(kernel)
                else:
                    e.set_information(self.loop_info)
                    kernel = g2o.RobustKernelHuber()
                    kernel.set_delta(self.loop_kernel_delta)
                    e.set_robust_kernel(kernel)
                self.optimizer.add_edge(e)
            return True

        def optimize(self, iterations=50):
            if self.optimizer is None:
                return False
            self.optimizer.initialize_optimization()
            success = self.optimizer.optimize(iterations)
            if success:
                try:
                    chi2 = self.optimizer.active_chi2()
                    print(f"[g2o] final chi2 = {chi2:.3f}")
                except Exception:
                    pass
            if not success:
                return False
            self.result = {}
            for vid in self.vertex_ids:
                vertex = self.optimizer.vertex(vid)
                if vertex is None:
                    continue
                est = vertex.estimate()
                T = np.eye(4)
                T[:3, :3] = est.rotation().matrix()
                T[:3, 3] = est.translation()
                original_id = self.rev_id_map.get(int(vid), int(vid))
                self.result[int(original_id)] = T
            return True

        def get_optimized_poses(self):
            return dict(self.result)


else:

    class G2OPoseGraph:
        def __init__(self):
            self.result = {}

        def build_from_pose_graph(self, nodes, edges, fixed_ids=None):
            return False

        def optimize(self, iterations=50):
            return False

        def get_optimized_poses(self):
            return {}
