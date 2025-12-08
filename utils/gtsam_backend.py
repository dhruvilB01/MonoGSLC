import numpy as np

try:
    import gtsam

    HAS_GTSAM = True
except Exception:
    gtsam = None
    HAS_GTSAM = False


def _mat_to_pose3(mat):
    R = mat[:3, :3]
    t = mat[:3, 3]
    qw = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    qy = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    qz = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    qx = np.copysign(qx, R[2, 1] - R[1, 2])
    qy = np.copysign(qy, R[0, 2] - R[2, 0])
    qz = np.copysign(qz, R[1, 0] - R[0, 1])
    quat_norm = np.linalg.norm([qw, qx, qy, qz])
    if quat_norm < 1e-12:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    else:
        qw /= quat_norm
        qx /= quat_norm
        qy /= quat_norm
        qz /= quat_norm
    rot = gtsam.Rot3.Quaternion(qw, qx, qy, qz)
    trans = gtsam.Point3(t[0], t[1], t[2])
    return gtsam.Pose3(rot, trans)


if HAS_GTSAM:

    class GTSAMPoseGraph:
        def __init__(self):
            self.graph = None
            self.initial = None
            self.result = None
            self.keys = []

        def build_from_pose_graph(self, nodes, edges, fixed_ids=None):
            self.graph = gtsam.NonlinearFactorGraph()
            self.initial = gtsam.Values()
            self.result = None
            self.keys = []
            fixed_ids = set(fixed_ids or [])

            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
            )
            odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.02, 0.02, 0.02, 0.002, 0.002, 0.002])
            )
            loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.1, 0.1, 0.1, 0.02, 0.02, 0.02])
            )

            min_node = None
            for node_id, Twc in nodes.items():
                pose = _mat_to_pose3(Twc)
                key = int(node_id)
                self.initial.insert(key, pose)
                self.keys.append(key)
                if min_node is None or key < min_node:
                    min_node = key

            if min_node is not None:
                pose = self.initial.atPose3(min_node)
                self.graph.add(gtsam.PriorFactorPose3(min_node, pose, prior_noise))
            for fixed_id in fixed_ids:
                if fixed_id in nodes and self.initial.exists(fixed_id):
                    pose = self.initial.atPose3(fixed_id)
                    self.graph.add(
                        gtsam.PriorFactorPose3(fixed_id, pose, prior_noise)
                    )

            for edge in edges:
                i = int(edge["i"])
                j = int(edge["j"])
                if i not in nodes or j not in nodes:
                    continue
                measurement = edge["measurement"]
                if edge["type"] == "loop":
                    Twc_anchor = nodes.get(i)
                    if Twc_anchor is None:
                        continue
                    measurement = np.linalg.inv(Twc_anchor) @ measurement
                pose_rel = _mat_to_pose3(measurement)
                if edge["type"] == "odom":
                    noise = odom_noise
                else:
                    kernel = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
                    noise = gtsam.noiseModel.Robust.Create(kernel, loop_noise)
                factor = gtsam.BetweenFactorPose3(i, j, pose_rel, noise)
                self.graph.add(factor)

        def optimize(self, iterations=50):
            if self.graph is None or self.initial is None:
                return False
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(iterations)
            optimizer = gtsam.LevenbergMarquardtOptimizer(
                self.graph, self.initial, params
            )
            try:
                self.result = optimizer.optimize()
                return True
            except Exception:
                self.result = None
                return False

        def get_optimized_poses(self):
            if self.result is None:
                return {}
            poses = {}
            for key in self.keys:
                try:
                    pose = self.result.atPose3(int(key))
                except RuntimeError:
                    continue
                T = pose.matrix()
                poses[int(key)] = T
            return poses


else:

    class GTSAMPoseGraph:
        def __init__(self):
            self.result = None

        def build_from_pose_graph(self, nodes, edges, fixed_ids=None):
            return False

        def optimize(self, iterations=50):
            return False

        def get_optimized_poses(self):
            return {}
