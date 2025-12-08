import numpy as np
import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

from utils.eval_utils import evaluate_evo
from evo.core.trajectory import PoseTrajectory3D

# Create dummy poses
poses_gt = [np.eye(4) for _ in range(10)]
poses_est = [np.eye(4) for _ in range(10)]

# Add some noise to est
for i in range(10):
    poses_est[i][0, 3] += 0.1 * i

# Create dummy plot dir
os.makedirs("test_results", exist_ok=True)

print("Running evaluate_evo...")
try:
    evaluate_evo(poses_gt, poses_est, "test_results", "test", monocular=False)
    print("evaluate_evo passed!")
except Exception as e:
    print(f"evaluate_evo failed: {e}")
    import traceback
    traceback.print_exc()
