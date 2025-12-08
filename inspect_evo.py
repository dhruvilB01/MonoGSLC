import evo.core.trajectory
print("Attributes of evo.core.trajectory:")
print(dir(evo.core.trajectory))

try:
    from evo.core.trajectory import align_trajectory
    print("Found align_trajectory in evo.core.trajectory")
except ImportError:
    print("align_trajectory NOT found in evo.core.trajectory")
