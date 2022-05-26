from src.cso import CatSwarmOptimization
from src.test_function import easom_function
from src.animation import animation3D

LOWER_BOUND = -10
UPPER_BOUND = 10
algo = CatSwarmOptimization(
    test_function=easom_function,
    n=50,
    lower_bound=LOWER_BOUND,
    upper_bound=UPPER_BOUND,
    dimensions=2,
    iterations=100,
    mixture_ratio=0.1,
    seeking_mem_pool_size=15,
    self_position_considering=False,
    dimensions_to_change=1,
    dimension_seeking_range=1.5,
    max_velocity=1,
)
animation3D(algo.snapshots, easom_function, LOWER_BOUND, UPPER_BOUND)
