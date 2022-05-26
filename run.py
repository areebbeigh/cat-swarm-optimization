from src.cso import CatSwarmOptimization
from src.test_function import (
    easom_function,
    bukin_function,
    sphere_function,
    ackley_function,
)
from src.animation import animation3D

LOWER_BOUND = -10
UPPER_BOUND = 10
test_functions = [
    sphere_function,
    easom_function,
    bukin_function,
    ackley_function,
]

for function in test_functions:
    algo = CatSwarmOptimization(
        test_function=function,
        n=100,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        dimensions=2,
        iterations=100,
        mixture_ratio=0.05,
        seeking_mem_pool_size=50,
        self_position_considering=False,
        dimensions_to_change=1,
        dimension_seeking_range=2,
        max_velocity=0.5,
    )

    res = algo.result()
    print(f"{function.__name__} median result:")
    print(f"point={res} f(x,y)={function(res)}")
    print("Close animation to see next.\n\n")
    animation3D(
        algo.snapshots,
        function,
        LOWER_BOUND,
        UPPER_BOUND,
    )
