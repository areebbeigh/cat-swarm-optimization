from copy import deepcopy
import numpy as np


class CatSwarmOptimization:
    SEEKING_MODE = 0
    TRACING_MODE = 1

    def __init__(
        self,
        test_function,
        n: int,
        lower_bound: int,
        upper_bound: int,
        dimensions: int,
        iterations: int,
        # no_of_tracing / no_of_seeking
        mixture_ratio: float,
        seeking_mem_pool_size: int,
        self_position_considering: bool,
        dimensions_to_change: int,
        dimension_seeking_range: float,
        max_velocity: float,
    ):
        self.test_function = test_function
        self.no_of_agents = n
        self.mixture_ratio = mixture_ratio
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.seeking_mem_pool_size = seeking_mem_pool_size
        self.self_position_considering = self_position_considering
        self.dimensions_to_change = dimensions_to_change
        self.dimension_seeking_range = dimension_seeking_range
        self.max_velocity = max_velocity
        self.velocity_update_constant = 2

        # Initialize cats, velocities, and seeking/tracing modes
        self.agents = np.random.uniform(lower_bound, upper_bound, (n, dimensions))
        self.velocities = np.zeros((n, dimensions))

        # Run iterations
        self.snapshots = []
        for itr in range(iterations):
            modes = self._assign_modes()
            self.current_best_agent_idx = self._get_best_agent_idx()
            self.snapshots.append(deepcopy(self.agents))

            for idx, agent in enumerate(self.agents):
                if modes[idx] == self.SEEKING_MODE:
                    self._run_seeking_mode(idx, agent)
                else:
                    self._run_tracing_mode(idx, agent)

        self.current_best_agent_idx = self._get_best_agent_idx()
        self.snapshots.append(deepcopy(self.agents))

    def result(self):
        return list(np.median(self.agents, axis=0))

    def _run_seeking_mode(self, agent_idx: int, agent: np.ndarray):
        # Create seeking pool
        seeking_pool = np.array([agent for _ in range(self.seeking_mem_pool_size)])
        for cat in seeking_pool:
            # Select dimension indexes to change
            dim_indices = np.random.choice(
                list(range(self.dimensions)),
                size=self.dimensions_to_change,
                replace=False,
            )
            for idx in dim_indices:
                # Add or subtract a percentage of seeking_range
                new_dim = cat[idx] + (
                    self.dimension_seeking_range
                    * np.random.random()
                    * np.random.choice([-1, 1])
                )
                cat[idx] = new_dim

        if self.self_position_considering:
            seeking_pool[-1] = agent.copy()

        fitness_values = [self.test_function(p) for p in seeking_pool]

        # Calculate selection probabilities
        if np.equal(fitness_values, fitness_values[0]).all():
            probabilities = [1] * self.seeking_mem_pool_size
        else:
            probabilities = []
            fmax = max(fitness_values)
            fmin = min(fitness_values)
            for fitness_value in fitness_values:
                probabilities.append(np.abs(fitness_value - fmax) / (fmax - fmin))

        probabilities = np.array(probabilities) / sum(probabilities)
        idx = np.random.choice(range(len(seeking_pool)), p=probabilities)
        self.agents[agent_idx] = seeking_pool[idx]

    def _run_tracing_mode(self, agent_idx: int, agent: np.ndarray):
        velocity = self.velocities[agent_idx]
        best_agent: np.ndarray = self.agents[self.current_best_agent_idx]
        # Update all velocity dimensions
        for idx, vd in enumerate(velocity):
            velocity[idx] = vd + np.random.random() * self.velocity_update_constant * (
                best_agent[idx] - agent[idx]
            )
            if velocity[idx] > self.max_velocity:
                velocity[idx] = self.max_velocity
            if velocity[idx] < np.negative(self.max_velocity):
                velocity[idx] = np.negative(self.max_velocity)
        # Update agent position
        for idx, _ in enumerate(agent):
            agent[idx] += velocity[idx]
        self.agents[agent_idx] = agent

    def _get_best_agent_idx(self):
        return np.array([self.test_function(p) for p in self.agents]).argmin()

    # Assign tracing/seeking modes to cats account to mixture_ratio
    def _assign_modes(self):
        no_of_seeking = self.no_of_agents / (1 + self.mixture_ratio)
        no_of_tracing = self.no_of_agents - no_of_seeking

        modes = np.zeros(self.no_of_agents)
        for i in range(self.no_of_agents):
            mode = np.random.choice(
                [
                    self.SEEKING_MODE,
                    self.TRACING_MODE,
                ],
                p=[
                    no_of_seeking / self.no_of_agents,
                    no_of_tracing / self.no_of_agents,
                ],
            )
            modes[i] = mode
        return modes
