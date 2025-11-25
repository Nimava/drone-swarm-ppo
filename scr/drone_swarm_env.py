# Authors: Tzyy-Leng Horng, Nima Vaziri 
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulate_jungle_case import simulate_jungle_case  # Import the real simulation

class DroneSwarmEnv(gym.Env):
    def __init__(self):
        super(DroneSwarmEnv, self).__init__()

        # Action space: [cohesion, alignment, separation, leader-following, max_speed]
        self.action_space = spaces.Box(low=0.1, high=10.0, shape=(5,), dtype=np.float32)

        # Observation space: [Amin, Bmin, Cmax]
        self.observation_space = spaces.Box(low=0.0, high=20.0, shape=(3,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = 1  # Single step episodes for evaluation-style reward

    def _get_obs(self, Amin, Bmin, Cmax):
        return np.array([Amin, Bmin, Cmax], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = np.array([10.0, 10.0, 1.0], dtype=np.float32)  # Neutral starting obs
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # Clamp the action to avoid invalid inputs
        action = np.clip(action, self.action_space.low, self.action_space.high)
        k_c, k_a, k_s, k_l, max_speed = action.tolist()

        # Run the simulation and get metrics
        Amin, Bmin, Cmax, *_ = simulate_jungle_case(k_c, k_s, k_a, k_l, max_speed)

        # Fix: Convert from list to scalar if needed
        Amin = Amin[0] if isinstance(Amin, list) else Amin
        Bmin = Bmin[0] if isinstance(Bmin, list) else Bmin
        Cmax = Cmax[0] if isinstance(Cmax, list) else Cmax


        obs = self._get_obs(Amin, Bmin, Cmax)

        # Reward function: reward safe and cohesive swarm
        reward = 0
        if Amin > 1: reward += 10
        else: reward -= 20

        if Bmin > 1: reward += 10
        else: reward -= 20

        if Cmax < 10: reward += 5
        else: reward -= 5

        terminated = True  # One-shot simulation
        truncated = False
        info = {"Amin": Amin, "Bmin": Bmin, "Cmax": Cmax}
        print(f"Step: {self.current_step}, Action: {action}")

        return obs, reward, terminated, truncated, info
