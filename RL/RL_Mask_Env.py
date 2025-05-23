import numpy as np
import pandas as pd
import random
import gymnasium as gym
from gymnasium import spaces

from Single_Source.Coverage_Guided_Row_Selection import (
    algo_main,
    compute_overall_coverage,
    compute_overall_penalty,
    optimize_selection,
)

class DataSelectionEnvMaskable(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sources_list, user_request, statistics, value_index, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__()
        self.sources_list = sources_list
        self.max_steps = len(self.sources_list)
        self.UR = user_request
        self.gamma = gamma
        self.statistics = statistics
        self.alpha = alpha
        self.beta = beta
        self.value_index = value_index
        self.obs_len = 2 + len(sources_list) + len(sources_list) * len(value_index)

        self.num_sources = len(self.sources_list)
        self.stop_action = self.num_sources  # index of the stop action
        self.action_space = spaces.Discrete(self.num_sources + 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_len,), dtype=np.float32
        )

        # For compatibility with Gymnasium
        self._state = None
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.selected_sources = set()
        self.current_table = pd.DataFrame(columns=self.UR.columns.tolist() + ['Identifiant'])
        self.current_coverage, _ = compute_overall_coverage(self.current_table, self.UR)
        self.current_penalty, _ = compute_overall_penalty(self.current_table, self.UR)
        self.last_coverage = 0.0
        self.last_penalty = 0.0
        self.last_steps = 0
        self.steps_taken = 0

        obs = self.get_state()
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    def step(self, action):
        action = int(np.asarray(action).item())
        terminated = False
        truncated = False
        info = {}

        if action == self.stop_action:
            terminated = True
            if len(self.selected_sources) == 0:
                reward = -10  # Penalty for STOP as first action
            else:
                reward = self.compute_reward(self.current_coverage, self.current_penalty)
                if self.current_coverage >= 1.0:
                    reward += 50
            info = {
                "stop": True,
                "coverage": self.current_coverage,
                "penalty": self.current_penalty,
                "steps": self.steps_taken,
                "selected_sources": list(self.selected_sources),
                "action_mask": self.get_action_mask(),
            }
            obs = self.get_state()
            return obs, reward, terminated, truncated, info

        if action in self.selected_sources:
            # This branch should never be reached by MaskableDQN!
            reward = -10
            terminated = True
            self.steps_taken += 1
            info = {
                "stop": False,
                "coverage": self.current_coverage,
                "penalty": self.current_penalty,
                "steps": self.steps_taken,
                "warning": f"Source {action} already selected.",
                "action_mask": self.get_action_mask(),
            }
            obs = self.get_state()
            return obs, reward, terminated, truncated, info

        # Valid source selection
        self.selected_sources.add(action)
        self.steps_taken += 1
        selected_source = self.sources_list[action]
        new_T = algo_main(selected_source, self.UR, 1)

        if self.current_table.empty:
            self.current_table = new_T
        elif not new_T.empty:
            self.current_table = (
                self.current_table.set_index("Identifiant")
                .combine_first(new_T.set_index("Identifiant"))
                .reset_index()
            )
            optimized_table, _ = optimize_selection(self.current_table, self.UR)
            self.current_table = optimized_table

        new_coverage, _ = compute_overall_coverage(self.current_table, self.UR)
        new_penalty, _ = compute_overall_penalty(self.current_table, self.UR)

        reward = self.compute_reward(new_coverage, new_penalty)
        self.current_coverage = new_coverage
        self.current_penalty = new_penalty
        self.last_coverage = new_coverage
        self.last_penalty = new_penalty
        self.last_steps = self.steps_taken

        # Episode done if all sources have been picked
        if len(self.selected_sources) == self.num_sources:
            terminated = True
            info["forced_stop"] = True

        obs = self.get_state()
        info.update({
            "stop": False,
            "coverage": self.current_coverage,
            "penalty": self.current_penalty,
            "steps": self.steps_taken,
            "selected_sources": list(self.selected_sources),
            "action_mask": self.get_action_mask(),
        })
        return obs, reward, terminated, truncated, info

    def compute_reward(self, new_coverage, new_penalty):
        normalized_steps = self.steps_taken / self.max_steps if self.max_steps > 0 else 0.0
        reward = (
            self.alpha * new_coverage
            - self.beta * new_penalty
            - self.gamma * normalized_steps
        )
        return 10 * reward

    def get_state(self):
        state = []
        state += [self.current_coverage, self.current_penalty]
        selected_mask = [1 if i in self.selected_sources else 0 for i in range(len(self.sources_list))]
        state += selected_mask
        for i in range(len(self.sources_list)):
            state += list(self.statistics.get(i, np.zeros(len(self.statistics[0]), dtype=np.float32)))
        return np.array(state, dtype=np.float32)

    def get_action_mask(self):
        """
        Returns a boolean mask for actions: True if action is valid, False otherwise.
        Only unpicked sources and STOP are valid.
        """
        mask = np.ones(self.action_space.n, dtype=bool)
        for idx in self.selected_sources:
            mask[idx] = False
        # STOP action always allowed
        mask[self.stop_action] = True
        return mask
