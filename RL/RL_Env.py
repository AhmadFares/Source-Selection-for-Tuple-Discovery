import numpy as np
import pandas as pd
import random
import gym
from collections import defaultdict
from gym import spaces

from Single_Source.Coverage_Guided_Row_Selection import algo_main, compute_overall_coverage, compute_overall_penalty

class DataSelectionEnv(gym.Env):
    def __init__(self, sources_list, user_request, theta, tau, statistics=None): 
        self.sources_list = sources_list  
        self.UR = user_request
        self.theta = theta
        self.tau = tau
        self.statistics = statistics or {}

        self.value_index = self._build_value_index()
        self.obs_len = 2 + len(self.sources_list) + len(self.sources_list) * len(self.value_index)

        self.action_space = spaces.Discrete(len(self.sources_list))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_len,),
            dtype=np.float32
        )

        # Ensures that every time the environment is initialized, it starts from a clean, consistent state
        self.reset()

    def _build_value_index(self):
        value_set = set()
        for stats in self.statistics.values():
            for val_freq in stats.values():
                value_set.update(val_freq.keys())
        return {val: i for i, val in enumerate(sorted(value_set))}

    def reset(self):
        self.selected_sources = set()
        self.current_table = pd.DataFrame(columns=self.UR.columns)
        self.current_coverage, _ = compute_overall_coverage(self.current_table, self.UR)
        self.current_penalty, _ = compute_overall_penalty(self.current_table, self.UR)
        self.statistics = {}

        return self.get_state()

    def encode_source_statistics(self, stat_dict):
        vec = [0.0] * len(self.value_index)
        for val, freq in stat_dict.items():
            if val in self.value_index:
                vec[self.value_index[val]] = freq
        return vec
    
    
    def print_rl_info(self, state, action, reward, done, info):
        """
        Prints information about the RL step, including state, action, reward, etc.
        
        Parameters:
        - state: The current state of the environment
        - action: The action chosen by the agent
        - reward: The reward received after the action
        - done: Whether the episode has ended
        - info: Additional info, e.g., coverage, penalty, selected sources
        """
        print("------ RL Step Info ------")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Additional Info: {info}")
        print("--------------------------")


    def step(self, action):
        action = int(np.asarray(action).item())
        source_idx = action

        if source_idx in self.selected_sources:
            reward = -1.0
            done = True
            info = {"error": f"Source {source_idx} already selected."}
            return self.get_state(), reward, done, info

        selected_source = self.sources_list[source_idx]
        self.selected_sources.add(source_idx)

        new_T = algo_main(selected_source, self.UR, self.theta)

        if self.current_table.empty:
            self.current_table = new_T
        else:
            self.current_table = (
                self.current_table.set_index("Identifiant")
                .combine_first(new_T.set_index("Identifiant"))
                .reset_index()
            )

        new_coverage, _ = compute_overall_coverage(self.current_table, self.UR)
        new_penalty, _ = compute_overall_penalty(self.current_table, self.UR)

        reward = self.compute_reward(new_coverage, new_penalty)
        self.current_coverage = new_coverage
        self.current_penalty = new_penalty

        done = self.check_termination()
        next_state = self.get_state()
        info = {
            "coverage": new_coverage,
            "penalty": new_penalty, 
            "selected_sources": list(self.selected_sources)
        }
        self.print_rl_info(self,next_state, action, reward, done, info)
        return next_state, reward, done, info

    def compute_reward(self, new_coverage, new_penalty):
        delta_cov = new_coverage - self.current_coverage
        delta_pen = new_penalty - self.current_penalty
        return delta_cov - delta_pen

    def check_termination(self):
        if self.current_coverage >= self.theta:
            if self.current_penalty > self.tau:
                return True
            return False

    def get_state(self):
        selected_mask = [1 if i in self.selected_sources else 0 for i in range(len(self.sources_list))]
        stats_vector = []
        for i in range(len(self.sources_list)):
            combined_col_stat = {}
            for col_stat in self.statistics.get(i, {}).values():
                for val, freq in col_stat.items():
                    combined_col_stat[val] = combined_col_stat.get(val, 0.0) + freq
            encoded = self.encode_source_statistics(combined_col_stat)
            stats_vector += encoded

        state_vector = [self.current_coverage, self.current_penalty] + selected_mask + stats_vector
        state = np.array(state_vector, dtype=np.float32)
        return state

    def get_available_actions(self):
        return [i for i in range(len(self.sources_list)) if i not in self.selected_sources]