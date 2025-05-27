import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch


from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from RL.RL_Env import DataSelectionEnv
from helpers.test_cases import TestCases
from helpers.Source_Constructors import SourceConstructor
from helpers.statistics_computation import compute_UR_value_frequencies_in_sources

class MetricLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricLoggerCallback, self).__init__(verbose)
        self.rewards = []
        self.coverages = []
        self.penalties = []
        self.actions = []
        self.steps = []
        self.stopped = []
        self.episode_actions = []
        self.q_values = []  


    def _on_step(self) -> bool:
        action = self.locals.get('actions')
        if action is not None:
            # Handles both single and vectorized envs
            if isinstance(action, (list, np.ndarray)) and len(action) == 1:
                action = action[0]
            self.episode_actions.append(action)
        
        obs = self.locals.get('new_obs')
        model = self.model
        if obs is not None and model is not None:
            # obs might be batched or not
            obs_arr = np.array(obs)
            if len(obs_arr.shape) == 1:
                obs_arr = obs_arr[None, :]
            q_vals = model.q_net(torch.tensor(obs_arr, dtype=torch.float32).to(model.device))
            # Save the max Q for the action taken
            self.q_values.append(q_vals.detach().cpu().numpy()[0])  # saves all Q-values for this step

        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    self.rewards.append(self.locals['rewards'][idx])
                    info = self.locals['infos'][idx]
                    self.coverages.append(info.get("coverage", 0.0))
                    self.penalties.append(info.get("penalty", 0.0))
                    self.steps.append(info.get("steps", 0))
                    self.stopped.append(info.get("stop", False))
                    
                    # When episode ends, store the episode's actions and reset
                    self.actions.append(self.episode_actions)
                    self.episode_actions = []
        return True


def dataframe_to_ur_dict(df):
    return {col: set(df[col].dropna().unique()) for col in df.columns}

def moving_avg(data, window_size=50):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().to_numpy()

def plot_and_save_sns(metric, name, save_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    smooth_metric = moving_avg(metric)
    sns.lineplot(data=smooth_metric)
    plt.title(f"{name} over Episodes")
    plt.xlabel("Episode")
    plt.ylabel(name)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{name.lower()}_curve.png"))
    plt.close()

def train_model(T, UR, sources, alpha, beta, gamma, save_path):
    value_index, source_stats = compute_UR_value_frequencies_in_sources(sources, UR)
    env = DataSelectionEnv(sources, UR, source_stats, value_index, alpha, beta, gamma)
    model = DQN("MlpPolicy", env, verbose=0, buffer_size=10000)
    callback = MetricLoggerCallback()
    model.learn(total_timesteps= 10000, callback=callback)

    os.makedirs(save_path, exist_ok=True)

    np.savez(os.path.join(save_path, "metrics.npz"),
             rewards=callback.rewards,
             coverages=callback.coverages,
             penalties=callback.penalties,
             steps=callback.steps,
             stopped=callback.stopped,
            actions=callback.actions,
            q_values=np.array(callback.q_values))

    plot_and_save_sns(callback.rewards, "Reward", save_path)
    plot_and_save_sns(callback.coverages, "Coverage", save_path)
    plot_and_save_sns(callback.penalties, "Penalty", save_path)
    plot_and_save_sns(callback.steps, "Steps", save_path)
    plot_and_save_sns(callback.stopped, "STOP Action Used", save_path)

    model.save(os.path.join(save_path, "dqn_model"))

    # Return metrics to aggregate outside
    return {
        "rewards": callback.rewards,
        "coverages": callback.coverages,
        "penalties": callback.penalties,
        "steps": callback.steps,
        "stopped": callback.stopped
    }

def run_all():
    test_cases = TestCases()
    ur_cases = [21]  # Reduced for example, add more if needed
    source_variants = {
        "low_penalty": lambda ctor: ctor.low_penalty_sources(),
        "high_penalty": lambda ctor: ctor.high_penalty_sources(),
        "low_coverage": lambda ctor: ctor.low_coverage_sources(),
    }
    alpha_values = [0.6]
    beta_values = [0.3]

    all_metrics = {
        "rewards": {},
        "coverages": {},
        "penalties": {},
        "steps": {},
        "stopped": {}
    }

    for case_id in ur_cases:
        T, UR = test_cases.get_case(case_id)
        constructor = SourceConstructor(T, UR)

        for variant_name, variant_fn in source_variants.items():
            sources = variant_fn(constructor)

            for alpha in alpha_values:
                for beta in beta_values:
                    gamma = 1.0 - alpha - beta
                    save_dir = f"results_UR2/case_{case_id}/{variant_name}/alpha_{alpha}_beta_{beta}"
                    print(f"Training: Case={case_id}, Source={variant_name}, Alpha={alpha}, Beta={beta}, Gamma={gamma}")

                    metrics = train_model(T, UR, sources, alpha, beta, gamma, save_dir)

                    key = f"case{case_id}_{variant_name}_a{alpha}_b{beta}"
                    for metric_name in all_metrics.keys():
                        all_metrics[metric_name][key] = metrics[metric_name]

    # Save all collected metrics from all runs
    np.savez("all_training_metrics_UR2.npz", **all_metrics)
    print("All training metrics saved to all_training_metrics.npz")

if __name__ == "__main__":
    run_all()

