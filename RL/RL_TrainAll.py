
import os
import numpy as np
import matplotlib.pyplot as plt

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
        self.steps = []
        self.stopped = []

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    self.rewards.append(self.locals['rewards'][idx])
                    env = self.training_env.envs[idx]
                    real_env = env.env
                    self.coverages.append(real_env.last_coverage)
                    self.penalties.append(real_env.last_penalty)
                    self.steps.append(real_env.steps_taken)
                    self.stopped.append(self.locals['infos'][idx].get("stop", False))
        return True

def dataframe_to_ur_dict(df):
    return {col: set(df[col].dropna().unique()) for col in df.columns}

def train_model(T, UR, sources, alpha, beta, gamma, save_path):
    value_index, source_stats = compute_UR_value_frequencies_in_sources(sources, UR)
    env = DataSelectionEnv(sources, UR, source_stats, value_index, alpha, beta, gamma)
    model = DQN("MlpPolicy", env, verbose=0, buffer_size=10000)
    callback = MetricLoggerCallback()
    model.learn(total_timesteps=1000, callback=callback)

    os.makedirs(save_path, exist_ok=True)

    np.savez(os.path.join(save_path, "metrics.npz"),
             rewards=callback.rewards,
             coverages=callback.coverages,
             penalties=callback.penalties,
             steps=callback.steps,
             stopped=callback.stopped)

    def plot_and_save(metric, name):
        plt.figure()
        plt.plot(metric)
        plt.title(name)
        plt.xlabel("Episode")
        plt.ylabel(name)
        plt.grid()
        plt.savefig(os.path.join(save_path, f"{name.lower()}_curve.png"))
        plt.close()

    plot_and_save(callback.rewards, "Reward")
    plot_and_save(callback.coverages, "Coverage")
    plot_and_save(callback.penalties, "Penalty")
    plot_and_save(callback.steps, "Steps")
    plot_and_save(callback.stopped, "STOP Action Used")

    model.save(os.path.join(save_path, "dqn_model"))

def run_all():
    test_cases = TestCases()
    ur_cases = [20, 21, 22]
    source_variants = {
        "low_penalty": lambda ctor: ctor.low_penalty_sources(),
        "high_penalty": lambda ctor: ctor.high_penalty_sources(),
        "low_coverage": lambda ctor: ctor.low_coverage_sources(),
        "group_by_attr": lambda ctor: ctor.group_by_attr_sources()
    }
    alpha_values = [0.3, 0.5, 0.7]
    beta_values = [0.2, 0.4, 0.6]

    for case_id in ur_cases:
        T, UR = test_cases.get_case(case_id)
       # UR_dict = dataframe_to_ur_dict(UR)
        constructor = SourceConstructor(T, UR)

        for variant_name, variant_fn in source_variants.items():
            sources = variant_fn(constructor)

            for alpha in alpha_values:
                for beta in beta_values:
                    gamma = 1.0 - alpha - beta
                    if gamma < 0:
                        continue
                    save_dir = f"results/case_{case_id}/{variant_name}/alpha_{alpha}_beta_{beta}"
                    print(f"Training: Case={case_id}, Source={variant_name}, Alpha={alpha}, Beta={beta}, Gamma={gamma}")
                    train_model(T, UR, sources, alpha, beta, gamma, save_dir)

if __name__ == "__main__":
    run_all()