import gym
import numpy as np
from stable_baselines3 import DQN
from RL.RL_Env import DataSelectionEnv
from helpers.T_splitter_into_M import split_uniform_by_rows
from helpers.Source_Constructors import SourceConstructor
from helpers.statistics_computation import compute_UR_value_frequencies_in_sources
from helpers.test_cases import TestCases
from stable_baselines3.common.callbacks import BaseCallback
import random
import torch


import matplotlib.pyplot as plt
rewards = []
coverages = []
penalties = []

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
                    self.stopped.append(
                        self.locals['infos'][idx].get("stop", False)
                    )  

        return True


def dataframe_to_ur_dict(df):
    return {
        col: set(df[col].dropna().unique())
        for col in df.columns
    }
TestCases = TestCases()
T_input, UR = TestCases.get_case(20)

UR_dict = dataframe_to_ur_dict(UR)

constructor = SourceConstructor(T_input, UR)
sources_list = constructor.low_penalty_sources()
# Filter relevant columns
#columns_to_keep = ["Identifiant", "Keyword1", "Keyword2", "Keyword3"]
#T_input = T_input[columns_to_keep].copy()

# Split table into 10 source tables


# Compute UR-based value frequencies
value_index, source_stats = compute_UR_value_frequencies_in_sources(sources_list, UR)

# Environment thresholds

# Create environment
env = DataSelectionEnv(
    sources_list=sources_list,
    user_request=UR,
    statistics=source_stats,
    value_index=value_index,
    alpha=0.5,   # coverage weight
    beta=0.4,    # penalty weight
    gamma=0.1    # step penalty weight (start low!)
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
env.seed(42) 

# Create and train model
model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000)
obs = env.reset()
# Initialize callback
metric_logger = MetricLoggerCallback()

# Train the model
model.learn(total_timesteps=1300, callback=metric_logger)

#  Save the metrics immediately after training
import numpy as np
np.savez("metrics_case1.npz", 
         rewards=metric_logger.rewards,
         coverages=metric_logger.coverages,
         penalties=metric_logger.penalties,
         steps=metric_logger.steps,
         stopped=metric_logger.stopped) 


def moving_avg(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


import matplotlib.pyplot as plt

# Plot Reward curve
plt.figure(figsize=(8,5))
plt.plot(moving_avg(metric_logger.rewards))
plt.title("Training Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.savefig("reward_curve.png")

# Plot Coverage curve
plt.figure(figsize=(8,5))
plt.plot(metric_logger.coverages)
plt.title("Coverage over Episodes")
plt.xlabel("Episode")
plt.ylabel("Coverage")
plt.grid()
plt.savefig("cov_curve.png")

# Plot Penalty curve
plt.figure(figsize=(8,5))
plt.plot(metric_logger.penalties)
plt.title("Penalty over Episodes")
plt.xlabel("Episode")
plt.ylabel("Penalty")
plt.grid()
plt.savefig("pen_curve.png")

#steps curve
plt.figure(figsize=(8,5))
plt.plot(metric_logger.steps)
plt.title("Steps Taken per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid()
plt.savefig("steps_curve.png")

# Plot STOP action usage
plt.figure(figsize=(8,5))
plt.plot(metric_logger.stopped)
plt.title("STOP Action Used")
plt.xlabel("Episode")
plt.ylabel("STOP Used (1=True, 0=False)")
plt.grid()
plt.savefig("stop_curve.png")

# Save the trained model
model.save("dqn_data_selection")

print("Training completed and model saved!")


