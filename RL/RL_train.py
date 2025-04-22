import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from RL.RL_Env import DataSelectionEnv 
from helpers.T_splitter_into_M import split_uniform_by_rows
from helpers.statistics_computation import compute_all_source_statistics
from helpers.test_cases import TestCases
import pandas as pd
import random

TestCases = TestCases()  # Initialize the TestCases class

T_input, UR = TestCases.get_case(1)
columns_to_keep = ["Identifiant", "Keyword1", "Keyword2", "Keyword3"]
T_input = T_input[columns_to_keep].copy()

sources_list = split_uniform_by_rows(T_input, 10)  # Split the input table into 5 sources
statistics = compute_all_source_statistics(sources_list)


# sources_list = [pd.DataFrame({'A': [random.randint(1, 3) for _ in range(5)],
#                               'B': [random.randint(1, 3) for _ in range(5)],
#                               'Identifiant': [i for i in range(5)]}) for _ in range(5)]

# UR = pd.DataFrame({'A': [1, 2], 'B': [2, 3], 'Identifiant': [0, 1]})



# Environment parameters
theta = 1.0
tau = 0.1


# Create environment
env = DataSelectionEnv(sources_list, UR, theta, tau, statistics=statistics)



# Wrap environment 
#env = gym.wrappers.TimeLimit(env, max_episode_steps=len(sources_list))
#print("State shape:", env.get_state().shape)
#print("Example state:", env.get_state())

# Create model
#model = DQN("MlpPolicy", env, verbose=1)
model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000)  # instead of 1,000,000 default

# Train model
model.learn(total_timesteps=5000)

# Save model
model.save("dqn_data_selection")

print("Training completed and model saved!")
