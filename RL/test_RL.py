# Assume your environment class is in data_selection_env.py
# from data_selection_env import DataSelectionEnv

# Dummy test case: create list of data sources (for testing)
import pandas as pd
import random

from RL.RL_Env import DataSelectionEnv
from helpers.T_splitter_into_M import split_by_diagonal
import helpers.test_cases as test_cases 
from helpers.test_cases import TestCases  # Import the TestCases class

# Let's say you have 5 dummy sources (replace with your real sources later)
sources_list = [pd.DataFrame({'A': [random.randint(1, 3) for _ in range(5)],
                               'B': [random.randint(1, 3) for _ in range(5)],
                              'Identifiant': [i for i in range(5)]}) for _ in range(5)]
# test_cases = TestCases()  # Initialize the TestCases class
# T_input, UR = test_cases.get_case(6) 
# sources_list = split_by_diagonal(T_input) 

# Dummy user request (UR)
UR = pd.DataFrame({'A': [1, 2], 'B': [2, 3], 'Identifiant': [0, 1]})

# Environment parameters
theta = 1.0  # target coverage
tau = 0.1    # penalty threshold
reward_mode = 'option1'  # or 'option2'

# Initialize environment
env = DataSelectionEnv(sources_list, UR, theta, tau, reward_mode)

# Start environment
state = env.reset()
done = False
total_reward = 0
step_count = 0

print("Starting random agent episode...\n")

while not done:
    # Random action: pick one of the unselected sources
    possible_actions = env.action_space()
    
    if not possible_actions:
        print("No more possible actions. Ending episode.")
        done = True
        break

    action = random.choice(possible_actions)

    # Apply action
    next_state, reward, done, info = env.step(action)

    # Logging
    print(f"Step {step_count}:")
    print(f"Selected Source Index: {action}")
    print(f"Reward: {reward}")
    print(f"New Coverage: {info['coverage']:.4f}, New Penalty: {info['penalty']:.4f}")
    print(f"Selected Sources: {info['selected_sources']}")
    print("-" * 50)

    # Update
    state = next_state
    total_reward += reward
    step_count += 1

print("\nEpisode finished!")
print(f"Total steps: {step_count}")
print(f"Total reward: {total_reward:.4f}")
