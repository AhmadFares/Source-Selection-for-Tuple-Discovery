import pandas as pd
import random
from stable_baselines3 import DQN
from RL.RL_Env import DataSelectionEnv  # Adjust path if needed

# Dummy data (replace with your real sources and UR)
sources_list = [pd.DataFrame({'A': [random.randint(1, 3) for _ in range(5)],
                              'B': [random.randint(1, 3) for _ in range(5)],
                              'Identifiant': [i for i in range(5)]}) for _ in range(5)]

UR = pd.DataFrame({'A': [1, 2], 'B': [2, 3], 'Identifiant': [0, 1]})

theta = 1.0
tau = 0.1
reward_mode = 'option1'

# Load environment and model
env = DataSelectionEnv(sources_list, UR, theta, tau, reward_mode)
model = DQN.load("dqn_data_selection")

# Evaluate the model
obs = env.reset()
done = False
step = 0

print("\nEvaluation of trained agent:\n")

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"Step {step}: Action={action}, Reward={reward}, Coverage={info['coverage']:.4f}, Penalty={info['penalty']:.4f}")
    step += 1

print("\nEvaluation finished!")
print(f"Final Coverage: {info['coverage']}")
print(f"Final Penalty: {info['penalty']}")
print(f"Selected Sources: {info['selected_sources']}")
