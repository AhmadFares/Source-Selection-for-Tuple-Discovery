from contextlib import contextmanager
import os
import time
import sys
from stable_baselines3 import DQN
from Multi_Source.Multi_Source import multi_source_algorithm
from RL.RL_Env import DataSelectionEnv
from Single_Source.Coverage_Guided_Row_Selection import compute_overall_coverage, compute_overall_penalty, optimize_selection
from helpers.Source_Constructors import SourceConstructor
from helpers.test_cases import TestCases
from helpers.statistics_computation import compute_UR_value_frequencies_in_sources

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Load test case and split sources
test_cases = TestCases()
test_cases.load_mathe_case()
T, UR = test_cases.get_case(19)

print(UR)
constructor = SourceConstructor(T, UR)
sources_list = constructor.low_coverage_sources()  # <-- ensure this is the same for RL/offline

value_index, source_stats = compute_UR_value_frequencies_in_sources(sources_list, UR)
env = DataSelectionEnv(sources_list, UR, statistics=source_stats, value_index=value_index, alpha=0.6, beta=0.3, gamma=0.1)

# --- RL Agent Evaluation ---
model = DQN.load("dqn_model_high_penalty_1")

start_time = time.time()

obs = env.reset()
done = False
chosen_order = []  # To store actions in order
while not done:
    action, _ = model.predict(obs, deterministic=True)
    chosen_order.append(action)
    obs, reward, done, info = env.step(action)

rl_time = time.time() - start_time
final_coverage = env.current_coverage
final_penalty = env.current_penalty
selected_sources = list(env.selected_sources)
# AFTER the episode ends:
env.current_table = optimize_selection(env.current_table, env.UR)[0]
final_cov_rl, _ = compute_overall_coverage(env.current_table, env.UR)
final_pen_rl, _ = compute_overall_penalty(env.current_table, env.UR)

print("\n=== RL Agent Result ===")
print("\n📊 Evaluation Result")
print("📦 Sources Used:", selected_sources)
print("🏁 Final Reward:", final_coverage - final_penalty)
print("\n🧾 Final Output Table:")
print("📦 Source selection order:", list(env.selected_sources))
print("📦 Final Coverage:", final_cov_rl)
print("❗ Final Penalty:", final_pen_rl)
print("Time taken:", rl_time, "seconds") 
print(env.current_table)


# --- Offline/Deterministic Baseline ---
start_time = time.time()

with suppress_output():
    T_output, i , chosen_order= multi_source_algorithm(sources_list, UR, theta=1.0)
    T_output, _ = optimize_selection(T_output, UR)

offline_time = time.time() - start_time

final_cov_off, _ = compute_overall_coverage(T_output, UR)
final_pen_off, _ = compute_overall_penalty(T_output, UR)

print("\n=== Offline Baseline Result ===")
print("Final Coverage:", final_cov_off)
print("Final Penalty:", final_pen_off)
print("Order of sources (offline):", [f"s{idx}" for idx in chosen_order])     # for one-based (if you used i+1 above)
print(offline_time, "seconds")
print(T_output)
