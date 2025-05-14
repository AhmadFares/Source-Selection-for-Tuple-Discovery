from stable_baselines3 import DQN
from RL.RL_Env import DataSelectionEnv
from Single_Source.Coverage_Guided_Row_Selection import compute_overall_coverage, compute_overall_penalty, optimize_selection
from helpers.test_cases import TestCases
from helpers.statistics_computation import compute_UR_value_frequencies_in_sources
from helpers.T_splitter_into_M import split_uniform_by_rows

# Reload test case
test_cases = TestCases()
T_input, UR_df = test_cases.get_case(20)

# Filter necessary columns
#T_input = T_input[["Identifiant", "Keyword1", "Keyword2", "Keyword3"]].copy()

# Split into sources
sources_list = split_uniform_by_rows(T_input, 10)

# Compute statistics
value_index, source_stats = compute_UR_value_frequencies_in_sources(sources_list, UR_df)

# Recreate env with the same theta
theta = 1.0
tau = 0.1
env = DataSelectionEnv(sources_list, UR_df, theta, tau, statistics=source_stats, value_index=value_index)

# Load model
model = DQN.load("dqn_data_selection")

# Run one evaluation episode
obs = env.reset()
done = False
final_info = None  # Track last info
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    final_info = info  # Save the most recent info

final_coverage = env.current_coverage
final_penalty = env.current_penalty
selected_sources = list(env.selected_sources)
# AFTER the episode ends:
env.current_table = optimize_selection(env.current_table, env.UR)[0]
final_cov_raw, _ = compute_overall_coverage(env.current_table, env.UR)
final_pen_raw, _ = compute_overall_penalty(env.current_table, env.UR)

print("\n📊 Evaluation Result")
print("✅ Final Coverage:", final_coverage)
print("❗ Final Penalty:", final_penalty)
print("📦 Sources Used:", selected_sources)
print("🏁 Final Reward:", final_coverage - final_penalty)
print("\n🧾 Final Output Table:")
print("📦 Source selection order:", list(env.selected_sources))
print("📦 Final Coverage:", final_cov_raw)
print("❗ Final Penalty:", final_pen_raw) 
print(env.current_table)


