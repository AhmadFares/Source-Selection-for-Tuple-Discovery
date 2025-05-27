import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def moving_average(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w), 'valid') / w

# Load your data (adjust filename if needed)
data = np.load('all_training_metrics_Ignore.npz', allow_pickle=True)
print(type(data['rewards']))
print(data['rewards'].shape)
rewards_dict    = data['rewards'].item()
coverages_dict  = data['coverages'].item()
penalties_dict  = data['penalties'].item()
steps_dict      = data['steps'].item()
stopped_dict    = data['stopped'].item()

# --- Set up consistent color palette for variants ---
variants = list(rewards_dict.keys())
palette = sns.color_palette("tab10", n_colors=len(variants))
variant_colors = {v: palette[i] for i, v in enumerate(variants)}

window = 50  # Smoothing window
sns.set(style="whitegrid", font_scale=1.15)

def plot_metric(metric_dict, title, ylabel, filename, is_binary=False):
    plt.figure(figsize=(11, 6))
    for variant in variants:
        # if variant == "case20_low_penalty_a0.6_b0.3" or variant == "case20_group_by_attr_a0.6_b0.3":
        #      continue
        values = np.array(metric_dict[variant])
        # Smooth binary metrics differently (e.g. STOP used)
        if is_binary:
            values = values.astype(float)
        if len(values) >= window:
            smoothed = moving_average(values, window)
            plt.plot(
                range(window-1, window-1+len(smoothed)),
                smoothed,
                label=variant,
                color=variant_colors[variant]
            )
        else:
            plt.plot(values, label=variant, color=variant_colors[variant])
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# --------- Plotting each metric ---------

plot_metric(rewards_dict,    "Reward per Episode (Smoothed) for All Variants",
            "Reward", "smoothed_all_variants_rewards.png")

plot_metric(coverages_dict,  "Coverage per Episode (Smoothed) for All Variants",
            "Coverage", "smoothed_all_variants_coverage.png")

plot_metric(penalties_dict,  "Penalty per Episode (Smoothed) for All Variants",
            "Penalty", "smoothed_all_variants_penalty.png")

plot_metric(steps_dict,      "Steps Taken per Episode (Smoothed) for All Variants",
            "Steps", "smoothed_all_variants_steps.png")

plot_metric(stopped_dict,    "STOP Action Used per Episode (Smoothed) for All Variants",
            "STOP Action Used (Fraction per window)", "smoothed_all_variants_stop.png", is_binary=True)
