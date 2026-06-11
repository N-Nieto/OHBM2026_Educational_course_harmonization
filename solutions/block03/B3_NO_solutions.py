# Brief comparison of estimated ComBat batch effects vs simulator batch effects
# -------------------------------------------------------------------------
# To use this cell:
# 1. In the earlier ComBat cell, set:
#       ReturnPriors = True
# 2. Re-run the ComBat cell.
# 3. Run this cell.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if not isinstance(combat_data, dict) or "priors" not in combat_data:
    raise ValueError(
        "This comparison needs ComBat priors. Set ReturnPriors = True in the earlier "
        "ComBat cell, re-run that cell, then run this comparison cell."
    )

priors = combat_data["priors"]
levels = list(priors["levels"])

# Choose which ComBat estimates to compare.
# Default: empirical Bayes estimates.
gamma_est = priors["gamma_hat"]
delta_est = priors["delta_hat"]
estimate_label = "EB-adjusted estimates: gamma_star / delta_star"

# Alternative: raw least-squares estimates.
# Uncomment these lines to compare raw estimates instead.
# gamma_est = priors["gamma_hat"]
# delta_est = priors["delta_hat"]
# estimate_label = "Raw estimates: gamma_hat / delta_hat"

# Simulator batch parameters.
# Here we compare:
# - gamma-like additive effect against add_mean
# - delta-like multiplicative/scale effect against multi_mean
#
# Note: ComBat estimates are on the standardized data scale, so the values
# are not expected to match the simulator parameters exactly on their original scale.
rows = []

for i, batch_name in enumerate(levels):
    if batch_name not in batch_params:
        print(f"Skipping {batch_name}: not found in batch_params")
        continue

    true_add = batch_params[batch_name]["add_mean"]
    true_multi = batch_params[batch_name]["multi_mean"]

    est_gamma_mean = np.mean(gamma_est[i, :])
    est_delta_mean = np.mean(delta_est[i, :])

    rows.append({
        "batch": batch_name,
        "true_add_mean": true_add,
        "estimated_gamma_mean": est_gamma_mean,
        "true_multi_mean": true_multi,
        "estimated_delta_mean": est_delta_mean,
    })

comparison_df = pd.DataFrame(rows)

print(estimate_label)
display(comparison_df.round(4))

# Brief visual comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

comparison_df.plot(
    x="batch",
    y=["true_add_mean", "estimated_gamma_mean"],
    kind="bar",
    ax=axes[0],
)
axes[0].set_title("Additive batch effect")
axes[0].set_ylabel("Effect size")
axes[0].set_xlabel("Batch")
axes[0].tick_params(axis="x", rotation=45)

comparison_df.plot(
    x="batch",
    y=["true_multi_mean", "estimated_delta_mean"],
    kind="bar",
    ax=axes[1],
)
axes[1].set_title("Multiplicative / variance batch effect")
axes[1].set_ylabel("Effect size")
axes[1].set_xlabel("Batch")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()