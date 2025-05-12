import numpy as np
import matplotlib.pyplot as plt

# Data organization
reflections = [0, 1, 2, 3]

# Without caching data
latency_no_cache = {
    0: [11675.0, 13147.0, 18954.0],
    1: [29638.0, 28086.0, 26993.0],
    2: [41972.0, 56863.0, 48044.0],
    3: [59327.0, 53439.0, 65698.0],
}
for r, v in latency_no_cache.items():
    latency_no_cache[r] = [x / 1000.0 for x in v]

cost_no_cache = {
    0: [0.01265, 0.011328, 0.011679],
    1: [0.027498, 0.028629, 0.027141],
    2: [0.050304, 0.054258, 0.04917],
    3: [0.075846, 0.079761, 0.081615],
}

# With caching data
latency_cache = {
    0: [16242.0, 12808.0, 12071.0],
    1: [30588.0, 28980.0, 32395.0],
    2: [83311.0, 40756.0, 37015.0],
    3: [66972.0, 54692.0, 84885.0],
}
for r, v in latency_cache.items():
    latency_cache[r] = [x / 1000.0 for x in v]

cost_cache = {
    0: [0.0112, 0.01035, 0.011184],
    1: [0.0265158, 0.023845, 0.024722],
    2: [0.042536, 0.0402345, 0.038704],
    3: [0.0539265, 0.0577167, 0.0578],
}


# Calculate means and percentiles
def get_stats(data):
    means = [np.mean(data[r]) for r in reflections]
    p25 = [np.percentile(data[r], 25) for r in reflections]
    p75 = [np.percentile(data[r], 75) for r in reflections]
    return means, p25, p75


# Calculate cost reduction percentages
def calculate_cost_reduction(no_cache_data, cache_data):
    no_cache_means = [np.mean(no_cache_data[r]) for r in reflections]
    cache_means = [np.mean(cache_data[r]) for r in reflections]

    reductions = []
    for nc, c in zip(no_cache_means, cache_means):
        reduction = ((nc - c) / nc) * 100
        reductions.append(reduction)
    return reductions


# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Latency plot
means_nc_lat, p25_nc_lat, p75_nc_lat = get_stats(latency_no_cache)
means_c_lat, p25_c_lat, p75_c_lat = get_stats(latency_cache)

ax1.plot(reflections, means_nc_lat, "b-", label="Without Cache")
ax1.fill_between(reflections, p25_nc_lat, p75_nc_lat, alpha=0.2, color="b")
ax1.plot(reflections, means_c_lat, "r-", label="With Cache")
ax1.fill_between(reflections, p25_c_lat, p75_c_lat, alpha=0.2, color="r")

ax1.set_xlabel("Number of Reflections")
ax1.set_ylabel("Latency (s)")
ax1.set_title("Latency vs Number of Reflections")
ax1.set_xticks(reflections)
ax1.legend()
ax1.grid(True)

# Cost plot
means_nc_cost, p25_nc_cost, p75_nc_cost = get_stats(cost_no_cache)
means_c_cost, p25_c_cost, p75_c_cost = get_stats(cost_cache)

ax2.plot(reflections, means_nc_cost, "b-", label="Without Cache")
ax2.fill_between(reflections, p25_nc_cost, p75_nc_cost, alpha=0.2, color="b")
ax2.plot(reflections, means_c_cost, "r-", label="With Cache")
ax2.fill_between(reflections, p25_c_cost, p75_c_cost, alpha=0.2, color="r")

# Calculate and add cost reduction annotations
cost_reductions = calculate_cost_reduction(cost_no_cache, cost_cache)
for i, reduction in enumerate(cost_reductions):
    y_pos = (means_nc_cost[i] + means_c_cost[i]) / 2
    ax2.annotate(
        f"-{reduction:.1f}%",
        xy=(i, y_pos),
        xytext=(10, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
    )

ax2.set_xlabel("Number of Reflections")
ax2.set_ylabel("Cost (USD)")
ax2.set_title("Cost vs Number of Reflections")
ax2.set_xticks(reflections)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("./caching.png")
