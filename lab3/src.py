from typing import Optional, List

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import stats
from scipy.spatial import KDTree


def lscv_loss(data: NDArray, bandwidth: float) -> float:
    n = len(data)
    density = stats.gaussian_kde(data, bw_method=bandwidth)(data)
    first_term = np.mean(density)
    loo_density_values = [
        stats.gaussian_kde(np.delete(data, i), bw_method=bandwidth)(data[i])
        for i in range(n)
    ]
    second_term = (2 / n) * np.sum(loo_density_values)
    return first_term - second_term


def refined_plugin_bandwidth_selection(data: NDArray, max_iter: int = 100, tol: float = 1e-5) -> float:
    n = len(data)
    std_dev = np.std(data)
    initial_bw = (4 * std_dev ** 5 / (3 * n)) ** (1 / 5)

    current_bw = initial_bw
    for _ in range(max_iter):
        current_loss = lscv_loss(data, current_bw)

        bw_left = current_bw - 0.01
        bw_right = current_bw + 0.01
        left_loss = lscv_loss(data, bw_left)
        right_loss = lscv_loss(data, bw_right)

        if left_loss < current_loss:
            current_bw = bw_left
        elif right_loss < current_loss:
            current_bw = bw_right
        else:
            break

        if abs(current_loss - min(left_loss, right_loss)) < tol:
            break

    return current_bw


def smoothed_bootstrap_bandwidth(
        data: NDArray, n_bootstrap: int = 100, initial_bandwidth: Optional[float] = None
) -> float:
    n = len(data)
    bootstrap_bandwidths = []

    # Step 1: Initial KDE smoothing
    if initial_bandwidth is None:
        initial_bandwidth = 1.06 * np.std(data) * n ** (-1 / 5)  # Silverman's rule
    kde = stats.gaussian_kde(data, bw_method=initial_bandwidth / np.std(data))

    # Step 2: Generate bootstrap samples
    for _ in range(n_bootstrap):
        bootstrap_sample = kde.resample(n).flatten()

        # Grid search for optimal bandwidth using the existing lscv_loss
        bandwidths = np.linspace(0.01, 1.0, 50) * np.std(bootstrap_sample)
        losses = [lscv_loss(bootstrap_sample, bw) for bw in bandwidths]
        optimal_bw = bandwidths[np.argmin(losses)]
        bootstrap_bandwidths.append(optimal_bw)

    final_bandwidth = np.mean(bootstrap_bandwidths)
    return final_bandwidth.item()


def compute_knn_density(data: NDArray, x_grid: NDArray, k: int) -> NDArray:
    n = len(data)
    tree = KDTree(data.reshape(-1, 1))
    densities = []

    for x in x_grid:
        distances, _ = tree.query(x, k=k)

        if not isinstance(distances, ArrayLike):
            distances = [distances]

        r_k = distances[-1]
        volume = 2 * r_k
        densities.append(k / (n * volume))

    return np.array(densities)


def evaluate_densities(
        true_density: NDArray,
        predicted_densities: List[NDArray],
        x_grid: NDArray,
        n_samples: int,
        method_names: List[str],
        seed: Optional[int] = None
) -> None:
    np.random.seed(seed)
    dx = x_grid[1] - x_grid[0]
    true_density /= np.sum(true_density) * dx
    true_probs = true_density / np.sum(true_density)

    results = []
    for predicted_density, method_name in zip(predicted_densities, method_names):
        predicted_density /= np.sum(predicted_density) * dx
        predicted_density = np.clip(predicted_density, 1e-10, None)

        predicted_probs = predicted_density / np.sum(predicted_density)
        true_samples = np.random.choice(x_grid, size=n_samples, p=true_probs)
        predicted_samples = np.random.choice(x_grid, size=n_samples, p=predicted_probs)

        ks_stat, ks_pval = stats.ks_2samp(true_samples, predicted_samples)
        ise = np.sum((predicted_density - true_density) ** 2) * dx
        mae = np.max(np.abs(predicted_density - true_density))
        kl_div = stats.entropy(true_density, predicted_density)

        results.append((method_name, ks_stat, ks_pval, ise, mae, kl_div))

    print(f"{'Method':<25}{'KS-Stat':<10}{'KS-Pval':<10}{'ISE':<10}{'MAE':<10}{'KL Div':<10}")
    print("-" * 80)
    for method_name, ks_stat, ks_pval, ise, mae, kl_div in results:
        print(f"{method_name:<25}{ks_stat:<10.6f}{ks_pval:<10.6f}{ise:<10.6f}{mae:<10.6f}{kl_div:<10.6f}")
