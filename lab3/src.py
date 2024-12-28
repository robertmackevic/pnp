from typing import Optional, List

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray, ArrayLike
from scipy import stats
from scipy.spatial import KDTree


def compute_kernel_density(x: NDArray, data: NDArray, bandwidth: NDArray) -> NDArray:
    return np.sum(stats.norm.pdf((x - data[:, None]) / bandwidth), axis=0) / (len(data) * bandwidth)


def lscv_loss(data: NDArray, bandwidth: NDArray, n_jobs: int = -1) -> NDArray:
    n = len(data)
    kde_estimate = compute_kernel_density(data, data, bandwidth)
    first_term = np.mean(kde_estimate)

    def loo_density(i):
        return np.sum(stats.norm.pdf((data[i] - np.delete(data, i)) / bandwidth)) / ((n - 1) * bandwidth)

    loo_density_values = Parallel(n_jobs=n_jobs)(delayed(loo_density)(i) for i in range(n))
    second_term = (2 / n) * np.sum(loo_density_values)

    return first_term - second_term


def plugin_bandwidth(data: NDArray) -> NDArray:
    # Step 1: Pre-smoothing (initial kernel density estimate with rule-of-thumb bandwidth)
    initial_bandwidth = 1.06 * np.std(data) * len(data) ** (-1 / 5)
    kde = stats.gaussian_kde(data, bw_method=initial_bandwidth / np.std(data))

    # Step 2: Estimate f''(x) by applying a second-order kernel
    # Generate a grid for density estimation
    x_grid = np.linspace(np.min(data), np.max(data), 1000)
    kde_values = kde(x_grid)

    # Numerical second derivative estimation
    second_derivative = np.gradient(np.gradient(kde_values, x_grid), x_grid)
    squared_integral = np.sum(second_derivative ** 2) * (x_grid[1] - x_grid[0])

    refined_bandwidth = (squared_integral * len(data) ** (-1)) ** (1 / 5)
    return refined_bandwidth


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
