from typing import List

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import stats
from scipy.integrate import quad
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


def kernel_function(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def kernel_second_derivative(x):
    return (x ** 2 - 1) * kernel_function(x)


def roughness_functional(kernel_sec_deriv):
    return quad(lambda u: kernel_sec_deriv(u) ** 2, -np.inf, np.inf)[0]


def estimate_density_second_derivative(data, h):
    kde = stats.gaussian_kde(data, bw_method=h)
    second_derivative = lambda x: kde.evaluate(x)  # Approximate for second derivative
    return second_derivative


def refined_plugin_bandwidth_selection(data: NDArray) -> float:
    n = len(data)
    h_pilot = np.std(data) * n ** (-1 / 5)
    g_second_derivative = estimate_density_second_derivative(data, h_pilot)
    R_K = roughness_functional(kernel_second_derivative)
    integrated_second_derivative = quad(lambda x: g_second_derivative(x) ** 2, -np.inf, np.inf)[0]
    h = (R_K / (n * integrated_second_derivative)) ** (1 / 5)
    return h


def smoothed_bootstrap_bandwidth(data: NDArray, n_bootstrap: int, bandwidths: NDArray) -> float:
    return np.mean([
        bandwidths[np.argmin([
            lscv_loss(np.random.choice(data, size=len(data), replace=True), bw)
            for bw in bandwidths])] for _ in range(n_bootstrap)
    ]).item()


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


def hellinger_distance(p: NDArray, q: NDArray, dx: NDArray) -> float:
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2) * dx / 2).item()


def compare_densities(
        true_density: NDArray,
        estimated_densities: List[NDArray],
        x_grid: NDArray,
        names: List[str],
        n_samples: int
) -> None:
    dx = x_grid[1] - x_grid[0]
    true_density /= np.sum(true_density) * dx
    true_samples = np.random.choice(x_grid, size=n_samples, p=true_density * dx)

    print(f"{'Estimator':<20}{'KS Statistic':<15}{'KS p-value':<15}{'ISE':<15}{'Hellinger Distance':<20}")
    print("-" * 80)

    for name, est_density in zip(names, estimated_densities):
        est_density /= np.sum(est_density) * dx
        est_samples = np.random.choice(x_grid, size=n_samples, p=est_density * dx)
        ks_stat, ks_pvalue = stats.ks_2samp(true_samples, est_samples)
        ise = np.sum((true_density - est_density) ** 2) * dx
        hd = hellinger_distance(true_density, est_density, dx)

        print(f"{name:<20}{ks_stat:<15.6f}{ks_pvalue:<15.6f}{ise:<15.6f}{hd:<20.6f}")
