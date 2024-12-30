from typing import List, Callable

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import stats
from scipy.spatial import KDTree


def lscv_loss(data: NDArray, bandwidth: float, true_density_func: Callable) -> float:
    n = len(data)
    lscv_error = 0

    for i in range(n):
        remaining_samples = np.delete(data, i)
        kde = stats.gaussian_kde(remaining_samples, bw_method=bandwidth)
        estimated_density = kde.evaluate([data[i]])

        true_density = true_density_func(data[i])
        error = (estimated_density - true_density) ** 2
        lscv_error += error

    return lscv_error / n


def silverman_bandwidth(data: NDArray) -> float:
    return (4 * np.std(data) ** 5 / (3 * len(data))) ** (1 / 5)


def _gaussian_kernel(x: NDArray) -> NDArray:
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def _gaussian_kernel_second_derivative(x: NDArray) -> NDArray:
    return (x ** 2 - 1) * np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def _kde(data: NDArray, bw: float, kernel: Callable) -> NDArray:
    n = len(data)
    y = np.zeros_like(data)

    for i in range(n):
        y += kernel((data - data[i]) / bw)

    return y / (n * bw)


def refined_plugin_bandwidth(data: NDArray) -> float:
    initial_bw = silverman_bandwidth(data)
    kde_values = _kde(data, initial_bw, _gaussian_kernel)
    estimate = np.mean(_gaussian_kernel_second_derivative((data - data[:, None]) / initial_bw), axis=0)
    optimal_bw = np.sqrt(np.sum(estimate ** 2) / np.sum(kde_values))
    return optimal_bw


def smoothed_bootstrap_bandwidth(
        data: NDArray, n_bootstrap: int, bandwidths: NDArray, true_density_func: Callable
) -> float:
    bootstrap_bandwidths = []
    kde = stats.gaussian_kde(data, bw_method=silverman_bandwidth(data))

    for _ in range(n_bootstrap):
        bootstrap_samples = kde.resample(size=len(data)).squeeze(0)
        lscv_losses = [
            lscv_loss(bootstrap_samples, bw.item(), true_density_func)
            for bw in bandwidths
        ]
        bootstrap_bandwidths.append(bandwidths[np.argmin(lscv_losses)])

    return sum(bootstrap_bandwidths) / n_bootstrap


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


def compare_densities(
        true_density: NDArray,
        estimated_densities: List[NDArray],
        names: List[str],
) -> None:
    print(f"{'Density':<15}{'MSE':<15}{'TVD':<15}{'KL Divergence':<20}{'Hellinger':<15}")
    print("-" * 75)

    for name, estimated_density in zip(names, estimated_densities):
        mse = np.mean((true_density - estimated_density) ** 2)
        tvd = 0.5 * np.sum(np.abs(true_density - estimated_density))
        kl_div = stats.entropy(true_density, estimated_density)
        hd = np.sqrt(0.5 * np.sum((np.sqrt(true_density) - np.sqrt(estimated_density)) ** 2))
        print(f"{name:<15}{mse:<15.6f}{tvd:<15.6f}{kl_div:<20.6f}{hd:<15.6f}")
