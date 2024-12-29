from typing import List

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
        names: List[str]
) -> None:
    results = []
    true_cdf = np.cumsum(true_density) / np.sum(true_density)
    dx = x_grid[1] - x_grid[0]

    print(f"{'Estimator':<20}{'KS Statistic':<15}{'KS p-value':<15}{'ISE':<15}{'Hellinger Distance':<20}")
    print("-" * 80)

    for name, est_density in zip(names, estimated_densities):
        est_cdf = np.cumsum(est_density) * dx
        ks_stat, ks_pvalue = stats.ks_2samp(true_cdf, est_cdf)
        ise = np.sum((true_density - est_density) ** 2) * dx
        hd = hellinger_distance(true_density, est_density, dx)
        results.append([name, ks_stat, ks_pvalue, ise, hd])

        print(f"{name:<20}{ks_stat:<15.6f}{ks_pvalue:<15.6f}{ise:<15.6f}{hd:<20.6f}")
