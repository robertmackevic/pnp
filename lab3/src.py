from typing import List, Callable

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import stats
from scipy.spatial import KDTree
from sklearn.neighbors import KernelDensity


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


def refined_plugin_bandwidth(data):
    n = len(data)

    h0 = stats.gaussian_kde(data).factor * np.std(data)

    kde = KernelDensity(bandwidth=h0, kernel='gaussian')
    kde.fit(data[:, np.newaxis])

    x_grid = np.linspace(min(data), max(data), 1000)[:, np.newaxis]
    log_dens = kde.score_samples(x_grid)
    density = np.exp(log_dens)

    dx = x_grid[1, 0] - x_grid[0, 0]
    d4 = np.gradient(np.gradient(np.gradient(np.gradient(density)))) / dx ** 4
    d6 = np.gradient(np.gradient(d4)) / dx ** 2

    psi4_hat = np.mean(np.abs(d4))
    psi6_hat = np.mean(np.abs(d6))

    h_opt = ((8 * np.sqrt(np.pi) * stats.norm.pdf(0)) /
             (3 * n * psi4_hat ** 2)) ** (1 / 5)

    return h_opt


def silverman_bandwidth(data: NDArray) -> float:
    return (4 * np.std(data) ** 5 / (3 * len(data))) ** (1 / 5)


def smoothed_bootstrap_bandwidth(
        data: NDArray, n_bootstrap: int, bandwidths: NDArray, true_density_func: Callable
) -> float:
    bootstrap_bandwidths = []
    kde = stats.gaussian_kde(data, bw_method=silverman_bandwidth(data))

    for _ in range(n_bootstrap):
        bootstrap_samples = kde.resample(size=len(data)).squeeze(0)
        lscv_losses = [lscv_loss(bootstrap_samples, bw.item(), true_density_func) for bw in bandwidths]
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
