import random
from math import sqrt
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy import stats


def compute_pareto_mean(scale: float, shape: float) -> float:
    return stats.pareto(b=shape, scale=scale).mean().item()


def compute_pareto_variance(scale: float, shape: float) -> float:
    return stats.pareto(b=shape, scale=scale).var().item()


def compute_theta(mean: float, var: float) -> Tuple[float, float]:
    _sqrt = sqrt(var ** 2 + var * mean ** 2)
    a1 = (var + _sqrt) / var
    a2 = (var - _sqrt) / var
    a = max(a1, a2)
    assert a > 2
    xm = mean * (a - 1) / a
    assert xm > 0
    return xm, a


def compute_g(scale: float, shape: float, num_samples: int) -> NDArray:
    return stats.pareto.rvs(b=shape, scale=scale, size=num_samples)


def plot_samples(samples: NDArray) -> None:
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.75, color="blue", edgecolor="black")
    plt.title("Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()


def plot_samples_with_pareto_baseline(samples: NDArray, scale: float, shape: float) -> None:
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.75, color="blue", edgecolor="black")

    x = np.linspace(scale, max(samples), 1000)
    pdf = stats.pareto.pdf(x, b=shape, scale=scale)
    plt.plot(x, pdf, "r-", lw=2, label="Theoretical PDF")

    plt.title(f"Pareto Distribution (shape={shape:.3f}, scale={scale:.3f})")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
