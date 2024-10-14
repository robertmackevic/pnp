import random
from math import sqrt
from typing import Tuple, List

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


def run_tests(
        theta_0: Tuple[float, float],
        theta_1: Tuple[float, float],
        theta_2: Tuple[float, float],
        p1: float,
        p2: float,
        alphas: List[float],
        num_samples: List[int],
) -> None:
    for n in num_samples:
        print(f"\n=======Testing with N={n} samples=======")
        G0 = compute_g(*theta_0, num_samples=n)
        thetas = [theta_1, theta_2]
        ps = [p1, p2]
        tests = {
            "Kolmogorov-Smirnov": _ks_test,
            "Cram´er–von Mises": _cm_test,
            "Anderson-Darling": _ad_test,
            "Dvoretzky–Kiefer–Wolfowitz Inequality": _dkw_test,
        }

        for theta, p in zip(thetas, ps):
            print(f"Testing with p={p} and theta={theta}")
            G = compute_g(*theta, num_samples=n)
            FY = (1 - p) * G0 + p * G

            for title, test in tests.items():
                print(f"\t{title}")
                test(FY, G0, alphas)


def _ks_test(ecdf: NDArray, cdf: NDArray, alphas: List[float]) -> None:
    p_value = stats.kstest(ecdf, cdf).pvalue
    print(f"\t\tP-value: {p_value:.5f}")
    for alpha in alphas:
        print(f"\t\tReject the null hypothesis: p-value < {alpha}(alpha) = {p_value < alpha}")


def _cm_test(ecdf: NDArray, cdf: NDArray, alphas: List[float]) -> None:
    p_value = stats.cramervonmises_2samp(ecdf, cdf).pvalue
    print(f"\t\tP-value: {p_value:.5f}")
    for alpha in alphas:
        print(f"\t\tReject the null hypothesis: p-value < {alpha}(alpha) = {p_value < alpha}")


def _ad_test(ecdf: NDArray, cdf: NDArray, alphas: List[float]) -> None:
    p_value = stats.anderson_ksamp(samples=[ecdf, cdf], method=stats.PermutationMethod()).pvalue
    print(f"\t\tP-value: {p_value:.5f}")
    for alpha in alphas:
        print(f"\t\tReject the null hypothesis: p-value < {alpha}(alpha) = {p_value < alpha}")


def _dkw_test(ecdf: NDArray, cdf: NDArray, alphas: List[float]) -> None:
    max_deviation = np.max(np.abs(ecdf - cdf))
    num_samples = len(ecdf)
    print(f"\t\tMax deviation: {max_deviation:.5f}")
    for alpha in alphas:
        epsilon = np.sqrt(np.log(2 / alpha) / (2 * num_samples))
        print(f"\t\tReject the null hypothesis: max-deviation > {epsilon:.5f}(epsilon) = {max_deviation > epsilon}")
