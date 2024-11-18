from typing import Tuple, List

from numpy.typing import NDArray
from scipy import stats


def estimate_pareto_parameters(data: NDArray, method: str = "MLE") -> Tuple[float, float]:
    shape, _, scale = stats.pareto.fit(data, floc=0, method=method)
    return scale.item(), shape.item()


def test_goodness_of_fit_with_bootstrap(
        theta_0: Tuple[float, float],
        theta_1: Tuple[float, float],
        theta_2: Tuple[float, float],
        p1: float,
        p2: float,
        alpha: float,
        sample_sizes: List[int],
        estimation_method: str = "MLE",
):
    thetas = [theta_1, theta_2]
    probabilities = [p1, p2]

    for N in sample_sizes:
        print(f"\n=======Testing with N={N} samples=======")
        G0 = stats.pareto.rvs(b=theta_0[1], scale=theta_0[0], size=N)

        for theta, p in zip(thetas, probabilities):
            print(f"Testing with p={p} and theta={theta}")
            G = stats.pareto.rvs(b=theta[1], scale=theta[0], size=N)
            FY = (1 - p) * G0 + p * G

            theta_N = estimate_pareto_parameters(FY, method=estimation_method)
            cdf = lambda x: stats.pareto.cdf(x, b=theta_N[1], scale=theta_N[0])
            T_N, _ = stats.kstest(FY, cdf)

            T_N_b_values = []
            B = 100 * N
            for _ in range(B):
                bootstrap_sample = stats.pareto.rvs(
                    b=theta_N[1], scale=theta_N[0], size=N
                )
                theta_N_b = estimate_pareto_parameters(
                    bootstrap_sample, method=estimation_method
                )
                cdf_b = lambda x: stats.pareto.cdf(
                    x, b=theta_N_b[1], scale=theta_N_b[0]
                )
                T_N_b, _ = stats.kstest(bootstrap_sample, cdf_b)
                T_N_b_values.append(T_N_b)

            p_value = sum(T_N_b > T_N for T_N_b in T_N_b_values) / B

            print(f"\talpha: {alpha}")
            print(f"\tp-value: {p_value:.5f}")
            print(f"\tReject the null hypothesis: {p_value < alpha}")
