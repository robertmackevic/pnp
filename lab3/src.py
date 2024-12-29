from typing import Optional, List

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import stats
from scipy.spatial import KDTree
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad

def gaussian_kernel(x: float)->float:
    """
    Standard Gaussian kernel function
    """
    return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)

def kde_leave_one_out(x: NDArray, h: float, eval_points: NDArray)->NDArray:
    """
    Compute leave-one-out kernel density estimate
    
    Parameters:
    x: array of observations
    h: bandwidth
    eval_points: points at which to evaluate the density
    """
    n = len(x)
    estimates = np.zeros_like(eval_points)
    
    for i in range(len(eval_points)):
        kernel_sum = 0
        for j in range(n):
            kernel_sum += gaussian_kernel((eval_points[i] - x[j])/h)
        estimates[i] = kernel_sum/(n*h)
    
    return estimates

def lscv_loss_with_integral(x: NDArray, h: float)-> float:
    """
    Compute LSCV loss for a given bandwidth h
    Returns:
    LSCV loss value
    """
    n = len(x)
    
    # First term: integral of squared density estimate
    def squared_kde(t):
        kde = kde_leave_one_out(x, h, np.array([t]))
        return kde[0]**2
    
    integral, _ = quad(squared_kde, min(x)-3*h, max(x)+3*h)
    
    # Second term: leave-one-out estimates
    loo_sum = 0
    for i in range(n):
        x_without_i = np.delete(x, i)
        xi = x[i]
        loo_estimate = kde_leave_one_out(x_without_i, h, np.array([xi]))
        loo_sum += loo_estimate[0]
    
    return integral - (2/n) * loo_sum

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



import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad

# Kernel and its second derivative
def kernel_function(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def kernel_second_derivative(x):
    return (x**2 - 1) * kernel_function(x)

# Roughness functional R(K)
def roughness_functional(kernel_sec_deriv):
    return quad(lambda u: kernel_sec_deriv(u)**2, -np.inf, np.inf)[0]

# Estimate density derivatives
def estimate_density_second_derivative(data, h):
    kde = gaussian_kde(data, bw_method=h)
    second_derivative = lambda x: kde.evaluate(x)  # Approximate for second derivative
    return second_derivative

def refined_plugin_bandwidth_selection(data: NDArray, max_iter: int = 100, tol: float = 1e-5) -> float:
    n = len(data)
    std_dev = np.std(data)
    initial_bw = (4 * std_dev ** 5 / (3 * n)) ** (1 / 5)

    n = len(data)
    h_pilot = np.std(data) * n**(-1/5)  # Initial pilot bandwidth
    kde_pilot = gaussian_kde(data, bw_method=h_pilot)
    
    # Estimate g''(u)
    g_second_derivative = estimate_density_second_derivative(data, h_pilot)
    
    # Compute roughness term R(K)
    R_K = roughness_functional(kernel_second_derivative)
    
    # Plug-in bandwidth formula
    integrated_second_derivative = quad(lambda x: g_second_derivative(x)**2, -np.inf, np.inf)[0]
    h = (R_K / (n * integrated_second_derivative))**(1/5)
    return h


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
