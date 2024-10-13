import numpy as np
from scipy.stats import pareto, ks_2samp, anderson_ksamp
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.distributions.empirical_distribution import ECDF
from sympy import symbols, Eq, solve, sqrt

# Variables
N = symbols('N', positive=True, real=True)  # Length of Name
S = symbols('S', positive=True, real=True)  # Length of Surname
theta1_1, theta1_2 = symbols('theta1_1 theta1_2', positive=True, real=True)  # Parameters for theta1
theta2_1, theta2_2 = symbols('theta2_1 theta2_2', positive=True, real=True)  # Parameters for theta2

# Mean and variance formulas for Pareto distribution
def pareto_mean(theta_1, theta_2):
    return (theta_1 * theta_2) / (theta_2 - 1)

def pareto_variance(theta_1, theta_2):
    return (theta_1**2 * theta_2) / ((theta_2 - 2) * (theta_2 - 1)**2)

# Basic distribution parameters for theta0 = (N, S+2)
mu_0 = pareto_mean(N, S+2)
v0_sq = pareto_variance(N, S+2)

# Equation set 1: Solving for theta1
mu_eq1 = Eq(mu_0, pareto_mean(theta1_1, theta1_2))
var_eq1 = Eq(N * v0_sq, pareto_variance(theta1_1, theta1_2))

# Solving for theta1_1 and theta1_2
theta1_solution = solve([mu_eq1, var_eq1], (theta1_1, theta1_2))

# Equation set 2: Solving for theta2
mu_eq2 = Eq(mu_0 + 2 * sqrt(v0_sq), pareto_mean(theta2_1, theta2_2))
var_eq2 = Eq(v0_sq, S * pareto_variance(theta2_1, theta2_2))

# Solving for theta2_1 and theta2_2
theta2_solution = solve([mu_eq2, var_eq2], (theta2_1, theta2_2))

theta1_solution, theta2_solution




# Parameters from the task
alpha1 = 0.1
alpha2 = 0.01
I1 = 4  # Placeholder for the last digit of the study book number
S = 5   # Length of the surname
N_name = 7  # Length of the name

# Probabilities for mixture model
tau = 1 / (1 + I1)
p1 = (alpha1 ** (1 - tau)) * (alpha2 ** tau)
p2 = 10 * np.sqrt(S) * p1

# Sample sizes
N1 = 10 * (2 + N_name)
N2 = 100 * (2 + N_name)

# Define the Pareto distribution parameters
theta0 = (N_name, S+2)
theta1 = theta1_solution
theta2 = theta2_solution

# Generate samples from the mixture models
def generate_pareto_samples(p, theta0, theta1, size):
    samples = []
    for _ in range(size):
        if np.random.rand() < p:  # Select from G1 or G2 based on probability p
            samples.append(pareto.rvs(b=theta1[1], scale=theta1[0]))
        else:
            samples.append(pareto.rvs(b=theta0[1], scale=theta0[0]))
    return np.array(samples)

# Generate samples for p1 and p2
samples_p1 = generate_pareto_samples(p1, theta0, theta1, N1)
samples_p2 = generate_pareto_samples(p2, theta0, theta2, N2)

# Kolmogorov-Smirnov test
ks_stat_p1, ks_pvalue_p1 = ks_2samp(samples_p1, pareto.rvs(b=theta0[1], scale=theta0[0], size=N1))
ks_stat_p2, ks_pvalue_p2 = ks_2samp(samples_p2, pareto.rvs(b=theta0[1], scale=theta0[0], size=N2))

# Anderson-Darling test
ad_stat_p1, _, ad_pvalue_p1 = anderson_ksamp([samples_p1, pareto.rvs(b=theta0[1], scale=theta0[0], size=N1)])
ad_stat_p2, _, ad_pvalue_p2 = anderson_ksamp([samples_p2, pareto.rvs(b=theta0[1], scale=theta0[0], size=N2)])

# Cramér-von Mises test (via Lilliefors approximation)
cvm_stat_p1, cvm_pvalue_p1 = lilliefors(samples_p1, dist='pareto')
cvm_stat_p2, cvm_pvalue_p2 = lilliefors(samples_p2, dist='pareto')

# Dvoretzky–Kiefer–Wolfowitz Inequality
def dvoretzky_kiefer_wolfowitz(samples, alpha):
    ecdf = ECDF(samples)
    n = len(samples)
    epsilon = np.sqrt(np.log(2/alpha) / (2*n))
    return epsilon

# Compute DK-W bound
epsilon_p1 = dvoretzky_kiefer_wolfowitz(samples_p1, alpha1)
epsilon_p2 = dvoretzky_kiefer_wolfowitz(samples_p2, alpha2)

# Output results
results = {
    'KS_Test': {
        'p1': {'stat': ks_stat_p1, 'p-value': ks_pvalue_p1},
        'p2': {'stat': ks_stat_p2, 'p-value': ks_pvalue_p2},
    },
    'AD_Test': {
        'p1': {'stat': ad_stat_p1, 'p-value': ad_pvalue_p1},
        'p2': {'stat': ad_stat_p2, 'p-value': ad_pvalue_p2},
    },
    'CVM_Test': {
        'p1': {'stat': cvm_stat_p1, 'p-value': cvm_pvalue_p1},
        'p2': {'stat': cvm_stat_p2, 'p-value': cvm_pvalue_p2},
    },
    'DKW_Bound': {
        'p1': epsilon_p1,
        'p2': epsilon_p2,
    }
}

results
