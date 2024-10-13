import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


N = 5  
S = 7   
I1 = 7  
I2 = 4  

tau = 1 / (1 + I1)
alpha1 = 0.1
alpha2 = 0.01
p1 = alpha1 ** (1 - tau) * alpha2 ** tau
p2 = 10 * np.sqrt(S) * p1

print(f"p1: {p1:.4f}, p2: {p2:.4f}")

