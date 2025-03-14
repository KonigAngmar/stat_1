import numpy as np
from scipy.stats import ks_2samp

def smirnov_test_a(n):
    np.random.seed(42)
    X1 = np.random.exponential(scale=1/1.0, size=n)
    X2 = np.random.exponential(scale=1/1.0, size=n)
    
    stat, p_value = ks_2samp(X1, X2)
    return stat, p_value

if __name__ == "__main__":
    for n in [1000, 10000, 100000]:
        stat, p_value = smirnov_test_a(n)
        print(f"n={n}, Smirnov Statistic={stat:.4f}, p-value={p_value:.4f}")
