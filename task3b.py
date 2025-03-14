import numpy as np
from scipy.stats import chi2

def empty_boxes_test_b(n, lambda_0=1.0, lambda_true=2.0):
    np.random.seed(42)
    X = np.random.exponential(scale=1/lambda_true, size=n)
    Y = 1 - np.exp(-lambda_0 * X)

    k = int(2 * (n ** (1/3)))
    bins = np.linspace(0, 1, k + 1)
    counts, _ = np.histogram(Y, bins=bins)
    
    expected_count = n / k
    chi_stat = np.sum((counts - expected_count) ** 2 / expected_count)
    p_value = 1 - chi2.cdf(chi_stat, df=k-1)

    return chi_stat, p_value

if __name__ == "__main__":
    for n in [1000, 10000, 100000]:
        chi_stat, p_value = empty_boxes_test_b(n)
        print(f"n={n}, Chi-Square Statistic={chi_stat:.4f}, p-value={p_value:.4f}")
