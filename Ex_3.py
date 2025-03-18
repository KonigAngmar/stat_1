import numpy as np
from Funcs import generate_sample
from scipy.stats import chi2


def task3():
    sizes = [1000, 10000, 100000]
    lambdas = [1, 1.2]  # Два значення λ
    alpha = 0.05

    for size in sizes:
        for lambd in lambdas:
            chi_stat, p_value = empty_boxes_test(size, lambda_0=1.0, lambda_1=lambd)
            print(
                f"Empty boxes test (n={size}, lambda={lambd}): Chi-Square={chi_stat:.4f}, p-value={p_value:.4f}"
            )
            print(
                "Hypothesis rejected" if p_value < alpha else "Hypothesis not rejected"
            )
            print("_-" * 25)


def empty_boxes_test(n, lambda_0=1.0, lambda_1=1.0):
    X = generate_sample(n, lambda_1)
    Y = 1 - np.exp(-lambda_0 * X)

    k = int(2 * (n ** (1 / 3)))
    bins = np.linspace(0, 1, k + 1)
    counts, _ = np.histogram(Y, bins=bins)

    expected_count = n / k
    chi_stat = np.sum((counts - expected_count) ** 2 / expected_count)
    p_value = 1 - chi2.cdf(chi_stat, df=k - 1)

    return chi_stat, p_value
