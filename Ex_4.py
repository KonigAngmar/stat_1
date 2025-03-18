from Funcs import generate_sample
import numpy as np
from scipy.stats import ks_2samp


def task4():
    sizes = [1000, 10000, 100000]
    lambdas = [1, 1.2]  # Два значення λ
    alpha = 0.05

    for size in sizes:
        for lambd in lambdas:
            stat, p_value = smirnov_test(size, lambda_0=1.0, lambda_1=lambd)
            print(
                f"Smirnov test (n={size}, lambda={lambd}): Stat={stat:.4f}, p-value={p_value:.4f}"
            )
            print(
                "Hypothesis rejected" if p_value < alpha else "Hypothesis not rejected"
            )
            print("_-" * 25)


def smirnov_test(n, lambda_0=1.0, lambda_1=1.0):

    m = n // 2  # розмір другої вибірки m = n/2
    X1 = generate_sample(n, lambda_0)
    X2 = generate_sample(m, lambda_1)
    stat, p_value = ks_2samp(X1, X2)
    return stat, p_value
