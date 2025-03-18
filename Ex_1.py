import numpy as np
from scipy import stats
from Funcs import generate_sample


def task1():
    sizes = [1000, 10000, 100000]
    lambdas = [1, 1.2]  # Два значення λ
    alpha = 0.05

    for size in sizes:
        print("-" * 50)
        for lambd in lambdas:
            sample = generate_sample(size, lambd)
            # Критерій Колмогорова
            d_stat, p_kolm = kolmogorov_test(sample, lambd)
            print(
                f"Kolmogorov test (n={size}, lambda={lambd}): D={d_stat:.4f}, p={p_kolm:.4f}"
            )
            print(
                "Hypothesis rejected" if p_kolm < alpha else "Hypothesis not rejected"
            )
    print("-" * 50)
    print("\n")


def kolmogorov_test(sample, lambd):
    transformed_sample = 1 - np.exp(-lambd * sample)
    d_statistic, p_value = stats.kstest(transformed_sample, "uniform")
    return d_statistic, p_value
