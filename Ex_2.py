import numpy as np
from scipy import stats
from Funcs import generate_sample


def task2():
    sizes = [1000, 10000, 100000]
    lambdas = [1, 1.2]  # Два значення λ
    alpha = 0.05
    k_values = [int(np.ceil(30 * size / 1000)) for size in sizes]  # Обираємо k

    for size, k in zip(sizes, k_values):
        print("-" * 50)

        for lambd in lambdas:
            sample = generate_sample(size, lambd)

            # Критерій χ²
            chi2_stat, p_chi2 = chi_square_test(sample, k, 1)
            print(
                f"Chi-square test (n={size}, k={k}, lambda={lambd}): χ²={chi2_stat:.4f}, p={p_chi2}"
            )
            print(
                "Hypothesis rejected" if p_chi2 < alpha else "Hypothesis not rejected"
            )
    print("-" * 50)
    print("\n")


def chi_square_test(sample, k, lambd):
    transformed_sample = 1 - np.exp(-lambd * sample)  # Перетворення для рівномірності
    observed, _ = np.histogram(transformed_sample, bins=k, range=(0, 1))
    expected = np.full(k, len(sample) / k)  # Очікувані значення
    chi2_statistic, p_value = stats.chisquare(observed, expected)
    return chi2_statistic, p_value
