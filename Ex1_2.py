import numpy as np
import scipy.stats as stats


def generate_sample(size, lambd):
    return np.random.exponential(1 / lambd, size)


def kolmogorov_test(sample):
    transformed_sample = 1 - np.exp(-sample)  # Перетворення для рівномірності
    d_statistic, p_value = stats.kstest(transformed_sample, 'uniform')
    return d_statistic, p_value


def chi_square_test(sample, k):
    transformed_sample = 1 - np.exp(-sample)  # Перетворення для рівномірності
    observed, _ = np.histogram(transformed_sample, bins=k, range=(0, 1))
    expected = np.full(k, len(sample) / k)  # Очікувані значення
    chi2_statistic, p_value = stats.chisquare(observed, expected)
    return chi2_statistic, p_value


# Параметри
sizes = [1000, 10000, 100000]
lambdas = [1, 1.2]  # Два значення λ
alpha = 0.05
k_values = [int(np.ceil(30 * size / 1000)) for size in sizes]  # Обираємо k

for size, k in zip(sizes, k_values):
    for lambd in lambdas:
        sample = generate_sample(size, lambd)

        # Критерій Колмогорова
        d_stat, p_kolm = kolmogorov_test(sample)
        print(f"Kolmogorov test (n={size}, lambda={lambd}): D={d_stat:.4f}, p={p_kolm:.4f}")
        print("Hypothesis rejected" if p_kolm < alpha else "Hypothesis not rejected")

        # Критерій χ²
        chi2_stat, p_chi2 = chi_square_test(sample, k)
        print(f"Chi-square test (n={size}, k={k}, lambda={lambd}): χ²={chi2_stat:.4f}, p={p_chi2:.4f}")
        print("Hypothesis rejected" if p_chi2 < alpha else "Hypothesis not rejected")
        print("-" * 50)
