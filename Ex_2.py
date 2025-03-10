import numpy as np
from Ex1_2 import generate_sample, chi_square_test


sizes = [1000, 10000, 100000]
lambdas = [1, 1.2]  # Два значення λ
alpha = 0.05
k_values = [int(np.ceil(30 * size / 1000)) for size in sizes]  # Обираємо k

for size, k in zip(sizes, k_values):
    for lambd in lambdas:
        sample = generate_sample(size, lambd)

        # Критерій χ²
        chi2_stat, p_chi2 = chi_square_test(sample, k)
        print(f"Chi-square test (n={size}, k={k}, lambda={lambd}): χ²={chi2_stat:.4f}, p={p_chi2:.4f}")
        print("Hypothesis rejected" if p_chi2 < alpha else "Hypothesis not rejected")
        print("_-" * 25)