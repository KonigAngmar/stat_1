from Ex1_2 import generate_sample, kolmogorov_test


sizes = [1000, 10000, 100000]
lambdas = [1, 1.2]  # Два значення λ
alpha = 0.05

for size in sizes:
    for lambd in lambdas:
        sample = generate_sample(size, lambd)
        # Критерій Колмогорова
        d_stat, p_kolm = kolmogorov_test(sample, lambd)
        print(f"Kolmogorov test (n={size}, lambda={lambd}): D={d_stat:.4f}, p={p_kolm:.4f}")
        print("Hypothesis rejected" if p_kolm < alpha else "Hypothesis not rejected")
        print("_-" * 25)