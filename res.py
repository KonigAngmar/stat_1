from task1 import empty_blocks_test
from task2 import test_kendall, test_spearman
from task3 import randomness_test

if __name__ == "__main__":
    for n, m in [(500, 1000), (5000, 10000), (50000, 100000)]:
        chi_stat, critical_value, hypothesis = empty_blocks_test(n, m)
        print(
            f"Task 1 | n={n}, m={m}: Chi-Square={chi_stat:.4f}, Critical Value={critical_value:.4f}, Гіпотеза: {hypothesis}"
        )

    sample_sizes = [500, 5000, 50000]
    alpha = 0.05
    print("-" * 50)
    for variant in ["a", "b"]:
        print(f"Task 2 | Спірмен, Варіант {variant}:")
        for n in sample_sizes:
            rho, p_value, hypothesis = test_spearman(n, variant, alpha)
            print(
                f"n={n}: Spearman rho = {rho:.4f}, p-value = {p_value:.4f}, Гіпотеза: {hypothesis}"
            )
    print("")
    for variant in ["a", "b"]:
        print(f"Task 2 | Кендалл, Варіант {variant}:")
        for n in sample_sizes:
            tau, p_value, hypothesis = test_kendall(n, variant, alpha)
            print(
                f"n={n}: Kendall tau = {tau:.4f}, p-value = {p_value:.4f}, Гіпотеза: {hypothesis}"
            )

    print("-" * 50)

    for n in [500, 5000, 10000]:
        z_stat, critical_value, hypothesis = randomness_test(n)
        print(
            f"Task 3 | n={n}: Z-Statistic={z_stat:.4f}, Critical Value={critical_value:.4f}, Гіпотеза: {hypothesis}"
        )
