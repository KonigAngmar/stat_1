from task1 import empty_blocks_test
from task3 import randomness_test

if __name__ == "__main__":
    for n, m in [(500, 1000), (5000, 10000), (50000, 100000)]:
        chi_stat, critical_value, hypothesis = empty_blocks_test(n, m)
        print(f"Task 1 | n={n}, m={m}: Chi-Square={chi_stat:.4f}, Critical Value={critical_value:.4f}, Гіпотеза: {hypothesis}")

    for n in [500, 5000, 10000]:
        z_stat, critical_value, hypothesis = randomness_test(n)
        print(f"Task 3 | n={n}: Z-Statistic={z_stat:.4f}, Critical Value={critical_value:.4f}, Гіпотеза: {hypothesis}")
