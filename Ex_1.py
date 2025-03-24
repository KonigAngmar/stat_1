import numpy as np
from scipy import stats
from Funcs import generate_sample

def task1():
    sizes = [1000, 10000, 100000]  # Розміри вибірок
    lambdas = [1, 1.2]  # Значення λ
    alpha = 0.05  # Рівень значущості

    for size in sizes:
        print("-" * 50)
        print(f"Sample size: {size}")

        for lambd in lambdas:
            sample = generate_sample(size, lambd)

            # Перевіряємо середнє значення для контролю
            print(f"Mean for lambda={lambd}: {np.mean(sample):.4f}")

            # Виконуємо критерій Колмогорова-Смирнова для перевірки
            d_stat, p_value = stats.kstest(sample, 'expon', args=(0, 1))  # Перевіряємо з теоретичним λ=1

            print(f"Kolmogorov test (n={size}, lambda={lambd}): D={d_stat:.4f}, p={p_value:.4f}")
            print("Hypothesis rejected" if p_value < alpha else "Hypothesis not rejected")

    print("-" * 50)
    print("\n")

