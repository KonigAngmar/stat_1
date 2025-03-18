from Funcs import generate_sample  # або з вашого модуля, де оголошено generate_sample
import numpy as np
from scipy.stats import ks_2samp


def task4():
    sizes = [1000, 10000, 100000]
    lambdas = [1, 1.2]  # Два значення λ
    alpha = 0.05

    for size in sizes:
        print("-" * 50)
        for lambd in lambdas:
            stat, p_value = smirnov_test(size, lambd)
            print(
                f"Smirnov test (n={size}, lambda={lambd}): Stat={stat:.4f}, p-value={p_value:.4f}"
            )
            print(
                "Hypothesis rejected" if p_value < alpha else "Hypothesis not rejected"
            )
    print("-" * 50)
    print("\n")

def smirnov_test(n, lambda_2=1.0):
    """
    Перевіряє гіпотезу про однорідність двох вибірок за допомогою
    критерію Смирнова (двовибірковий KS-тест).

    :param n: обсяг першої вибірки
    :param lambda_2: параметр другої вибірки (1 або 1.2)
    :return: (stat, p_value)
    """
    # Розмір другої вибірки m = n/2
    m = n // 2

    # Перша вибірка з λ = 1 (за умовою)
    X1 = generate_sample(n, 1.0)

    # Друга вибірка з λ = lambda_2
    X2 = generate_sample(m, lambda_2)

    # Двовибірковий критерій Колмогорова–Смирнова (KS)
    stat, p_value = ks_2samp(X1, X2)
    return stat, p_value
