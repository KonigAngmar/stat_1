import numpy as np
from Funcs import generate_sample
from scipy.stats import chi2


def task3():
    sizes = [1000, 10000, 100000]
    lambdas = [1, 1.2]  # Два значення λ
    alpha = 0.05

    for size in sizes:
        print("-" * 50)    
        for lambd in lambdas:
            chi_stat, p_value = empty_boxes_test(size, lambda_0=1.0, lambda_1=lambd)
            print(
                f"Empty boxes test (n={size}, lambda={lambd}): Chi-Square={chi_stat:.4f}, p-value={p_value:.4f}"
            )
            print(
                "Hypothesis rejected" if p_value < alpha else "Hypothesis not rejected"
            )
    print("-" * 50)
    print("\n")


def empty_boxes_test(n, lambda_0=1.0, lambda_1=1.0):
    """
    Перевіряє гіпотезу H0: X_i ∼ F(u; lambda_0) за допомогою критерію пустих ящиків.
    Дані генеруються з експоненційного розподілу з параметром lambda_1.
    :param n: обсяг вибірки
    :param lambda_0: параметр в нульовій гіпотезі (за умовою = 1)
    :param lambda_1: справжній параметр (1 для H0, 1.2 для H1)
    :return: (chi_stat, p_value)
    """
    # Генеруємо вибірку обсягу n з експоненційного розподілу з параметром lambda_1
    X = generate_sample(n, lambda_1)

    # Перетворення для перевірки на рівномірність
    Y = 1 - np.exp(-lambda_0 * X)

    # За умовою р = 2 => n/r = 2 => r = n/2
    r = n // 2  # цілочисельний поділ
    bins = np.linspace(0, 1, r + 1)

    # Рахуємо кількість потраплянь у кожен із r проміжків
    counts, _ = np.histogram(Y, bins=bins)

    # Очікувана кількість спостережень у кожному інтервалі
    expected_count = n / r

    # Критерій χ² для "пустих ящиків" (в асимптотичному наближенні)
    chi_stat = np.sum((counts - expected_count) ** 2 / expected_count)
    p_value = 1 - chi2.cdf(chi_stat, df=r - 1)

    return chi_stat, p_value
