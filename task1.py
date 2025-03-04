import numpy as np
from scipy.stats import t, chi2, norm
"""t – t-розподіл Стьюдента (використовується для довірчого інтервалу коли дисперсія невідома).
chi2 – хі-квадрат розподіл (використовується для довірчого інтервалу дисперсії).
norm – стандартний нормальний розподіл (використовується для великої вибірки за ЦГТ)."""

def generate_normal_data(n):


    theta1 = np.random.rand(n) # від 0 до 1
    theta2 = np.random.rand(n)
    R = np.sqrt(-2 * np.log(theta1))
    Xi = R * np.sin(2 * np.pi * theta2)  # перетворення через sin
    Eta = R * np.cos(2 * np.pi * theta2)  # перетворення через cos
    samples = np.concatenate([Xi, Eta]) # об'єднання


    return samples


def ci_mean_normal(data, gamma=0.99):
    n = len(data) # розмір вибірки.
    alpha = 1 - gamma
    x_bar = np.mean(data) # вибіркове середнє.
    s = np.std(data, ddof=1)  # вибіркове стандартне відхилення

    t_val = t.ppf(1 - alpha / 2, df=n-1) # квантиль t-розподілу

# формула довірчого інтервалу
    half_width = t_val * s / np.sqrt(n)
    left = x_bar - half_width
    right = x_bar + half_width

    print("Довірчий інтервал для математичного сподівання (норм. випадок):")
    print(f"  Кількість реалізацій: {n}")
    print(f"  Оцінка (x̄): {x_bar:.5f}")
    print(f"  Довірчий інтервал: ({left:.5f}, {right:.5f})")
    print(f"  Ширина інтервалу: {right - left:.5f}\n")


def ci_variance_normal(data, gamma=0.99):

    n = len(data)
    alpha = 1 - gamma
    s2 = np.var(data, ddof=1)  # вибіркова дисперсія

    # квантилі хі-квадрат розподілу
    chi2_left = chi2.ppf(1 - alpha / 2, df=n - 1)
    chi2_right = chi2.ppf(alpha / 2, df=n - 1)

# формула інтервалу
    left = (n - 1) * s2 / chi2_left
    right = (n - 1) * s2 / chi2_right

    print("Довірчий інтервал для дисперсії (норм. випадок):")
    print(f"  Кількість реалізацій: {n}")
    print(f"  Оцінка (S^2): {s2:.5f}")
    print(f"  Довірчий інтервал: ({left:.5f}, {right:.5f})")
    print(f"  Ширина інтервалу: {right - left:.5f}\n")


def ci_mean_unknown_dist(data, gamma=0.99):
    n = len(data)
    alpha = 1 - gamma
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)

    z_val = norm.ppf(1 - alpha / 2) #квантиль стандартного нормального розподілу

    half_width = z_val * s / np.sqrt(n)
    left = x_bar - half_width
    right = x_bar + half_width

    print("Довірчий інтервал для математичного сподівання (без припущення про розподіл):")
    print(f"  Кількість реалізацій: {n}")
    print(f"  Оцінка (x̄): {x_bar:.5f}")
    print(f"  Довірчий інтервал: ({left:.5f}, {right:.5f})")
    print(f"  Ширина інтервалу: {right - left:.5f}\n")


def task1_main():

    n = [50,5000, 500000]
    for i in n:
        data = generate_normal_data(i)

        ci_mean_normal(data)

        ci_variance_normal(data)

        ci_mean_unknown_dist(data)



