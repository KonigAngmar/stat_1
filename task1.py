import numpy as np
from scipy.stats import chi2

def empty_blocks_test(n, m, alpha=0.05):
    np.random.seed(42)
    
    # Генеруємо дві вибірки
    X = np.random.exponential(scale=1.0, size=n)
    Y = np.random.exponential(scale=1/1.2, size=m)
    
    # Об'єднуємо та сортуємо
    combined = np.sort(np.concatenate([X, Y]))
    
    # Лічильник пустих блоків
    empty_blocks = 0
    i, j = 0, 0
    while i < n and j < m:
        if X[i] < Y[j]:
            i += 1
        elif Y[j] < X[i]:
            j += 1
        else:
            i += 1
            j += 1
        if i == n or j == m:
            empty_blocks += 1

    # Обчислення критерію хі-квадрат
    chi_stat = (empty_blocks - (n / (n + m) * m))**2 / (n / (n + m) * m)
    critical_value = chi2.ppf(1 - alpha, df=1)
    
    # Висновок
    hypothesis = "Підтверджується" if chi_stat < critical_value else "Відхиляється"
    
    return chi_stat, critical_value, hypothesis

if __name__ == "__main__":
    for n, m in [(500, 1000), (5000, 10000), (50000, 100000)]:
        chi_stat, critical_value, hypothesis = empty_blocks_test(n, m)
        print(f"Task 1 | n={n}, m={m}: Chi-Square={chi_stat:.4f}, Critical Value={critical_value:.4f}, Гіпотеза: {hypothesis}")
