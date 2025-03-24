import numpy as np
from scipy.stats import norm

def randomness_test(n, alpha=0.05):
    np.random.seed(42)
    
    # Генеруємо вибірку
    X = np.array([np.mean(np.random.uniform(-1, 1, i)) for i in range(1, n+1)])
    
    # Підрахунок кількості інверсій
    inversions = sum(1 for i in range(n) for j in range(i+1, n) if X[i] > X[j])
    
    # Очікуване значення і дисперсія
    expected = n * (n - 1) / 4
    variance = n * (n - 1) * (2 * n + 5) / 72
    
    # Z-критерій
    z_stat = (inversions - expected) / np.sqrt(variance)
    critical_value = norm.ppf(1 - alpha / 2)  # двосторонній критерій
    
    # Висновок
    hypothesis = "Підтверджується" if abs(z_stat) < critical_value else "Відхиляється"
    
    return z_stat, critical_value, hypothesis

if __name__ == "__main__":
    for n in [500, 5000, 50000]:
        z_stat, critical_value, hypothesis = randomness_test(n)
        print(f"Task 3 | n={n}: Z-Statistic={z_stat:.4f}, Critical Value={critical_value:.4f}, Гіпотеза: {hypothesis}")
