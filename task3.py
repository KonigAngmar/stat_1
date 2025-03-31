import numpy as np
import scipy.stats as stats

def count_inversions(arr):
    count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                count += 1
    return count

def inversion_test(n, gamma=0.05):
    X = np.random.uniform(0, 2, n)  # Генеруємо вибірку
    k = count_inversions(X)
    
    z_gamma = stats.norm.ppf(1 - gamma / 2)  # Виправлено обчислення z_gamma
    threshold = (6 / (n * np.sqrt(n))) * abs(k - (n * (n - 1) / 4))
    
    result = "Гіпотеза H0 відхиляється" if threshold > z_gamma else "Гіпотеза H0 прийнята"
    print(f"n = {n}, k = {k}, threshold = {threshold}, z_gamma = {z_gamma}\n{result}")

# Перевіряємо три випадки
inversion_test(500)
inversion_test(5000)
inversion_test(10000)
