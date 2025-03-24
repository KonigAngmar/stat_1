import numpy as np
from scipy.stats import spearmanr, kendalltau

def test_spearman(n, variant='a', alpha=0.05):
    """
    Перевірка гіпотези незалежності за допомогою критерія Спірмена.
    
    Параметри:
    n       - розмір вибірки;
    variant - варіант генерації Y:
              'a' - Y ~ U[0,1] (незалежна від X);
              'b' - Y = 1 - X (повна монотонна залежність);
    alpha   - рівень значимості.
    
    Повертає:
    rho      - коефіцієнт Спірмена,
    p_value  - p-value тесту,
    hypothesis - "Підтверджується", якщо p_value > alpha, інакше "Відхиляється".
    """
    np.random.seed(42)
    X = np.random.uniform(0, 1, n)
    if variant == 'a':
        Y = np.random.uniform(0, 1, n)
    elif variant == 'b':
        Y = 1 - X  # забезпечує повну негативну залежність
    else:
        raise ValueError("Вказано невірний варіант. Оберіть 'a' або 'b'.")
    
    rho, p_value = spearmanr(X, Y)
    hypothesis = "Підтверджується" if p_value > alpha else "Відхиляється"
    return rho, p_value, hypothesis

def test_kendall(n, variant='a', alpha=0.05):
    """
    Перевірка гіпотези незалежності за допомогою критерія Кендалла.
    
    Параметри:
    n       - розмір вибірки;
    variant - варіант генерації Y:
              'a' - Y ~ U[0,1] (незалежна від X);
              'b' - Y = 1 - X (повна монотонна залежність);
    alpha   - рівень значимості.
    
    Повертає:
    tau      - коефіцієнт Кендалла,
    p_value  - p-value тесту,
    hypothesis - "Підтверджується", якщо p_value > alpha, інакше "Відхиляється".
    """
    np.random.seed(42)
    X = np.random.uniform(0, 1, n)
    if variant == 'a':
        Y = np.random.uniform(0, 1, n)
    elif variant == 'b':
        Y = 1 - X
    else:
        raise ValueError("Вказано невірний варіант. Оберіть 'a' або 'b'.")
    
    tau, p_value = kendalltau(X, Y)
    hypothesis = "Підтверджується" if p_value > alpha else "Відхиляється"
    return tau, p_value, hypothesis
