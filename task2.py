import numpy as np
from scipy.stats import spearmanr, kendalltau

def test_spearman(n, variant='a', alpha=0.05):
    """
    Перевірка гіпотези незалежності за допомогою критерію Спірмена.

    Варіанти:
    a) Y_i = ξ_i * η_i
    b) Y_i = ξ_i + η_i
    де ξ_i, η_i ~ U[-1, 1]
    """
    np.random.seed(42)

    xi = np.random.uniform(-1, 1, n)
    eta = np.random.uniform(-1, 1, n)
    X = xi

    if variant == 'a':
        Y = xi * eta
    elif variant == 'b':
        Y = xi + eta
    else:
        raise ValueError("Варіант має бути 'a' або 'b'")

    rho, p_value = spearmanr(X, Y)
    hypothesis = "Підтверджується" if p_value > alpha else "Відхиляється"
    return rho, p_value, hypothesis

def test_kendall(n, variant='a', alpha=0.05):
    """
    Перевірка гіпотези незалежності за допомогою критерію Кендалла.

    Варіанти:
    a) Y_i = ξ_i * η_i
    b) Y_i = ξ_i + η_i
    де ξ_i, η_i ~ U[-1, 1]
    """
    np.random.seed(42)

    xi = np.random.uniform(-1, 1, n)
    eta = np.random.uniform(-1, 1, n)
    X = xi

    if variant == 'a':
        Y = xi * eta
    elif variant == 'b':
        Y = xi + eta
    else:
        raise ValueError("Варіант має бути 'a' або 'b'")

    tau, p_value = kendalltau(X, Y)
    hypothesis = "Підтверджується" if p_value > alpha else "Відхиляється"
    return tau, p_value, hypothesis
