import numpy as np
import scipy.stats as stats


def generate_sample(size, lambd):
    return np.random.exponential(1 / lambd, size)


def kolmogorov_test(sample, lambd):
    transformed_sample = 1 - np.exp(-lambd * sample)
    d_statistic, p_value = stats.kstest(transformed_sample, 'uniform')
    return d_statistic, p_value


def chi_square_test(sample, k, lambd):
    transformed_sample = 1 - np.exp(-lambd * sample)  # Перетворення для рівномірності
    observed, _ = np.histogram(transformed_sample, bins=k, range=(0, 1))
    expected = np.full(k, len(sample) / k)  # Очікувані значення
    chi2_statistic, p_value = stats.chisquare(observed, expected)
    return chi2_statistic, p_value

