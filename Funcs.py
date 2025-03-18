import numpy as np
import scipy.stats as stats


def generate_sample(size, lambd):
    return np.random.exponential(1 / lambd, size)
