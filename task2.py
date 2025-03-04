import numpy as np
from scipy import stats

"""
Завдання 2: Обчислення ймовірності Q(α) = P{ξₐ < η} чотирма методами

Для в.в. ξₐ та η маємо:
  ξₐ = Fₐ⁻¹(ω) = ( -ln ω )^(1/4) / α,  
  η   = G⁻¹(ω) = ( -ln ω )^(1/2),
де ω – рівномірна на [0,1].

Метод 1: q̂ᵢ = I(ξₐ^(i) < ηᵢ)
Метод 2: q̂ᵢ = 1 - G(ξₐ^(i)) = exp( - [ξₐ^(i)]² )
Метод 3: q̂ᵢ = Fₐ(ηᵢ) = 1 - exp( - (αηᵢ)⁴ )
Метод 4: q̂ᵢ = 2/(βᵢ⁴) · [1 - exp( - (αβᵢ)⁴ )],
де βᵢ = √(θ₁ + θ₂ + θ₃) з θⱼ = -ln(ωⱼ), для ω₍₃ᵢ₋₂₎, ω₍₃ᵢ₋₁₎, ω₍₃ᵢ₎.

Нижче реалізовано усі методи.
"""

def method1(alpha, M):
    # Генеруємо ξₐ та η за відповідними оберненими функціями
    # Для ξₐ: ξₐ = (-ln ω)^(1/4) / α, для η: η = (-ln ω)^(1/2)
    omega = np.random.rand(M)
    xi = ((-np.log(omega)) ** 0.25) / alpha
    eta = (-np.log(omega)) ** 0.5
    return (xi < eta).astype(float)

def method2(alpha, M):
    # Використовуємо: q̂ᵢ = exp( - [ξₐ^(i)]² ), де ξₐ = (-ln ω)^(1/4) / α
    omega = np.random.rand(M)
    xi = ((-np.log(omega)) ** 0.25) / alpha
    return np.exp( - xi**2 )

def method3(alpha, M):
    # Використовуємо: q̂ᵢ = 1 - exp( - (αηᵢ)⁴ ), де η = (-ln ω)^(1/2)
    omega = np.random.rand(M)
    eta = (-np.log(omega)) ** 0.5
    return 1 - np.exp( - (alpha * eta)**4 )

def method4(alpha, M):
    # Для кожного запуску генеруємо 3 незалежних рівномірних в.в.
    # Обчислюємо θ = -ln(ω) для кожного та β = sqrt(θ₁+θ₂+θ₃)
    # Потім q̂ᵢ = 2/(β⁴)*[1 - exp( - (αβ)⁴)]
    omega = np.random.rand(M, 3)
    theta = -np.log(omega)
    beta_val = np.sqrt(np.sum(theta, axis=1))
    return (2.0 / (beta_val ** 4)) * (1 - np.exp( - (alpha * beta_val) ** 4))

def simulate_method(method_func, alpha, pilot_M=1000, epsilon=0.01, confidence=0.99, max_iter=1e7):
    z_quant = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    M = int(pilot_M)
    pilot_est = method_func(alpha, M)
    pilot_sum = np.sum(pilot_est)
    pilot_sum_sq = np.sum(pilot_est ** 2)
    
    Q_hat = pilot_sum / M
    var_hat = (pilot_sum_sq - M * Q_hat ** 2) / (M - 1) if M > 1 else 0.0

    if Q_hat == 0:
        return {
            'M_used': M,
            'Q_est': 0.0,
            'variance': 0.0,
            'ci': (0.0, 0.0),
            'ci_width': 0.0
        }
    
    N_required = int(np.ceil(M * (z_quant**2 * var_hat) / ((epsilon * Q_hat) ** 2)))
    if N_required > max_iter:
        N_required = int(max_iter)
        print(f"Warning: N_required capped at max_iter = {max_iter}")

    if N_required > M:
        extra_M = N_required - M
        extra_est = method_func(alpha, extra_M)
        total_sum = pilot_sum + np.sum(extra_est)
        total_sum_sq = pilot_sum_sq + np.sum(extra_est ** 2)
        N = N_required
        Q_final = total_sum / N
        var_final = (total_sum_sq - N * Q_final ** 2) / (N - 1) if N > 1 else 0.0
    else:
        N = M
        Q_final = Q_hat
        var_final = var_hat

    margin = z_quant * np.sqrt(var_final / N)
    ci = (Q_final - margin, Q_final + margin)

    return {
        'M_used': N,
        'Q_est': Q_final,
        'variance': var_final,
        'ci': ci,
        'ci_width': ci[1] - ci[0]
    }

def task2_simulation(alphas, pilot_M=1000, epsilon=0.01, confidence=0.99, max_iter=1e7):
    methods = {
        'Метод 1': method1,
        'Метод 2': method2,
        'Метод 3': method3,
        'Метод 4': method4
    }
    
    results = {}
    for alpha in alphas:
        alpha_res = {}
        true_Q = alpha / (alpha + 1)  # аналітичне значення Q(α)
        for method_name, method_func in methods.items():
            res = simulate_method(method_func, alpha, pilot_M, epsilon, confidence, max_iter)
            res['true_Q'] = true_Q
            res['error'] = abs(res['Q_est'] - true_Q)
            alpha_res[method_name] = res
        results[alpha] = alpha_res
    return results

def task2_main():
    alphas = [1, 0.3, 0.1]
    results = task2_simulation(alphas, pilot_M=1000, epsilon=0.01, confidence=0.99, max_iter=1e7)
    for alpha in alphas:
        print(f"\nЗавдання 2. Параметр α = {alpha}, аналітичне Q(α) = {alpha/(alpha+1):.5f}")
        for method_name, res in results[alpha].items():
            print(f"{method_name}:")
            print(f"  Кількість симуляцій: {res['M_used']}")
            print(f"  Оцінка Q(α): {res['Q_est']:.5f}")
            print(f"  Вибіркова дисперсія: {res['variance']:.5e}")
            print(f"  Довірчий інтервал: ({res['ci'][0]:.5f}, {res['ci'][1]:.5f})")
            print(f"  Ширина CI: {res['ci_width']:.5f}")
            print(f"  Абсолютна похибка: {res['error']:.5e}")