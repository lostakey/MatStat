import numpy as np
from scipy import stats
import warnings

def chi2_test_normal(data, alpha=0.05):
    n = len(data)

    mu = np.mean(data)
    sigma = np.std(data, ddof=0)
    if sigma == 0:
        sigma = 1e-8

    k = max(4, int(1 + 3.3 * np.log10(n)))
    while n / k < 5:
        k -= 1
        if k < 4:
            raise ValueError("Слишком малая выборка для выполнения условия np>=5")
    df = k - 3

    prob_edges = np.linspace(0, 1, k + 1)
    edges = mu + sigma * stats.norm.ppf(prob_edges)

    edges[0] = -np.inf
    edges[-1] = np.inf

    observed, _ = np.histogram(data, bins=edges)

    expected = np.full(k, n / k)

    chi2_stat = np.sum((observed - expected) ** 2 / expected)

    critical = stats.chi2.ppf(1 - alpha, df)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    reject = chi2_stat > critical

    return {
        'statistic': chi2_stat,
        'p_value': p_value,
        'critical': critical,
        'df': df,
        'reject': reject,
        'mu': mu,
        'sigma': sigma
    }


def power_simulation(distribution, n, alpha=0.05, reps=100):
    rejections = 0
    for _ in range(reps):
        if distribution == 'uniform':
            sample = np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
        elif distribution == 'laplace':
            sample = np.random.laplace(0, 1, n)
        else:
            raise ValueError
        res = chi2_test_normal(sample, alpha)
        if res['reject']:
            rejections += 1
    return rejections / reps


def main():
    warnings.filterwarnings('ignore')

    np.random.seed(2026)
    n_norm = 100
    data_norm = np.random.normal(0, 1, n_norm)
    print("\n1. Выборка из N(0,1), объём =", n_norm)
    res_norm = chi2_test_normal(data_norm, alpha=0.05)
    print(f"   Оценки: mu = {res_norm['mu']:.4f}, sigma = {res_norm['sigma']:.4f}")
    print(f"   Статистика chi2 = {res_norm['statistic']:.4f}, df = {res_norm['df']}")
    print(f"   Критическое значение (alpha=0.05) = {res_norm['critical']:.4f}")
    print(f"   p-значение = {res_norm['p_value']:.4f}")
    if res_norm['reject']:
        print("   Результат: гипотеза о нормальности ОТВЕРГАЕТСЯ")
    else:
        print("   Результат: гипотеза о нормальности ПРИНИМАЕТСЯ")

    n_alt =20
    a = -np.sqrt(3)
    b = np.sqrt(3)
    data_uniform = np.random.uniform(a, b, n_alt)
    print("\n2. Выборка из равномерного U(-sqrt3,sqrt3), объём =", n_alt)
    res_uniform = chi2_test_normal(data_uniform, alpha=0.05)
    print(f"   Оценки нормального распределения: mu = {res_uniform['mu']:.4f}, sigma = {res_uniform['sigma']:.4f}")
    print(f"   Статистика chi2 = {res_uniform['statistic']:.4f}, df = {res_uniform['df']}")
    print(f"   Критическое значение = {res_uniform['critical']:.4f}")
    print(f"   p-значение = {res_uniform['p_value']:.4f}")
    if res_uniform['reject']:
        print("   Результат: гипотеза о нормальности ОТВЕРГАЕТСЯ")
    else:
        print("   Результат: гипотеза о нормальности ПРИНИМАЕТСЯ")

    data_laplace = np.random.laplace(0, 1, n_alt)
    print("\n3. Выборка из распределения Лапласа L(0,1), объём =", n_alt)
    res_laplace = chi2_test_normal(data_laplace, alpha=0.05)
    print(f"   Оценки нормального распределения: mu = {res_laplace['mu']:.4f}, sigma = {res_laplace['sigma']:.4f}")
    print(f"   Статистика chi2 = {res_laplace['statistic']:.4f}, df = {res_laplace['df']}")
    print(f"   Критическое значение = {res_laplace['critical']:.4f}")
    print(f"   p-значение = {res_laplace['p_value']:.4f}")
    if res_laplace['reject']:
        print("   Результат: гипотеза о нормальности ОТВЕРГАЕТСЯ")
    else:
        print("   Результат: гипотеза о нормальности ПРИНИМАЕТСЯ")

    print("\nМощность критерия (доля отвержений при 100 повторах):")
    power_uniform = power_simulation('uniform', n=20, reps=100)
    power_laplace = power_simulation('laplace', n=20, reps=100)
    print(f"Равномерное: {power_uniform:.2f}, Лапласа: {power_laplace:.2f}")

if __name__ == "__main__":
    main()