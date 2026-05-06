import numpy as np
from scipy import stats
import warnings

def confidence_interval_mean(data, gamma=0.95):
    n = len(data)
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)
    t_crit = stats.t.ppf((1 + gamma) / 2, df=n-1)
    margin = t_crit * s / np.sqrt(n)
    return x_bar - margin, x_bar + margin

def confidence_interval_std(data, gamma=0.95):
    n = len(data)
    s = np.std(data, ddof=1)
    alpha = 1 - gamma
    chi2_low = stats.chi2.ppf(alpha/2, df=n-1)
    chi2_high = stats.chi2.ppf(1 - alpha/2, df=n-1)
    low = s * np.sqrt((n-1) / chi2_high)
    high = s * np.sqrt((n-1) / chi2_low)
    return low, high

def f_test(data1, data2, alpha=0.05):
    n1, n2 = len(data1), len(data2)
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)

    if var1 >= var2:
        F_stat = var1 / var2
        df1, df2 = n1-1, n2-1
    else:
        F_stat = var2 / var1
        df1, df2 = n2-1, n1-1

    p_value = 1 - stats.f.cdf(F_stat, df1, df2)
    F_crit_upper = stats.f.ppf(1 - alpha/2, df1, df2)
    F_crit_lower = stats.f.ppf(alpha/2, df1, df2)
    reject = (F_stat < F_crit_lower) or (F_stat > F_crit_upper)

    return {
        'F_stat': F_stat,
        'df1': df1,
        'df2': df2,
        'F_crit_lower': F_crit_lower,
        'F_crit_upper': F_crit_upper,
        'p_value': p_value,
        'reject': reject
    }

def main():
    warnings.filterwarnings('ignore')
    np.random.seed(2026)

    n1, n2 = 20, 100
    gamma = 0.95
    alpha_f = 0.05

    sample1 = np.random.normal(0, 1, n1)
    sample2 = np.random.normal(0, 1, n2)

    ci_mean1 = confidence_interval_mean(sample1, gamma)
    print(f"Выборка 1 (n={n1}):")
    print(f"  Среднее: {np.mean(sample1):.4f}")
    print(f"  {gamma*100:.0f}%-доверительный интервал для mu: "
          f"[{ci_mean1[0]:.4f}, {ci_mean1[1]:.4f}]")

    ci_std1 = confidence_interval_std(sample1, gamma)
    print(f"  {gamma*100:.0f}%-доверительный интервал для sigma: "
          f"[{ci_std1[0]:.4f}, {ci_std1[1]:.4f}]")

    ci_mean2 = confidence_interval_mean(sample2, gamma)
    print(f"\nВыборка 2 (n={n2}):")
    print(f"  Среднее: {np.mean(sample2):.4f}")
    print(f"  {gamma*100:.0f}%-доверительный интервал для mu: "
          f"[{ci_mean2[0]:.4f}, {ci_mean2[1]:.4f}]")

    ci_std2 = confidence_interval_std(sample2, gamma)
    print(f"  {gamma*100:.0f}%-доверительный интервал для sigma: "
          f"[{ci_std2[0]:.4f}, {ci_std2[1]:.4f}]")

    print(f"\nКритерий Фишера (H0: sigma1^2 = sigma2^2, alpha={alpha_f}):")
    res_f = f_test(sample1, sample2, alpha_f)
    print(f"  Выборочные дисперсии: s1^2 = {np.var(sample1, ddof=1):.4f}, "
          f"s2^2 = {np.var(sample2, ddof=1):.4f}")
    print(f"  Статистика F = {res_f['F_stat']:.4f}")
    print(f"  Числа степеней свободы: df1={res_f['df1']}, df2={res_f['df2']}")
    print(f"  Критические границы: "
          f"F_нижн={res_f['F_crit_lower']:.4f}, F_верх={res_f['F_crit_upper']:.4f}")
    print(f"  p-value = {res_f['p_value']:.4f}")
    if res_f['reject']:
        print("  Результат: H0 отвергается.")
    else:
        print("  Результат: H0 принимается.")

if __name__ == "__main__":
    main()