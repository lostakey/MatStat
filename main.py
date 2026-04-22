import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(2026)

x = np.arange(-1.8, 2.01, 0.2)  # включая 2.0
n = len(x)

a_true = 2.0
b_true = 2.0

eps = np.random.normal(0, 1, n)
y_clean = a_true + b_true * x + eps

y_contaminated = y_clean.copy()
y_contaminated[0] += 10
y_contaminated[-1] -= 10

def mnk(x, y):
    b_hat = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
    a_hat = np.mean(y) - b_hat * np.mean(x)
    return a_hat, b_hat


def mnm_loss(params, x, y):
    a, b = params
    return np.sum(np.abs(y - (a + b * x)))


def mnm(x, y):
    init_a, init_b = mnk(x, y)
    res = minimize(mnm_loss, [init_a, init_b], args=(x, y), method='Nelder-Mead')
    if res.success:
        return res.x[0], res.x[1]
    else:
        raise RuntimeError("Оптимизация МНМ не сошлась")

a_mnk_clean, b_mnk_clean = mnk(x, y_clean)
a_mnm_clean, b_mnm_clean = mnm(x, y_clean)

a_mnk_cont, b_mnk_cont = mnk(x, y_contaminated)
a_mnm_cont, b_mnm_cont = mnm(x, y_contaminated)

def rel_error(true, est):
    abs_err = np.abs(est - true)
    rel_err = abs_err / np.abs(true) * 100 if true != 0 else np.inf
    return abs_err, rel_err

results = {
    'МНК (чистые)': {
        'a': a_mnk_clean, 'Δa': rel_error(a_true, a_mnk_clean)[0], 'δa%': rel_error(a_true, a_mnk_clean)[1],
        'b': b_mnk_clean, 'Δb': rel_error(b_true, b_mnk_clean)[0], 'δb%': rel_error(b_true, b_mnk_clean)[1]
    },
    'МНМ (чистые)': {
        'a': a_mnm_clean, 'Δa': rel_error(a_true, a_mnm_clean)[0], 'δa%': rel_error(a_true, a_mnm_clean)[1],
        'b': b_mnm_clean, 'Δb': rel_error(b_true, b_mnm_clean)[0], 'δb%': rel_error(b_true, b_mnm_clean)[1]
    },
    'МНК (с выбросами)': {
        'a': a_mnk_cont, 'Δa': rel_error(a_true, a_mnk_cont)[0], 'δa%': rel_error(a_true, a_mnk_cont)[1],
        'b': b_mnk_cont, 'Δb': rel_error(b_true, b_mnk_cont)[0], 'δb%': rel_error(b_true, b_mnk_cont)[1]
    },
    'МНМ (с выбросами)': {
        'a': a_mnm_cont, 'Δa': rel_error(a_true, a_mnm_cont)[0], 'δa%': rel_error(a_true, a_mnm_cont)[1],
        'b': b_mnm_cont, 'Δb': rel_error(b_true, b_mnm_cont)[0], 'δb%': rel_error(b_true, b_mnm_cont)[1]
    }
}

print("\n" + "=" * 70)
print("Результаты оценивания параметров линейной регрессии")
print("Истинные значения: a = 2.0, b = 2.0")
print("=" * 70)
print(f"{'Метод':<18} {'a':<8} {'Δa':<8} {'δa,%':<8} {'b':<8} {'Δb':<8} {'δb,%':<8}")
print("-" * 70)
for method, vals in results.items():
    print(
        f"{method:<18} {vals['a']:<8.4f} {vals['Δa']:<8.4f} {vals['δa%']:<8.2f} {vals['b']:<8.4f} {vals['Δb']:<8.4f} {vals['δb%']:<8.2f}")
print("=" * 70)

def plot_regression(x, y_clean, y_cont, a_mnk, b_mnk, a_mnm, b_mnm, title, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_clean, label='Данные без выбросов', alpha=0.7, edgecolors='k')
    plt.scatter(x, y_cont, label='Данные с выбросами', alpha=0.7, marker='s', edgecolors='k')

    x_line = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_line, a_true + b_true * x_line, 'k--', label='Истинная прямая', linewidth=2)
    plt.plot(x_line, a_mnk + b_mnk * x_line, 'r-', label='МНК', linewidth=2)
    plt.plot(x_line, a_mnm + b_mnm * x_line, 'b-', label='МНМ', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

plot_regression(x, y_clean, y_clean,
                a_mnk_clean, b_mnk_clean,
                a_mnm_clean, b_mnm_clean,
                'Регрессия на чистых данных', 'lab6_clean.png')

plot_regression(x, y_clean, y_contaminated,
                a_mnk_cont, b_mnk_cont,
                a_mnm_cont, b_mnm_cont,
                'Регрессия при наличии выбросов', 'lab6_contaminated.png')
