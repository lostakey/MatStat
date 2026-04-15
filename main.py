import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from matplotlib.patches import Ellipse
import os

# --------------------- Функции ---------------------
def quadrant_correlation(x, y):
    med_x = np.median(x)
    med_y = np.median(y)
    x_centered = x - med_x
    y_centered = y - med_y

    q1 = (x_centered > 0) & (y_centered > 0)
    q2 = (x_centered < 0) & (y_centered > 0)
    q3 = (x_centered < 0) & (y_centered < 0)
    q4 = (x_centered > 0) & (y_centered < 0)

    n1, n2, n3, n4 = np.sum(q1), np.sum(q2), np.sum(q3), np.sum(q4)
    n = len(x)
    return ((n1 + n3) - (n2 + n4)) / n

def compute_correlations(x, y):
    r_pearson, _ = pearsonr(x, y)
    r_spearman, _ = spearmanr(x, y)
    r_quadrant = quadrant_correlation(x, y)
    return r_pearson, r_spearman, r_quadrant

def generate_bivariate_normal(n, rho, mean=(0,0), var=(1,1)):
    cov = [[var[0], rho * np.sqrt(var[0]*var[1])],
           [rho * np.sqrt(var[0]*var[1]), var[1]]]
    return np.random.multivariate_normal(mean, cov, size=n)

def generate_mixture(n):
    comp1 = generate_bivariate_normal(n, rho=0.9)
    comp2 = generate_bivariate_normal(n, rho=-0.9, var=(10,10))
    mask = np.random.rand(n) < 0.9
    data = np.where(mask[:, np.newaxis], comp1, comp2)
    return data

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=width, height=height,
                      angle=angle, facecolor=facecolor, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

# --------------------- Основной блок ---------------------
n_values = [20, 60, 100]
rho_values = [0, 0.5, 0.9]
n_simulations = 1000

np.random.seed(42)

results = {}

print("Результаты моделирования (1000 повторений):")
print("n\tρ\tКоэффициент\tСреднее\tДисперсия")
print("-" * 60)

# Чистое нормальное
for n in n_values:
    for rho in rho_values:
        pearson_vals, spearman_vals, quadrant_vals = [], [], []
        for _ in range(n_simulations):
            data = generate_bivariate_normal(n, rho)
            x, y = data[:, 0], data[:, 1]
            p, s, q = compute_correlations(x, y)
            pearson_vals.append(p)
            spearman_vals.append(s)
            quadrant_vals.append(q)

        key = (n, f'rho={rho}')
        results[key] = (pearson_vals, spearman_vals, quadrant_vals)

        print(f"{n}\t{rho}\tПирсон\t\t{np.mean(pearson_vals):.4f}\t{np.var(pearson_vals):.6f}")
        print(f"{n}\t{rho}\tСпирмен\t\t{np.mean(spearman_vals):.4f}\t{np.var(spearman_vals):.6f}")
        print(f"{n}\t{rho}\tКвадрантный\t{np.mean(quadrant_vals):.4f}\t{np.var(quadrant_vals):.6f}")
        print()

# Смесь
for n in n_values:
    pearson_vals, spearman_vals, quadrant_vals = [], [], []
    for _ in range(n_simulations):
        data = generate_mixture(n)
        x, y = data[:, 0], data[:, 1]
        p, s, q = compute_correlations(x, y)
        pearson_vals.append(p)
        spearman_vals.append(s)
        quadrant_vals.append(q)

    key = (n, 'mixture')
    results[key] = (pearson_vals, spearman_vals, quadrant_vals)

    print(f"{n}\tmixture\tПирсон\t\t{np.mean(pearson_vals):.4f}\t{np.var(pearson_vals):.6f}")
    print(f"{n}\tmixture\tСпирмен\t\t{np.mean(spearman_vals):.4f}\t{np.var(spearman_vals):.6f}")
    print(f"{n}\tmixture\tКвадрантный\t{np.mean(quadrant_vals):.4f}\t{np.var(quadrant_vals):.6f}")
    print()

# --------------------- Построение и сохранение графиков ---------------------
plot_n = [20, 60, 100]
plot_rho = [0, 0.5, 0.9]

# Чистые нормальные
fig, axes = plt.subplots(len(plot_n), len(plot_rho), figsize=(12, 10))
if len(plot_n) == 1:
    axes = [axes]
if len(plot_rho) == 1:
    axes = [[ax] for ax in axes]

for i, n in enumerate(plot_n):
    for j, rho in enumerate(plot_rho):
        ax = axes[i][j]
        data = generate_bivariate_normal(n, rho)
        x, y = data[:, 0], data[:, 1]
        ax.scatter(x, y, alpha=0.7, s=20, edgecolors='k', linewidth=0.5)
        confidence_ellipse(x, y, ax, n_std=2, edgecolor='red', linewidth=2)
        ax.set_title(f'n={n}, ρ={rho}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_aspect('equal', adjustable='datalim')

plt.tight_layout()
plt.suptitle('Диаграммы рассеяния и эллипсы (чистое нормальное распределение)', y=1.02)
plt.savefig("normal_combined.png", dpi=150, bbox_inches='tight')
plt.close()

# Смесь
fig, axes = plt.subplots(1, len(plot_n), figsize=(15, 5))
if len(plot_n) == 1:
    axes = [axes]

for i, n in enumerate(plot_n):
    ax = axes[i]
    data = generate_mixture(n)
    x, y = data[:, 0], data[:, 1]
    ax.scatter(x, y, alpha=0.7, s=20, edgecolors='k', linewidth=0.5)
    confidence_ellipse(x, y, ax, n_std=2, edgecolor='red', linewidth=2)
    ax.set_title(f'Смесь, n={n}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_aspect('equal', adjustable='datalim')

plt.tight_layout()
plt.suptitle('Диаграммы рассеяния и эллипсы (смесь распределений)', y=1.02)
plt.savefig("mixture_combined.png", dpi=150, bbox_inches='tight')
plt.close()

print("Графики сохранены в папку 'lab5_plots'")