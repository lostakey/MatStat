import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform

np.random.seed(2026)

distributions = {
    'Нормальное': lambda n: norm.rvs(loc=0, scale=1, size=n),
    'Коши': lambda n: cauchy.rvs(loc=0, scale=1, size=n),
    'Лапласа': lambda n: laplace.rvs(loc=0, scale=1 / 2, size=n),
    'Пуассона': lambda n: poisson.rvs(mu=5, size=n),
    'Равномерное': lambda n: uniform.rvs(loc=-np.sqrt(3), scale=2 * np.sqrt(3), size=n)
}

n_values = [20, 100]

n_sim = 1000

def outlier_fraction(sample):
    q1 = np.percentile(sample, 25)
    q3 = np.percentile(sample, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = sample[(sample < lower_bound) | (sample > upper_bound)]
    return len(outliers) / len(sample)

avg_outliers = {name: {} for name in distributions}

fig, axes = plt.subplots(len(n_values), len(distributions), figsize=(15, 6))

for row, n in enumerate(n_values):
    for col, (name, gen) in enumerate(distributions.items()):
        sample = gen(n)

        axes[row, col].boxplot(sample)
        axes[row, col].set_title(f'{name}, n={n}')
        axes[row, col].set_ylabel('Значения')
        axes[row, col].grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('boxplots.png', dpi=300, bbox_inches='tight', facecolor='white')

plt.tight_layout()
plt.show()

print("\nСредняя доля выбросов (по 1000 выборок):")
print("-" * 60)
print(f"{'Распределение':<15} {'n=20':>10} {'n=100':>10}")
print("-" * 60)

for name, gen in distributions.items():
    fracs_20 = []
    fracs_100 = []
    for _ in range(n_sim):
        fracs_20.append(outlier_fraction(gen(20)))
        fracs_100.append(outlier_fraction(gen(100)))
    avg_outliers[name][20] = np.mean(fracs_20)
    avg_outliers[name][100] = np.mean(fracs_100)
    print(f"{name:<15} {avg_outliers[name][20]:>10.4f} {avg_outliers[name][100]:>10.4f}")

print("-" * 60)