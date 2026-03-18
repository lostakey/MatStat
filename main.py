import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform
from scipy.stats import gaussian_kde

np.random.seed(2026)

def truncated_cauchy_rvs(loc=0, scale=1, size=1, trunc_range=50):
    samples = np.zeros(size)
    for i in range(size):
        while True:
            x = cauchy.rvs(loc=loc, scale=scale, size=1)[0]
            if abs(x) <= trunc_range:
                samples[i] = x
                break
    return samples

# Параметры распределений
distributions = {
    'Нормальное': {
        'gen': lambda n: norm.rvs(loc=0, scale=1, size=n),
        'pdf': lambda x: norm.pdf(x, loc=0, scale=1),
        'cdf': lambda x: norm.cdf(x, loc=0, scale=1),
        'x_range': [-4, 4],
        'theor_range': [-4, 4]
    },
    'Коши': {
        'gen': lambda n: truncated_cauchy_rvs(loc=0, scale=1, size=n, trunc_range=100),
        'pdf': lambda x: cauchy.pdf(x, loc=0, scale=1),
        'cdf': lambda x: cauchy.cdf(x, loc=0, scale=1),
        'x_range': [-4, 4],
        'theor_range': [-4, 4]
    },
    'Лапласа': {
        'gen': lambda n: laplace.rvs(loc=0, scale=1 / 2, size=n),
        'pdf': lambda x: laplace.pdf(x, loc=0, scale=1 / 2),
        'cdf': lambda x: laplace.cdf(x, loc=0, scale=1 / 2),
        'x_range': [-4, 4],
        'theor_range': [-4, 4]
    },
    'Пуассона': {
        'gen': lambda n: poisson.rvs(mu=5, size=n),
        'pdf': lambda x: poisson.pmf(x, mu=5),
        'cdf': lambda x: poisson.cdf(x, mu=5),
        'x_range': [6, 14],
        'theor_range': [0, 15]
    },
    'Равномерное': {
        'gen': lambda n: uniform.rvs(loc=-np.sqrt(3), scale=2 * np.sqrt(3), size=n),
        'pdf': lambda x: uniform.pdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
        'cdf': lambda x: uniform.cdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
        'x_range': [-4, 4],
        'theor_range': [-4, 4]
    }
}

# Объёмы выборок
n_values = [20, 60, 100]

for dist_name, dist_params in distributions.items():
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle(f'Распределение: {dist_name}', fontsize=16, fontweight='bold')

    if dist_name == 'Пуассона':
        x_plot = np.arange(0, 16, 1)
        x_continuous = np.linspace(0, 15, 1000)
    else:
        x_plot = np.linspace(dist_params['x_range'][0], dist_params['x_range'][1], 1000)
        x_continuous = x_plot

    for i, n in enumerate(n_values):
        sample = dist_params['gen'](n)

        ax_ecdf = axes[i, 0]

        sorted_sample = np.sort(sample)
        y_ecdf = np.arange(1, n + 1) / n

        ax_ecdf.step(sorted_sample, y_ecdf, where='post',
                     label=f'Эмпирическая (n={n})', linewidth=2, color='blue')

        if dist_name == 'Пуассона':
            x_cdf = np.arange(0, 16, 1)
            y_cdf = dist_params['cdf'](x_cdf)
            ax_ecdf.step(x_cdf, y_cdf, where='post',
                         label='Теоретическая', linewidth=2, color='red', linestyle='--')
        else:
            y_cdf = dist_params['cdf'](x_continuous)
            ax_ecdf.plot(x_continuous, y_cdf,
                         label='Теоретическая', linewidth=2, color='red', linestyle='--')

        ax_ecdf.set_xlabel('x')
        ax_ecdf.set_ylabel('F(x)')
        ax_ecdf.set_title(f'Функция распределения, n={n}')
        ax_ecdf.grid(True, alpha=0.3)
        ax_ecdf.legend(loc='lower right')
        ax_ecdf.set_xlim(dist_params['x_range'])
        ax_ecdf.set_ylim(0, 1.05)

        ax_kde = axes[i, 1]

        ax_kde.hist(sample, bins='auto', density=True, alpha=0.3,
                    color='gray', label='Гистограмма')

        try:
            kde = gaussian_kde(sample)
            if dist_name == 'Пуассона':
                y_kde = kde.evaluate(x_continuous)
            else:
                y_kde = kde.evaluate(x_continuous)
            ax_kde.plot(x_continuous, y_kde,
                        label='Ядерная оценка', linewidth=2, color='green')
        except:
            ax_kde.text(0.5, 0.5, 'Ошибка при построении KDE',
                        transform=ax_kde.transAxes, ha='center')

        if dist_name == 'Пуассона':
            x_pmf = np.arange(0, 16, 1)
            y_pmf = dist_params['pdf'](x_pmf)
            ax_kde.stem(x_pmf, y_pmf, linefmt='r-', markerfmt='ro',
                        basefmt='k-', label='Теоретическая')
        else:
            y_pdf = dist_params['pdf'](x_continuous)
            ax_kde.plot(x_continuous, y_pdf,
                        label='Теоретическая', linewidth=2, color='red', linestyle='--')

        ax_kde.set_xlabel('x')
        ax_kde.set_ylabel('Плотность вероятности')
        ax_kde.set_title(f'Плотность распределения, n={n}')
        ax_kde.grid(True, alpha=0.3)
        ax_kde.legend(loc='upper right')
        ax_kde.set_xlim(dist_params['x_range'])

    plt.tight_layout()
    plt.savefig(f'lab4_{dist_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

print("Все графики сохранены")

print("\n" + "=" * 60)
print("АНАЛИЗ СХОДИМОСТИ ЭМПИРИЧЕСКОЙ ФУНКЦИИ РАСПРЕДЕЛЕНИЯ")
print("=" * 60)

n_values_detailed = [10, 20, 50, 100, 200, 500, 1000]
max_deviations = []

for n in n_values_detailed:
    deviations = []
    for _ in range(100):
        sample = norm.rvs(loc=0, scale=1, size=n)
        sorted_sample = np.sort(sample)
        y_ecdf = np.arange(1, n + 1) / n
        y_cdf = norm.cdf(sorted_sample)
        deviations.append(np.max(np.abs(y_ecdf - y_cdf)))
    max_deviations.append(np.mean(deviations))
