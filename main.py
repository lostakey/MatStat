import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (15, 10)


# Функция для оптимального выбора количества бинов (формула Фридмана-Диакониса)
def optimal_bins(data):
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data) ** (-1 / 3)
    if bin_width > 0:
        bins = int((data.max() - data.min()) / bin_width)
        return max(1, min(50, bins))  # Ограничиваем количество бинов
    return 30  # Значение по умолчанию


# Функция для построения гистограммы для одного распределения
def plot_distribution(data, dist_name, theoretical_pdf, x_range, n, ax):
    bins = optimal_bins(data)
    ax.hist(data, bins=bins, density=True, alpha=0.7,
            color='skyblue', edgecolor='black', linewidth=1,
            label=f'Гистограмма (n={n}, бинов={bins})')

    x = np.linspace(x_range[0], x_range[1], 1000)
    y = theoretical_pdf(x)
    ax.plot(x, y, 'r-', linewidth=2.5, label='Теоретическая плотность')

    ax.set_title(f'{dist_name}\nОбъём выборки: {n}', fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('Плотность вероятности', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    distributions = [
        {
            'name': 'Нормальное N(0,1)',
            'generator': lambda n: np.random.normal(0, 1, n),
            'pdf': lambda x: stats.norm.pdf(x, 0, 1),
            'x_range': (-4, 4)
        },
        {
            'name': 'Коши C(0,1)',
            'generator': lambda n: np.random.standard_cauchy(n),
            'pdf': lambda x: stats.cauchy.pdf(x, 0, 1),
            'x_range': (-10, 10)
        },
        {
            'name': 'Лапласа L(0, 1/√2)',
            'generator': lambda n: np.random.laplace(0, 1 / np.sqrt(2), n),
            'pdf': lambda x: stats.laplace.pdf(x, 0, 1 / np.sqrt(2)),
            'x_range': (-5, 5)
        },
        {
            'name': 'Пуассона P(5)',
            'generator': lambda n: np.random.poisson(5, n),
            'pdf': lambda x: stats.poisson.pmf(np.round(x), 5),
            'x_range': (0, 12),
            'discrete': True
        },
        {
            'name': 'Равномерное U(-√3, √3)',
            'generator': lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n),
            'pdf': lambda x: stats.uniform.pdf(x, -np.sqrt(3), 2 * np.sqrt(3)),
            'x_range': (-2, 2)
        }
    ]

    sample_sizes = [10, 100, 1000]

    fig, axes = plt.subplots(len(distributions), len(sample_sizes),
                             figsize=(18, 24))

    for i, dist in enumerate(distributions):
        for j, n in enumerate(sample_sizes):
            data = dist['generator'](n)

            if dist['name'].startswith('Коши'):
                data = data[(data > -10) & (data < 10)]

            ax = axes[i, j]

            if dist.get('discrete', False):
                min_val = max(0, int(data.min()) - 1)
                max_val = int(data.max()) + 2

                k_values = np.arange(min_val, max_val + 1)

                frequencies = []
                for k in k_values:
                    count = np.sum(data == k)
                    frequencies.append(count / n)

                n_bins = len(k_values)

                ax.bar(k_values, frequencies, width=0.9, alpha=0.7,
                       color='skyblue', edgecolor='black', linewidth=1,
                       label=f'Гистограмма (n={n}, бинов={n_bins})')

                y = stats.poisson.pmf(k_values, 5)
                ax.plot(k_values, y, 'ro-', linewidth=2, markersize=4,
                        label='Теоретическая PMF')

                ax.set_title(f'{dist["name"]}\nОбъём выборки: {n}', fontsize=12, fontweight='bold')
                ax.set_xlabel('k', fontsize=10)
                ax.set_ylabel('Вероятность', fontsize=10)
                ax.set_xlim(min_val - 0.5, max_val + 0.5)
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                plot_distribution(data, dist['name'], dist['pdf'],
                                  dist['x_range'], n, ax)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    plt.savefig('lab1_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    main()