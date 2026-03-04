import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

np.random.seed(42)

distributions = {
    'Нормальное': lambda n: np.random.normal(0, 1, n),
    'Коши': lambda n: np.random.standard_cauchy(n),
    'Лапласа': lambda n: np.random.laplace(0, 1 / np.sqrt(2), n),  # параметр масштаба для дисперсии 1/2
    'Пуассона': lambda n: np.random.poisson(5, n),  # Изменено с 10 на 5
    'Равномерное': lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
}


def calculate_statistics(sample):
    """
    Вычисление всех характеристик для одной выборки
    """
    n = len(sample)
    sorted_sample = np.sort(sample)

    # 1. Выборочное среднее
    sample_mean = np.mean(sample)

    # 2. Медиана
    sample_median = np.median(sample)

    # 3. Полусумма экстремальных элементов
    zR = (sorted_sample[0] + sorted_sample[-1]) / 2

    # 4. Полусумма квартилей
    q1 = np.percentile(sample, 25)
    q3 = np.percentile(sample, 75)
    zQ = (q1 + q3) / 2

    # 5. Усечённое среднее (отбрасываем 10% наименьших и наибольших)
    trim_percent = 0.1
    lower_bound = int(n * trim_percent)
    upper_bound = int(n * (1 - trim_percent))
    if upper_bound > lower_bound:
        trimmed_sample = sorted_sample[lower_bound:upper_bound]
        ztr = np.mean(trimmed_sample)
    else:
        ztr = np.nan

    return {
        'mean': sample_mean,
        'median': sample_median,
        'zR': zR,
        'zQ': zQ,
        'ztr': ztr
    }


def analyze_distribution(dist_name, sample_generator, n, num_iterations=1000):
    """
    Анализ распределения: генерация выборок и вычисление статистик
    """
    results = {key: [] for key in ['mean', 'median', 'zR', 'zQ', 'ztr']}

    for _ in range(num_iterations):
        sample = sample_generator(n)
        stats_dict = calculate_statistics(sample)

        for key in results:
            if not np.isnan(stats_dict[key]):
                results[key].append(stats_dict[key])

    analysis_results = {}
    for key, values in results.items():
        values_array = np.array(values)
        if len(values_array) > 0:
            mean_val = np.mean(values_array)
            std_val = np.std(values_array, ddof=1)
            analysis_results[key] = {
                'E': mean_val,
                'D': np.var(values_array, ddof=1),
                'std': std_val,
                'E ± √D': f"{mean_val:.3f} ± {std_val:.3f}"
            }
        else:
            analysis_results[key] = {
                'E': np.nan,
                'D': np.nan,
                'std': np.nan,
                'E ± √D': "NaN ± NaN"
            }

    return analysis_results


def main():
    """
    Основная функция для выполнения лабораторной работы №2
    """
    sample_sizes = [10, 100, 1000]
    num_iterations = 1000

    print("=" * 100)
    print("ЛАБОРАТОРНАЯ РАБОТА №2: Характеристики положения и рассеяния")
    print("=" * 100)

    all_results = []

    for dist_name, generator in distributions.items():
        print(f"\nАнализ распределения: {dist_name}")
        print("-" * 50)

        for n in sample_sizes:
            print(f"  Объем выборки n = {n}")

            results = analyze_distribution(dist_name, generator, n, num_iterations)

            for stat_name, stat_values in results.items():
                all_results.append([
                    dist_name,
                    n,
                    stat_name,
                    stat_values['E'],
                    stat_values['std'],
                    stat_values['E ± √D']
                ])

                if not np.isnan(stat_values['E']):
                    print(f"    {stat_name:8s}: E = {stat_values['E']:8.4f}, √D = {stat_values['std']:8.4f}, "
                          f"представление: {stat_values['E ± √D']}")
                else:
                    print(f"    {stat_name:8s}: не удалось вычислить")

    print("\n" + "=" * 100)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 100)

    df_results = pd.DataFrame(all_results,
                              columns=['Распределение', 'n', 'Статистика', 'E', '√D', 'E ± √D'])

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(df_results.to_string(index=False, float_format='%.4f'))

    print("\n" + "=" * 100)
    print("АНАЛИЗ СХОДИМОСТИ")
    print("=" * 100)

    theoretical_values = {
        'Нормальное': {'mean': 0, 'median': 0},
        'Коши': {'mean': 'не определено', 'median': 0},
        'Лапласа': {'mean': 0, 'median': 0},
        'Пуассона': {'mean': 5, 'median': 5},
        'Равномерное': {'mean': 0, 'median': 0}
    }

    for dist_name in distributions.keys():
        print(f"\n{dist_name} распределение:")
        print(f"  Теоретические значения: среднее = {theoretical_values[dist_name]['mean']}, "
              f"медиана = {theoretical_values[dist_name]['median']}")

        dist_data = df_results[df_results['Распределение'] == dist_name]

        for stat in ['mean', 'median']:
            stat_data = dist_data[dist_data['Статистика'] == stat]
            if not stat_data.empty:
                print(f"  {stat}:")
                for _, row in stat_data.iterrows():
                    print(f"    n={int(row['n'])}: E={row['E']:.4f}")

    print("\n" + "=" * 100)
    print("АНАЛИЗ РОБАСТНОСТИ (устойчивости к выбросам)")
    print("=" * 100)

    print("\nСравнение стандартных отклонений для n=1000 (меньшее √D = более устойчивая оценка):")
    print("-" * 80)

    comparison_data = []
    stats_list = ['mean', 'median', 'zR', 'zQ', 'ztr']

    for stat in stats_list:
        normal_data = df_results[(df_results['Распределение'] == 'Нормальное') &
                                 (df_results['n'] == 1000) &
                                 (df_results['Статистика'] == stat)]

        cauchy_data = df_results[(df_results['Распределение'] == 'Коши') &
                                 (df_results['n'] == 1000) &
                                 (df_results['Статистика'] == stat)]

        if not normal_data.empty and not cauchy_data.empty:
            normal_std = normal_data['√D'].values[0]
            cauchy_std = cauchy_data['√D'].values[0]

            if not np.isnan(normal_std) and not np.isnan(cauchy_std) and np.isfinite(cauchy_std):
                ratio = cauchy_std / normal_std if normal_std != 0 else float('inf')
                comparison_data.append([
                    stat,
                    f"{normal_std:.4f}",
                    f"{cauchy_std:.4f}",
                    f"{ratio:.2f}"
                ])
            else:
                comparison_data.append([
                    stat,
                    f"{normal_std:.4f}" if not np.isnan(normal_std) else "NaN",
                    f"{cauchy_std:.4f}" if not np.isnan(cauchy_std) else "NaN",
                    "∞" if np.isinf(cauchy_std) else "NaN"
                ])

    headers = ["Статистика", "√D (Нормальное)", "√D (Коши)", "Отношение (Коши/Нормальное)"]
    print(tabulate(comparison_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()