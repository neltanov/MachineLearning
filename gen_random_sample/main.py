import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.linalg import lstsq


def generate_random_sample(sample_size, initial_epsilon, eps_distribution, fn):
    xs = np.random.uniform(-1, 1, sample_size)
    if eps_distribution == "uniform":
        epsilons = np.random.uniform(-initial_epsilon, initial_epsilon, sample_size)
    elif eps_distribution == "normal":
        epsilons = np.random.normal(0, initial_epsilon, sample_size)

    if fn.__name__ == "f":
        a, b, c, d = np.random.uniform(-3, 3, 4)
        ys = np.array(list(map(lambda y, epsilon: y + epsilon, map(lambda x: fn(x, a, b, c, d), xs), epsilons)))
        x_actual = np.linspace(-1, 1, 100)
        y_actual = fn(x_actual, a, b, c, d)
    else:
        ys = np.array(list(map(lambda y, epsilon: y + epsilon, map(lambda x: fn(x), xs), epsilons)))
        x_actual = np.linspace(-1, 1, 100)
        y_actual = list(fn(x_actual[i]) for i in range(len(x_actual)))

    return xs, ys, x_actual, y_actual


def function_restoration(sample, M, N):
    A, B = [], []
    for i in range(len(sample[0])):
        A.append(list(sum(sample[0][k] ** (i + j) for k in range(N)) for j in range(M + 1)))
        B.append(sum(sample[0][k] ** i * sample[1][k] for k in range(N)))

    w = lstsq(A, B)[0]
    x_restored = np.linspace(-1, 1, 100)
    f_restored = list(sum(x ** i * w[i] for i in range(M + 1)) for x in x_restored)
    return x_restored, f_restored


def f(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def g(x):
    return x * math.sin(10 * math.pi * x)


def f2(x):
    return 1 / x * math.sin(1 / x)


def main():
    N = 120
    eps_0 = 1e-1
    M = 40

    sample = generate_random_sample(N, eps_0, "normal", g)
    restored_values = function_restoration(sample, M, N)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(sample[2], sample[3], 'b')  # фактическая функция
    plt.scatter(sample[0], sample[1], s=None, c="r", edgecolors="y")  # выборка (фактическая функция с ошибкой)
    plt.plot(restored_values[0], restored_values[1], 'g')  # восстановленная функция
    plt.legend(['Фактическая функция', 'Элементы выборки', 'Восстановленная функция'], loc='best')
    plt.ylim(-3, 3)
    plt.show()


if __name__ == '__main__':
    main()
