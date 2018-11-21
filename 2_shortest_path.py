import numpy as np
from matplotlib import pyplot as plt
import math


def generate_points(end=10, count=1):
    x = np.random.randint(0, end, count)
    y = np.random.randint(0, end, count)
    return np.array(list(zip(x, y)))


def dist(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)


def euclid_distance(a, b):
    return math.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))


def algorithm(k=4, l=10):
    plt.ion()
    # Генерируем вершины
    points = generate_points(1000, l)
    # Находим центр масс
    center = np.mean(points, axis=0)
    # Находим радиус вписанной окружности
    radius = np.nanmax(dist(points, center))
    # Находим дальние точки
    x_boundary, y_boundary = points.max(axis=0)

    plt.xlim([-radius, x_boundary + radius])
    plt.ylim([-radius, y_boundary + radius])

    for point in enumerate(points):
        (index, [x, y]) = point
        plt.scatter(x, y, s=10, c='r')
        plt.text(x + 6, y + 6, index, fontsize=6)

    min_distances = np.zeros(l, dtype=[('x', 'i4'), ('y', 'f4')])
    for i in range(l):
        point = points[i]
        other_points_indexes = list(range(l))
        other_points_indexes.remove(i)
        d = np.array([euclid_distance(point, points[j]) for j in other_points_indexes])
        min_distances[i] = (np.nanargmin(d), np.nanmin(d))
    for fromAxis, distance in np.ndenumerate(min_distances):
        toIndex = distance[0]
        fromIndex = fromAxis[0]
        [fromX, fromY] = points[fromIndex]
        [toX, toY] = points[toIndex]
        plt.plot([fromX, toX], [fromY, toY])

algorithm()
