import numpy as np
from matplotlib import pyplot as plt
import math
import sys


def generate_points(end=10, count=1):
    x = np.random.randint(0, end, count)
    y = np.random.randint(0, end, count)
    return np.array(list(zip(x, y)))


def dist(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)


def euclid_distance(a, b):
    return math.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))


def get_points_min(points, insulated, uninsulated):
    min_distances = np.zeros(len(points), dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'f4')])
    for point_index in insulated:
        point = points[point_index]
        d = np.array([euclid_distance(point, points[j]) for j in uninsulated])
        np.put(d, point_index, sys.maxsize)
        min_distances.put(point_index, (point_index, np.nanargmin(d), np.nanmin(d)))
    min_distances_list = min_distances.tolist()
    min_dist = min(min_distances_list, key=lambda t: t[2])
    from_index = min_dist[0]
    to_index = min_dist[1]
    [from_x, from_y] = points[from_index]
    [to_x, to_y] = points[to_index]
    plt.plot([from_x, to_x], [from_y, to_y])
    insulated.remove(from_index)
    insulated.remove(to_index)
    uninsulated.append(from_index)
    uninsulated.append(to_index)
    return (insulated, uninsulated)


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
        plt.draw()
        plt.pause(1)

    min_distances = np.zeros(l, dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'f4')])
    # Ищем ближайшие друг к другу точки
    for i in range(l):
        point = points[i]
        other_points_indexes = list(range(l))
        d = np.array([euclid_distance(point, points[j]) for j in other_points_indexes])
        np.put(d, i, sys.maxsize)
        min_distances.put(i, (i, np.nanargmin(d), np.nanmin(d)))

    # Список изолированных точек
    insulated_points = list(range(l))
    # Список неизолированных точек
    uninsulated_points = []
    # Список ребер
    edges = []
    # Список прямых из pyplot
    plots = []
    # Самая первая пара точек
    min_distances_list = min_distances.tolist()
    min_dist = min(min_distances_list, key=lambda t: t[2])
    from_index = min_dist[0]
    to_index = min_dist[1]
    [from_x, from_y] = points[from_index]
    [to_x, to_y] = points[to_index]

    plots.append(plt.plot([from_x, to_x], [from_y, to_y]))
    plt.draw()
    plt.pause(1)
    edges.append((from_index, to_index, min_dist[2]))
    insulated_points.remove(from_index)
    insulated_points.remove(to_index)
    uninsulated_points.append(from_index)
    uninsulated_points.append(to_index)

    print(insulated_points)
    print(uninsulated_points)

    while len(insulated_points) > 0:
        min_distances = np.zeros(len(insulated_points), dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'f4')])
        for index, point_index in enumerate(insulated_points):
            point = points[point_index]
            d = np.array([euclid_distance(point, points[j]) for j in uninsulated_points])
            min_distances.put(index, (point_index, uninsulated_points[np.nanargmin(d)], np.nanmin(d)))
        min_distances_list = min_distances.tolist()
        min_dist = min(min_distances_list, key=lambda t: t[2])
        from_index = min_dist[0]
        to_index = min_dist[1]
        [from_x, from_y] = points[from_index]
        [to_x, to_y] = points[to_index]

        plots.append(plt.plot([from_x, to_x], [from_y, to_y]))
        plt.draw()
        plt.pause(1)
        edges.append((from_index, to_index, min_dist[2]))
        insulated_points.remove(from_index)
        uninsulated_points.append(from_index)
    print(edges)

    # Удаляем k-1 длинных ребер
    for i in list(range(k-1)):
        max_edge = max(edges, key=lambda t: t[2])
        max_index = edges.index(max_edge)
        edges.remove(max_edge)
        print(max_index)
        [from_x, from_y] = points[max_edge[0]]
        [to_x, to_y] = points[max_edge[1]]
        plt.plot([from_x, to_x], [from_y, to_y], linestyle='None',)
        plots[max_index][0].remove()
        plt.draw()
        plt.pause(1)
    plt.draw()
    plt.pause(60)

algorithm()
