# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle


def generate_points(end=10, count=1):
    x = np.random.randint(0, end, count)
    y = np.random.randint(0, end, count)
    return np.array(list(zip(x, y)))


def dist(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)


def kmeans(k=4, points=3000):
    plt.ion()
    # Генерируем точки
    points = generate_points(end=200000, count=points)
    # Находим центр масс
    center = np.mean(points, axis=0)
    # Находим радиус вписанной окружности
    radius = np.nanmax(dist(points, center))
    temp_x, temp_y = np.zeros((2, k))
    # Соответсвие точки и её кластера
    clusters = np.zeros(len(points))
    # Цвета кластеров
    cluster_colors = [
        '#845EC2',
        '#D65DB1',
        '#FF6F91',
        '#FF9671',
        '#FFC75F',
        '#F9F871',
        '#2C73D2',
        '#0081CF',
        '#0089BA',
        '#008E9B',
        '#008F7A'
    ]
    shuffle(cluster_colors)
    # Центры кластеров по осям
    cluster_centers = np.array(list(zip(temp_x, temp_y)))
    # Массив, в котором будут хранится промежуточные
    # центры кластеров
    cluster_center_old = np.zeros(cluster_centers.shape)
    # Координаты центра по осям
    x_center, y_center = center

    for i in range(k):
        cluster_centers[i][0] = radius * np.cos(2 * np.pi * (i + 1) / k) + x_center
        cluster_centers[i][1] = radius * np.sin(2 * np.pi * (i + 1) / k) + y_center

    plt.gca().set_xlim([x_center - radius - 1, x_center + radius + 1])
    plt.gca().set_ylim([y_center - radius - 1, y_center + radius + 1])

    print (cluster_centers)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='.', s=None, c='#050505')
    plt.scatter(points[:, 0], points[:, 1], s=3, c='r')

    plt.draw()
    plt.pause(2)

    # Растояние между старым и новым местоположением центроид
    distance = dist(cluster_centers, cluster_center_old, None)
    while distance != 0:
        plt.clf()
        # Присваиваем каждой точке кластер, имеющий кратчайшее расстояние до точки
        clusters = [np.argmin(dist(point, cluster_centers)) for point in points]
        # Сохраняем координаты центроид
        cluster_center_old = np.array(cluster_centers)
        for i in range(k):
            # Точки, лежащие в кластере
            cluster_points = np.array([points[j] for j in range(len(points)) if clusters[j] == i])
            # Новые координаты центроид
            cluster_centers[i] = np.mean(cluster_points, axis=0)
            # Рисуем точки, задавая цвета в соответствии с кластером
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=3, c=cluster_colors[i])
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='.', s=None, c='#050505')
        plt.gca().set_xlim([x_center - radius - 1, x_center + radius + 1])
        plt.gca().set_ylim([y_center - radius - 1, y_center + radius + 1])
        plt.draw()
        plt.pause(1)
        distance = dist(cluster_centers, cluster_center_old, None)


kmeans(k=6)
