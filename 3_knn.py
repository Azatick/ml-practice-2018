import numpy as np
from matplotlib import pyplot as plt
import random
import math


def generate_data(points_count, classes_count):
    data = []
    for class_index in range(classes_count):
        # Choose random center of 2-dimensional gaussian
        center_x, center_y = random.random() * 5.0, random.random() * 5.0  # Choose numberOfClassEl random nodes with RMS=0.5
        for rowNum in range(points_count):
            data.append([[random.gauss(center_x, 0.5), random.gauss(center_y, 0.5)], class_index])
    return data


def generate_test():
    center_x, center_y = random.random() * 5.0, random.random() * 5.0
    return [random.gauss(center_x, 0.5), random.gauss(center_y, 0.5)]


def euclid_distance(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)


def algorithm(points_count=100, classes_count=5, test_points_count=20):
    plt.ion()
    k = int(math.sqrt(points_count))
    # Список рандомных цветов классов
    colors = ["#" + '%06X' % random.randint(0, 0xFFFFFF) for i in range(classes_count)]
    cases = np.zeros(len(colors))
    # Сгенерированные данные
    data = generate_data(points_count, classes_count)
    # Координаты точек
    points = np.array(list(map(lambda v: v[0], data)))
    # Соседи
    neighbours = np.zeros(k)

    # Рисуем сгенерированные точки
    for i in range(len(data)):
        class_index = data[i][1]
        color_index = colors.index(colors[class_index])
        if cases[color_index] == 0:
            label = str(class_index + 1) + ' ' + colors[class_index]
        else:
            label = ''
        [x, y] = points[i]
        plt.scatter(x, y, s=1, c=colors[class_index], label=label)
        cases[color_index] = 1
        plt.draw()
        plt.pause(.000001)

    for i in range(test_points_count):
        test_point = generate_test()
        plt.title('')
        plt.scatter(test_point[0], test_point[1], s=20, c='#000000')
        plt.pause(2)
        distances = euclid_distance(test_point, points)
        class_a = np.zeros(classes_count)
        for i in range(k):
            neighbours[i] = np.nanargmin(distances)
            distances[int(neighbours[i])] = np.nan
            neighbours[i] = data[int(neighbours[i])][1]

        for i in range(k):
            class_a[int(neighbours[i])] += 1
        class_point = np.argmax(class_a)
        plt.title("k = √" + str(points_count) + "  " + str(class_a))
        plt.pause(1)
        print(class_point)
        plt.scatter(test_point[0], test_point[1], s=20, c=colors[class_point])
        plt.pause(3)
    plt.draw()
    plt.title("END")
    plt.pause(10)


algorithm()
