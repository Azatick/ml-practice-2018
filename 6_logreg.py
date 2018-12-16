import random
import math

import numpy as np
from matplotlib import pyplot as plt


random.seed(1000)


def generate_data(points_count):
    data = []
    classes = [0, 1]
    centers = [(random.random() * 5.0, random.random() * 5.0)
               for class_num in classes]
    for i in range(points_count):
        class_index = random.choice([0, 1])
        center_x, center_y = centers[class_index]
        data.append([[random.gauss(center_x, 0.5), random.gauss(
            center_y, 0.5)], classes[class_index]])
    return data


def logistic_regression(x, y, alpha=0.05, lambda_=0):
    m, n = np.shape(x)
    w = np.ones(n)
    x_trans = x.transpose()
    old_cost = 0.0
    value = True
    while value:
        hypothesis = np.dot(x, w)
        logistic = hypothesis / (np.exp(-hypothesis) + 1)
        reg = (lambda_ / 2 * m) * np.sum(np.power(w, 2))
        loss = logistic - y
        cost = np.sum(loss ** 2)

        gradient = np.dot(x_trans, loss) / m

        if reg:
            cost = cost + reg
            w = (w - alpha * (gradient + reg))
        else:
            w = w - (alpha / m) * gradient
        if old_cost == cost:
            value = False
        else:
            old_cost = cost
    return w


def predict(w, X):
    result = []
    for x in X:
        p = np.dot(x[:2], w[:2]) - w[2]
        if p <= .5:
            result.append(0)
        else:
            result.append(1)
    return result


def draw_line(w):
    x = np.arange(-6, 6, 0.5)
    y = (w[1] - w[0] * x) / w[1]
    plt.plot(x, y, c='blue')


def algorithm(points_count=1000, test_points_count=20):
    # Список рандомных цветов классов
    colors = {
        0: "#" + '%06X' % random.randint(0, 0xFFFFFF),
        1: "#" + '%06X' % random.randint(0, 0xFFFFFF)
    }
    # Сгенерированные данные
    data = generate_data(points_count)
    # Координаты точек
    train_points = np.array(
        list(map(lambda v: v[0], data[:points_count - test_points_count])))
    train_classes = np.array(
        list(map(lambda v: v[1], data[:points_count - test_points_count])))

    # Рисуем сгенерированные точки
    plt.scatter(train_points[:, 0], train_points[:, 1],
                s=1, c=[colors[i] for i in train_classes])

    test_points = np.array(
        list(map(lambda v: v[0], data[points_count - test_points_count:])))

    train_points = np.hstack(
        [train_points, -1 * np.ones((train_points.shape[0], 1))])
    test_points = np.hstack(
        [test_points, -1 * np.ones((test_points.shape[0], 1))])

    w = logistic_regression(train_points, train_classes)

    predicted_classes = predict(w, test_points)

    draw_line(w)

    plt.scatter(test_points[:, 0], test_points[:, 1], s=20, c=[
        colors[i] for i in predicted_classes])

    plt.show()


algorithm(points_count=35, test_points_count=10)
