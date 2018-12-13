import random
import math

import numpy as np
from matplotlib import pyplot as plt


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


def logistic_regression(x, y, alpha=0.05, lamda=0):
    m, n = np.shape(x)
    theta = np.ones(n)
    xTrans = x.transpose()
    oldcost = 0.0
    value = True
    while(value):
        hypothesis = np.dot(x, theta)
        logistic = hypothesis/(np.exp(-hypothesis)+1)
        reg = (lamda/2*m)*np.sum(np.power(theta, 2))
        loss = logistic - y
        cost = np.sum(loss ** 2)
        # print(cost)
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        # avg gradient per example
        gradient = np.dot(xTrans, loss)/m
        # update
        if(reg):
            cost = cost+reg
            theta = (theta - (alpha) * (gradient+reg))
        else:
            theta = theta - (alpha/m) * gradient
        if(oldcost == cost):
            value = False
        else:
            oldcost = cost
    return theta


def predict(w, X):
    result = []
    for x in X:
        a = math.exp(w[2] + np.dot(x[:2], w[:2]))
        result.append(np.sign(a/(1 + a)))
    return result


def draw_line(w):
    x=np.arange(-6, 6, 0.1)
    y=(w[2]-w[0]*x)/w[1]
    plt.plot(x, y, c='blue')


def algorithm(points_count=1000, test_points_count=20):
    # Список рандомных цветов классов
    colors={
        0: "#" + '%06X' % random.randint(0, 0xFFFFFF),
        1: "#" + '%06X' % random.randint(0, 0xFFFFFF)
    }
    # Сгенерированные данные
    data=generate_data(points_count)
    # Координаты точек
    train_points=np.array(
        list(map(lambda v: v[0], data[:points_count-test_points_count])))
    train_classes=np.array(
        list(map(lambda v: v[1], data[:points_count-test_points_count])))

    # Рисуем сгенерированные точки
    plt.scatter(train_points[:, 0], train_points[:, 1],
                s=1, c=[colors[i] for i in train_classes])

    test_points=np.array(
        list(map(lambda v: v[0], data[points_count-test_points_count:])))

    train_points=np.hstack(
        [train_points, -1*np.ones((train_points.shape[0], 1))])
    test_points=np.hstack(
        [test_points, -1*np.ones((test_points.shape[0], 1))])

    w=logistic_regression(train_points, train_classes)

    draw_line(w)

    print(predict(w, test_points))

    plt.show()


algorithm(points_count=35, test_points_count=10)
