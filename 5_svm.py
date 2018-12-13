import numpy as np
import random
from matplotlib import pyplot as plt


class SVM:

    def __init__(self):
        # Матрица весов
        self.W = None

    def sgd(self, X, y, iter_count=10000):
        """
        Производит обучение машины опорных векторов
        :param X: Обучающая выборка
        :param y: Соответсвие классов
        :param iter_count: Количество итераций градиентого спуска
        :return:
        """
        if self.W is None:
            self.W = np.zeros(len(X[0]))
        eta = 1

        for it in range(1, iter_count):
            for i, x in enumerate(X):
                if (y[i] * np.dot(x, self.W)) < 1:
                    self.W += eta * ((x * y[i]) + (-2 * (1 / it) * self.W))
                else:
                    self.W += eta * (-2 * (1/it) * self.W)
                if it % 100 == 0:
                    print('Weight vector: ', self.W)

    def draw_line(self):
        w = self.W
        x = np.arange(-6, 6, 0.1)
        y = (w[2]-w[0]*x)/w[1]
        plt.plot(x, y, c='blue')

    def predict(self, X):
        return [np.sign(np.dot(x[:2], self.W[:2]) - self.W[2]) for x in X]


def generate_data(points_count):
    data = []
    classes = [1, -1]
    centers = [(random.random() * 5.0, random.random() * 5.0)
               for class_num in classes]
    for i in range(points_count):
        class_index = random.choice([0, 1])
        center_x, center_y = centers[class_index]
        data.append([[random.gauss(center_x, 0.5), random.gauss(
            center_y, 0.5)], classes[class_index]])
    return data


def algorithm(points_count=1000, test_points_count=20):
    # Список рандомных цветов классов
    colors = {
        1: "#" + '%06X' % random.randint(0, 0xFFFFFF),
        -1: "#" + '%06X' % random.randint(0, 0xFFFFFF)
    }
    # Сгенерированные данные
    data = generate_data(points_count)
    # Координаты точек
    train_points = np.array(
        list(map(lambda v: v[0], data[:points_count-test_points_count])))
    train_classes = np.array(
        list(map(lambda v: v[1], data[:points_count-test_points_count])))

    # Рисуем сгенерированные точки
    plt.scatter(train_points[:, 0], train_points[:, 1],
                s=1, c=[colors[i] for i in train_classes])

    test_points = np.array(
        list(map(lambda v: v[0], data[points_count-test_points_count:])))

    train_points = np.hstack(
        [train_points, -1*np.ones((train_points.shape[0], 1))])
    test_points = np.hstack(
        [test_points, -1*np.ones((test_points.shape[0], 1))])

    svm = SVM()
    svm.sgd(train_points, train_classes, iter_count=1000)
    predicted_classes = svm.predict(test_points)

    svm.draw_line()

    plt.scatter(test_points[:, 0], test_points[:, 1], s=20, c=[
                colors[i] for i in predicted_classes])

    plt.show()


algorithm(points_count=35, test_points_count=10)
