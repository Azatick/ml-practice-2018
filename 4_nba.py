import numpy as np
from matplotlib import pyplot as plt
import random
from beautifultable import BeautifulTable

symptoms_label = [
    'Недомогание',
    'Слабость',
    'Головная боль',
    'Отсутствие аппетита',
    'Болезненное открывание рта',
    'Боль в горле',
    'Боль при глотании слева',
    'Боль при глотании справа',
    'Дисфагия',
    'Тризм жевательной мускулатуры I степени',
    'Тризм жевательной мускулатуры III степени',
    'Болезненность л/узлов в левой подчелюстной области'
]

diseases_label = [
    'Острый левостор паратонз абсцесс',
    'Острый правостор паратонз абсцесс',
    'Острый двухстор паратонз абсцесс',
    'Острый левостор паратонзиллит',
    'Острый правостор паратонзиллит',
    'Острый левостор парафарин абсцесс',
    'Острый правостор парафарин абсцесс',
    'Острый левостор парафарингит',
    'Острый правостор парафарингит'
]

p_s_d = np.array(
    [
        [.97, .96, 1, .92, .91, 1, 1, 1, .9],
        [.99, .98, 1, .96, 1, 1, 1, .7, 1],
        [.66, .73, .88, .76, .79, .5, .5, 1, .6],
        [.73, .75, .88, .76, .86, .5, .5, .8, .6],
        [.93, .9, 1, .8, .75, .5, .5, .57, .4],
        [.97, .96, 1, 1, .83, .5, .1, .85, .9],
        [.93, .01, .77, .96, .03, .1, .0001, .85, .0001],
        [.01, .94, .77, .0001, .86, .0001, .1, .0001, .8],
        [.96, .94, 1, 1, .93, .5, 1, 1, 1],
        [.53, .64, .77, .6, .41, .0001, 1, .42, .6],
        [.04, .02, 0, .08, .06, .5, .0001, .0001, .0001],
        [.75, .08, 1, .72, .0001, 1, .5, .85, .09]
    ]
)


def generate_data(count):
    return [random.choice([0, 1]) for i in range(count)]


def algorithm():
    symptoms = generate_data(len(p_s_d))

    symptoms_table = BeautifulTable(max_width=120)
    symptoms_table.column_headers = symptoms_label[:]
    symptoms_table.append_row(map(lambda v: 'Да' if v == 1 else 'Нет', symptoms))
    print(symptoms_table)

    # Количество случаев болезней
    disease_counts = [116, 96, 9, 30, 26, 3, 3, 12, 8]
    # Вероятности болезней
    p_disease = disease_counts / np.sum(disease_counts)

    # число болезней
    d_len = p_s_d.shape[1]
    # число симптомов
    s_len = p_s_d.shape[0]
    # Вероятности болезней
    result = np.zeros(d_len)
    for i in range(d_len):
        p_d_s = 1
        for j in range(s_len):
            if symptoms[j] == 1:
                p_d_s *= p_disease[i] * p_s_d[j][i]
        result.put(i, p_d_s)

    diseases_table = BeautifulTable(max_width=120)
    diseases_table.numeric_precision = 20
    diseases_table.column_headers = diseases_label[:]
    diseases_table.append_row(result)
    print(diseases_table)

    print('Вероятно у пациента ' + diseases_label[result.argmax()])


algorithm()
