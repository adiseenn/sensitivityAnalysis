from scipy.optimize import linprog
from numpy import *
import numpy as np

data = []
with open("data.txt", 'r') as file:
    for line in file:
            data.append([int(x) for x in line.split()])

c = data[0]
A_1 = [data[1], data[2], data[3]]
b_1 = data[4]

res = linprog(c, A_1, b_1, method='revised simplex')
print(res)

#анализ на чувствительность - изменение коэффицентов целевой функции (при изменении цен на реактивы)
c_new = data[5]

m = 3
A = np.array(A_1)
B = np.array(b_1)

# формирование матрицы единичных векторов для дополнительных переменных
A_slack = np.zeros([m, m])
for i in range(m):
    A_slack[i, i] = 1

print()
print('Матрица единичных векторов: ')
print(A_slack)
print()

# получение номеров базисных векторов по решению задачи (ненулевые компоненты) - основные вектора
base_ind = np.nonzero(res.x)[0]
# получение номеров базисных векторов по решению задачи (ненулевые компоненты) - дополнительные вектора
base_ind_dop = np.nonzero(res.slack)[0] + m
# объединение списка номеров базисных переменных
base_ind = np.concatenate((base_ind, base_ind_dop))
print('Номера базисных переменных: ')
print(base_ind)
print()

# формирование базисной матрицы и столбца базисных коэффициентов целевой функции
basis = []
c_bas = []
for i in range(m):
    # если вектор основной
    if base_ind[i] < m:
        basis.append(A[:, base_ind[i]])
        print(basis)
        c_bas.append(c_new[base_ind[i]])
        print(c_bas)

    else:
        # если вектор дополнительный
        basis.append(A_slack[:, base_ind[i] - m])
        print(basis)
        c_bas.append(0)
        print(c_bas)

# вектора добавляются в матрицу как вектора-строки. Нужно - вектора-столбцы проведение транспонирвания
basis = np.reshape(basis, (m, m)).T
print(basis)
# получение обратной матрицы
basis_l = np.linalg.inv(basis)
print(basis_l)

while True:
    # получение вектора cb*B(-1) для дальнейшего получения оценок
    cb = np.dot(c_bas, basis_l)
    # получение оценок основных векторов
    delta = np.dot(cb, A) - c_new
    # получение оценок дополнительных векторов
    delta_dop = np.dot(cb, A_slack)
    # получение коэффициентов разложения вектора p_0
    p_0 = np.dot(basis_l, B)

    # получение максимальной оценки
    max_delta = np.max(delta)
    # если максимальная оценка равна 0, получен оптимум
    max_delta_dop = np.max(delta_dop)
    if max(max_delta, max_delta_dop) == 0:
        print('Оптимум')
        break
    else:
        # если положительная оценка у основного вектора
        if max_delta > 0:
            # запоминаем индекс основного вектора
            ind_to_basis = np.argmax(delta)
        else:
            # запоминаем индекс дополнительного вектора - корректируем номера
            # векторов
            ind_to_basis = np.argmax(delta_dop) + m

    # вычисляем коэффициенты разложения по базису вектора с положительно
    # оценкой
    if ind_to_basis < m:
        p_j = np.dot(basis_l, A[:, ind_to_basis])
        c_new_basis = c_new[ind_to_basis]
    else:
        p_j = np.dot(basis_l, A_slack[:, ind_to_basis - m])
        c_new_basis = 0

    # находим в вектор, который выводится из базиса
    ind = -1
    min = 100000
    for i in range(m):
        if p_j[i] > 0:
            if p_0[i]/p_j[i] < min:
                ind = i
                min = p_0[i]/p_j[i]

    # осуществляем пересчет по формулам Гаусса
    # замена номера базисного вектора
    base_ind[ind] = ind_to_basis
    # замена коэффициента целевой функции при базисном векторе
    c_bas[ind] = c_new_basis
    # пересчет
    basis_l[ind, :] = basis_l[ind, :]/p_j[ind]
    for i in range(m):
        if i != ind:
            basis_l[i, :] = basis_l[i, :] - basis_l[i, :] * p_j[i]

# печать коэффициентов разложения p_0, из которого можно получить ответ
print(p_0)
print(base_ind)

