import numpy as np
from contextlib import redirect_stdout
EPSILON = 1.0E-2  # погрешность решения


def gershgorin(a):
    a = a.tolist()
    m = False
    M = False
    for i in range(len(a)):
        center = abs(a[i][i])
        radius = 0
        for j in range(len(a[0])):
            radius += abs(a[i][j]) if i != j else 0
        local_m = center - radius
        local_M = center + radius
        m = min(m, local_m) if m is not False else local_m
        M = max(M, local_M) if M is not False else local_M
    return M, m if m > 0 else 1.0E-2


def matrix_trans_none(A, b):
    return A, b


def matrix_trans_discrep(A, b):
    Q, R = np.linalg.qr(A)
    A = R.transpose()*R
    b = R.transpose()*(Q.transpose()*b)
    # scalar = min(A.min(), b.min())/2
    return (A, b)


tau = False
identity = False
A_n = False
b_n = False


def matrix_trans_simple(A, b):
    global tau, A_n, b_n, identity
    tau = 2/sum(list(gershgorin(A)))
    n = A.shape
    w, v = np.linalg.eig(A)
    # tau = 2/(min(w) + max(w))
    identity = np.matrix([[1 if j == i else 0 for j in range(n[0])] for i in range(n[1])])
    print(f'ОПТИМАЛЬНЫЙ ПАРАМЕТР ПРОСТОЙ ИТТЕРАЦИИ: {tau}\n')
    A_n = identity - tau*A
    b_n = tau*b
    return A, b


def simple_iteration(A, b, x):
    return A, b, A_n*x + b_n


def min_discrepancies(A, b, x):
    r = A * x - b  # высчитываем направление невязки (вектор)
    # высчитываем коэф. поправки вектора невязки
    t = np.dot((A * r).transpose(), r) / (np.linalg.norm(A * r)**2)
    x = x - float(t) * r  # новое решение СЛАУ
    return (A, b, x)


def tridiag(A, B):
    print(f'TRIDIAGONAL MATRIX ALGORITHM')
    print(f'Matrix A:\n{A}')
    print(f'Vector b:\n{B}')
    a, b, x = [], [], []
    n = len(B)
    a.append(-A[0][1] / A[0][0])
    b.append(B[0] / A[0][0])
    for i in range(1, n - 1):
        a.append(-A[i][i + 1] / (A[i][i - 1] * a[i - 1] + A[i][i]))
        b.append((B[i] - A[i][i - 1] * b[i - 1]) /
                 (A[i][i - 1] * a[i - 1] + A[i][i]))
    x += [(B[n - 1] - A[n - 1][n - 2] * b[n - 2]) /
          (A[n - 1][n - 1] + A[n - 1][n - 2] * a[n - 2])]
    for i in range(n - 2, -1, -1):
        x += [a[i] * x[n - 2 - i] + b[i]]
    x = np.array(x[::-1])
    print(f'\nVector x:\n{x}')
    return x


def file_input(data_file):
    with open(data_file, 'r') as file:  # считывание происходит с файла !!!
        matrix_height = int(file.readline( ))  # считываю количетво уравнений
        A = np.matrix([list(map(float, file.readline().split()))
                       for i in range(matrix_height)])  # считываю матрицу СЛАУ
        b = np.matrix(list(map(float, file.readline().split()))).transpose()  # считываю правую часть
        # инициализирую начальное приближение метода
        x = np.matrix(list(map(float, file.readline().split()))).transpose()
        return (A, b, x)


def file_print(data_file, out_data, method=min_discrepancies,
                                matrix_transform=matrix_trans_none):
    A, b, x = file_input(data_file)
    with open(out_data, 'w') as file, redirect_stdout(file):
        if method == min_discrepancies:
            print('МЕТОД МИНИМАЛЬНЫХ НЕВЯЗОК|МАМЕДОВ ВАЛЕНТИН ЮРЬЕВИЧ[17144]\n')
        else:
            print('МЕТОД ПРОСТОЙ ИТЕРАЦИИ|МАМЕДОВ ВАЛЕНТИН ЮРЬЕВИЧ[17144]\n')
        print('ДО ПРЕОБРАЗОВАНИЯ:')
        print('Матрица A:\n', A)
        print(f'Ёе число обусловленности: {np.linalg.cond(A)}\n')
        print('Правый вектор: ', b.transpose())
        print('Начальный вектор приближения', x.transpose(), end='\n\n')
        try:  # блок нужен для проверки матрицы А на положительную
            # самая быстрая проверка на полож. опред. через метод Холесского
            A, b = matrix_transform(A, b)
            #np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            print('ОШИБКА! МАТРИЦА НЕ ПОЛОЖИТЕЛЬНО ОПРЕДЕЛЕННАЯ / СИММЕТРИЧНАЯ!')
        else:
            counter = 0
            print('ПОСЛЕ ПРЕОБРАЗОВАНИЯ:')
            print('Матрица A:\n', A)
            print(f'Ёе число обусловленности: {np.linalg.cond(A)}\n')
            print('Правый вектор: ', b.transpose())
            print('Начальный вектор приближения', x.transpose(), end='\n\n')
            print('Шаг', 'Вектор решения на шаге', sep='\t\t')
            while np.linalg.norm(A*x - b) > EPSILON:  # ОСНОВНОЙ ЦИКЛ РАБОТЫ‘
                print(counter, x.transpose(), sep='\t\t')
                A, b, x = method(A, b, x)
                counter += 1

            print('\nВектор решения с точностью ',
                                EPSILON, ': ', x.transpose())
            print('Количество итераций: ', counter)



file_print('data.txt', 'out_data_discr.txt', method=min_discrepancies,
                 matrix_transform=matrix_trans_none)

file_print('data.txt', 'out_data_simpl.txt', method=simple_iteration,
                 matrix_transform=matrix_trans_simple)


