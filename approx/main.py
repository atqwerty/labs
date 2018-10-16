# Маркитанов Денис КБТУ 2018 Машинное обучение
import re
import numpy
import math
from matplotlib import pyplot as plt

# наша функция
def f(x):
    return math.sin(x / 5) * math.exp(x / 10) + 5 * math.exp(-x / 2)
# создаем матрицу
def createMatrix(x):
    n = len(x)
    matrix = numpy.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = x[i] ** j
    return matrix
# раскручиваем решение полиномов n порядка (где n это размерность решений линейной системы уравнений)
def getSolutions(rank, scope):
    y = numpy.zeros(shape=(len(scope)))
    for i in range(len(scope)):
        for j in range(len(rank)):
                y[i] += rank[j] * (scope[i] ** j) 
    return y

def linearEquationSolver(x): # функция для решения систем линейных уравнений
    m = createMatrix(x)
    v = list(map(f, x))
    return numpy.linalg.solve(m, v) # решение системы линейных уравнений

scope = numpy.arange(1, 15, .1) # делаю полотно для построения графиков
x = [[1, 15], [1, 8, 15], [1, 4, 10, 15]] # данные икса из задачи
buffer = [] # будет содержать решения линейных систем
y = []

for i in range(3):
    buffer.append(linearEquationSolver(x[i])) # для каждого полинома, который представлен в задаче решаю систему линейных уравнений
                    
yMain = list(map(f, scope))
for i in range(len(buffer)):
    y.append(getSolutions(buffer[i], scope)) # нахожу решения полинома n порядка при помощи решенных систем линейного уравнения (A * x = B)

for i in range(len(y)): # рисую
    plt.plot(scope, y[i])
plt.plot(scope, yMain)
plt.show()