# Маркитанов Денис КБТУ 2018 Машинное Обучение
from scipy.optimize import minimize
import math
import numpy
from matplotlib import pyplot as plt

# Начальная функция
def f(x):
    return math.sin(x / 5) * math.exp(x / 10) + 5 * math.exp(-x / 2)

result = minimize(f, 2, method='BFGS') # оптимизирую для 2
print(round(f(float(result.x)), 2))

result = minimize(f, 30, method='BFGS') # оптимизирую для 30
print(round(f(float(result.x)), 2))

steps = numpy.arange(1, 30, 0.5)
plt.plot(steps, list(map(f, steps)))