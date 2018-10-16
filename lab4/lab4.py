#cell 0
# kNN Hash Example

#cell 1
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from functools import partial
from random import random
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from collections import Counter
import math

#cell 2
## Iris dataset

#cell 3
df = load_iris()
df.data.shape

#cell 4
def f_hash(w,r,b,x):
    return int((np.dot(w,x)+b)/r)

def distance(x, y): # евклидово расстояние
    retVal = 0
    for i in range(x.shape[0]):
        retVal += pow(x[i] - y[0][i], 2)
    return math.sqrt(retVal)



#cell 5
# * https://docs.python.org/2/library/functools.html Here you can read about "partial"
# * http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html About mapping to [0,1]

#cell 6
class KNNHash(object):
    def __init__(self,m,L,nn):
        self.m = m
        self.L = L
        self.nn = nn

    def fit(self,X,y):
        self.t_hh = [] #hash table
        for j in range(self.L):
            f_hh = [] #compositional hash function
            for i in range(self.m):
                w = np.random.rand(1,X[0].shape[0]) #  weights of a hash function
                f_hh.append(partial(f_hash,w = w,r=random(),b=random())) # list of initialized hash function
            self.t_hh.append(
                (defaultdict(list),f_hh)
            )
        for n in range(X.shape[0]): 
            for j in range(self.L):
                ind = 0
                for i in range(self.m):
                    ind = ind + self.t_hh[j][1][i](x=X[n]) #calculation of index in hash table, simply sum of all hash func
                self.t_hh[j][0][ind].append((X[n],y[n])) #saving sample into corresponding index
    
    def predict(self,u): # передаем вектор
        
        for j in range(self.L):
            inds = []
            holder = []
            labelHolder = []
            for i in range(self.m):
                inds.append(self.t_hh[j][1][i](x=u)) # добавляем значение хэша с ветором u в хэш таблицу
            cntr = Counter([outp for inpt,outp in self.t_hh[j][0][sum(inds)]])
            # print(cntr) # выше можно будет вытащить все X[n] похожие на u, и считать длинну

            # решение снизу
            for out in self.t_hh[j][0][sum(inds)]: # вытаскиваем все X[n], которые лежат в одном пуле с u
                holder.append(distance(u, out)) # массив расстояний
                labelHolder.append(out[1]) # массив лэйблов, которые мы вытаскиваем из X[n]
            answer1 = holder.index(min(holder)) # находим индекс минимального
            print(min(holder)) # минимальное из всех расстояний
            print("Output: " + str(labelHolder[answer1])) # соответствующий индекс для лэйблв
            
            #Here you must put your code, extend the method with distance function and calculation with unknown sample "u"
            #Develop the rest part of kNN predict method that was discussed at the lecture

#cell 7 создание границ для ввходных данныхr
scaler = MinMaxScaler()
scaler.fit(df.data) # вычисление минимума и максимума для будущей аппроксимации
x = scaler.transform(df.data) # аппроксимация
y = df.target # перетаскиваем лэйблы в игрик


#cell 8
knnhash = KNNHash(4,4,4)
test1x = x[0] # отдельно сохраняем тестовые строки, без лэйбла
test2x = x[75]
test3x = x[149]

test1y = y[0] # отдельно сохраняем тестовые лэйблы
test2y = y[75]
test3y = y[149]
x = np.delete(x,[0,75,149],axis=0) # убираем из икса тестовые строки
y = np.delete(y,[0,75,149],axis=0) # убираем из игрика соответствующие тестовые лэйблы

knnhash.fit(x,y)
print("Expected output: " + str(test1y))
knnhash.predict(test1x)
print("-------------")
print("Expected output: " + str(test2y))
knnhash.predict(test2x)
print("-------------")
print("Expected output: " + str(test3y))
knnhash.predict(test3x)

#cell 9
# * Each string above corresponds to the particular hash table. And index in counter maps to the class. For example Counter({0: 13, 1: 1}) means that there are 13 samples close to "u" with "0" class labels and 1 sample with "1" class label.

#cell 10


