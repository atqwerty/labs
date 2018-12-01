# Markitanov Denis, KBTU, Machine Learning 2018

import numpy as np
import pandas as pd
from math import sqrt
import scipy
from scipy import linalg

def SVD(mat):
    print(mat)
    AT = mat.transpose()
    ATdotA = AT.dot(mat)            
    lmbd, V = linalg.eig(ATdotA)
    print(lmbd)
    VT = V.transpose()

    for i in range(len(lmbd)):
        lmbd[i] = sqrt(abs(lmbd[i]))

    lmbd = sorted(lmbd, reverse = True)
    S = np.diag(lmbd)
    S_inversed = np.linalg.inv(S)

    for i in range(len(V)):
        print(V[:,i])

    U = mat.dot(V.dot(S_inversed))
    u, s, v = np.linalg.svd(mat)

    print("-------------")
    print(U)
    print(S)
    print(V)
    print("-------------")
    print(u)
    print(s)
    print(v)          
    print("-------------")
    print(mat)
    print(U.dot(S.dot(VT)))


# ----- Main
mat = np.array([[0, 1, 1], [sqrt(2), 2, 0], [0, 1, 1]])
SVD(mat)