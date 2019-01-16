def devided_matrix(A,i,j):#行列関係の関数の定義
    m = len(A)
    n = len(A[0])
    if i < 1 or j < 1 or i > m or j > n:
        return None
    else:
        B = [[0 for l in range(0,n-1)]for k in range(0,m-1)]
        for k in range(0,m):
            for l in range(0,n):
                if k < i-1 and l < j-1:
                    B[k][l] = A[k][l]
                elif k < i-1 and l > j-1:
                    B[k][l-1] = A[k][l]
                elif k > i-1 and l < j-1:
                    B[k-1][l] = A[k][l]
                elif k > i-1 and l > j-1:
                    B[k-1][l-1] = A[k][l]
        return B
def det(A):
    m = len(A)
    n = len(A[0])
    if m != n:
        return None
    else:
        if m == 1:
            return A[0][0]
        elif m == 2:
            return A[0][0]*A[1][1]-A[0][1]*A[1][0]
        else:
            d = 0
            for i in range(0,n):
                if i/2 == i//2:
                    d = d + A[i][0]*det(devided_matrix(A,i+1,1))
                else:
                    d = d - A[i][0]*det(devided_matrix(A,i+1,1))
            return d
def inversed(A):
    if len(A) != len(A[0]):
        return None
    else:
        d = det(A)
        if d == 0:
            return None
        else:
            n = len(A)
            B = [[0 for l in range(0,n)]for k in range(0,n)]
            for i in range(0,n):
                for j in range(0,n):
                    if (i+j)/2==(i+j)//2:
                        B[i][j] = det(devided_matrix(A,j+1,i+1))/d
                    elif (i+j)/2!=(i+j)//2:
                        B[i][j] = -det(devided_matrix(A,j+1,i+1))/d
            return B
def gauss_inverse(A):
    n = len(A)
    A2 = [[A[i][j] for j in range(n)] for i in range(n)]
    B = I(n)
    for i in range(n):
        if A2[i][i] != 1:
            p = A2[i][i]
            for j in range(n):
                A2[i][j] /= p
                B[i][j] /= p
        for j in range(n):
            if j != i and A2[j][i] != 0:
                q = A2[j][i]/A2[i][i]
                for k in range(n):
                    A2[j][k] -= q*A2[i][k]
                    B[j][k] -= q*A2[i][k]
    return B
def mat_times(A,B):
    i = len(A)
    j = len(A[0])
    k = len(B)
    l = len(B[0])
    if j != k:
        return None
    else:
        C = [[0 for p in range(0,l)]for q in range(0,i)]
        for x in range(0,i):
            for y in range(0,l):
                for m in range(0,j):
                    C[x][y] = C[x][y] + A[x][m]*B[m][y]
        return C
def mat_sum(A,B):
    return [[A[i][j]+B[i][j] for j in range(len(A[0]))]for i in range(len(A))]
def mat_sub(A,B):
    return [[A[i][j]-B[i][j] for j in range(len(A[0]))]for i in range(len(A))]
def mat_sch(A,a):
    return [[A[i][j]*a for j in range(len(A[0]))]for i in range(len(A))]
def t(a):
    n = len(a)
    m = len(a[0])
    return [[a[i][j] for i in range(n)]for j in range(m)]
def I(n):
    return [[1 if j == i else 0 for j in range(n)]for i in range(n)]
def zero(n,m):
    return [[0 for j in range(n)]for i in range(n)]
def diag(a,n):
    return [[a[i] if j == i else 0 for j in range(n)]for i in range(n)]

def f(x):#扱う微分方程式の右辺
    return [[simplify(-x[1][0])],[simplify(x[0][0])]]

def yo_euler(h,n):#陽的Euler法(P,Q)=(p,q)+h(-q,p)=(p-hq,q+hp)
    p,q = 1,0
    res = [[1],[0]]
    for i in range(n):
        p,q = p-h*q,q+h*p
        res[0].append(p)
        res[1].append(q)
        #print(p,q)
    return res

def in_euler(h,n):#陰的Euler法(P,Q)=(p,q)+h(-Q,P)->(P,Q)=(p-hq,q+hp)/(1+h^2)
    p,q = 1,0
    res = [[1],[0]]
    for i in range(n):
        p,q = (p-h*q)/(1+h**2),(q+h*p)/(1+h**2)
        res[0].append(p)
        res[1].append(q)
        #print(p,q)
    return res

#台形則(P,Q)=(p,q)+h/2(-q,p)+h/2(-Q,P)-> (P,Q)=((1-h^2/4)p-hq,hp+(1-h^2/4)q)/(1+h^2/4)
def daikei(h,n):
    p,q = 1,0
    res = [[1],[0]]
    for i in range(n):
        p,q = ((1-h**2/4)*p-h*q)/(1+h**2/4),((1-h**2/4)*q+h*p)/(1+h**2/4)
        res[0].append(p)
        res[1].append(q)
        #print(p,q)
    return res

import sympy#ルンゲクッタ法の表式を代数計算により求める．(p,q,h)が(x,y,z)に対応
from sympy import *
from scipy.optimize import fmin
from scipy import optimize
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
from sympy import init_printing
init_printing()
X = [[x],[y]]
k_1=f(X)
k_2=f(mat_sum(X,mat_sch(k_1,z/2)))
k_3=f(mat_sum(X,mat_sch(k_2,z/2)))
k_4=f(mat_sum(X,mat_sch(k_3,z)))
l=mat_sum(k_1,mat_sum(mat_sch(k_2,2),mat_sum(mat_sch(k_3,2),k_4)))
mat_sum(X,mat_sch(l,z/6))

def runge_kutta(h,n):
    p,q = 1,0
    res = [[1],[0]]
    for i in range(n):
        p,q = p+h*((p*h**3)/24-(p*h)/2+(q*h**2)/6-q),q+h*(q*h**3/24-q*h/2-p*h**2/6+p)
        res[0].append(p)
        res[1].append(q)
        #print(p,q)
    return res

#結果の描画

get_ipython().magic('matplotlib notebook')
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
h,n = 0.1,10000
res = yo_euler(h,n)
plt.plot(res[0],res[1],"-",color=(0.0,0.0,0.0))
plt.title("Explicit_Euler_Method,"+"n="+str(n)+",h="+str(h))
plt.xlabel("p")
plt.ylabel("q")
plt.show()

h,n = 0.1,10000
res = in_euler(h,n)
plt.plot(res[0],res[1],"-",color=(0.0,0.0,0.0))
plt.title("Implicit_Euler_Method,"+"n="+str(n)+",h="+str(h))
plt.xlabel("p")
plt.ylabel("q")
plt.show()

get_ipython().magic('matplotlib notebook')
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
h,n = 0.1,10000
res = daikei(h,n)
plt.plot(res[0],res[1],"-",color=(0.0,0.0,0.0))
plt.title("daikeisoku,"+"n="+str(n)+",h="+str(h))
plt.xlabel("p")
plt.ylabel("q")
plt.show()

get_ipython().magic('matplotlib notebook')
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
h,n = 0.1,10000
res = runge_kutta(h,n)
plt.plot(res[0],res[1],"-",color=(0.0,0.0,0.0))
plt.title("runge_kutta,"+"n="+str(n)+",h="+str(h))
plt.xlabel("p")
plt.ylabel("q")
plt.show()
