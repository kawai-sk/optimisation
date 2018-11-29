# coding: utf-8

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

def inner_product(x,y):#内積とノルム
    return mat_times(t(x),y)[0][0]
def two_norm(x):
    return inner_product(x,x)**0.5

A = [[1,0,0],[0,5,-1],[0,3,0]]#第1問で扱う行列
B = [[1,0,0],[0,0,-1],[0,3,0]]
x_0 = [[1],[1],[1]]

def power_method(A,x_0,e):#べき乗法
    pre = 0
    new = 10000
    x = x_0
    while abs(new-pre) >= e*abs(pre):
        y = mat_times(A,x)
        pre = new
        new = inner_product(x,y)
        x = mat_sch(y,1/two_norm(y))
    print(new)

power_method(A,x_0,10**(-6))#実際の計算

import math#実際の固有値の確認
5/2+math.sqrt(13)/2

power_method(B,x_0,0.1)

def N(n):#第2問で扱う行列
    a = zero(n,n)
    for i in range(n-1):
        a[i][i+1] = -1
    a = mat_sum(a,t(a))
    return a
def M(n,c):
    return mat_sch(I(n),c)
def C(n,c):
    return mat_sum(M(n,c),N(n))

def Jacobi(A,b,x_0,e):#Jacobi法
    n = len(A)
    M = diag([A[i][i] for i in range(n)],n)
    M2 = diag([1/A[i][i] for i in range(n)],n)
    H = mat_times(M2,mat_sub(M,A))
    c = mat_times(M2,b)
    x = x_0
    r = two_norm(mat_sub(mat_times(A,x),b))
    R = [r]
    k = 0
    while r >= e:
        x = mat_sum(mat_times(H,x),c)
        r = two_norm(mat_sub(mat_times(A,x),b))
        R.append(r)
        k += 1
        print(k,"&",r,"¥¥")
    return x,R

n = 100
D = C(n,2)
b = mat_times(D,[[i+1] for i in range(n)])
x_0 = [[1] for i in range(n)]
R1 = Jacobi(D,b,x_0,10**(0))[1]

n = 300
D = C(n,20)
b = mat_times(D,[[i+1] for i in range(n)])
x_0 = [[1] for i in range(n)]
R2 = Jacobi(D,b,x_0,10**(-10))[1]

def CG(A,b,x_0,e):#CG法
    n = len(A)
    x = x_0
    r = mat_sub(b,mat_times(A,x))
    p = r
    R = two_norm(r)
    rs = [R]
    k = 0
    while R >= e:
        q = mat_times(A,p)
        s = inner_product(p,q)
        a = inner_product(r,p)/s
        x = mat_sum(x,mat_sch(p,a))
        r = mat_sub(r,mat_sch(q,a))
        b = -inner_product(r,q)/s
        p = mat_sum(r,mat_sch(p,b))
        R = two_norm(r)
        rs.append(R)
        k += 1
        print(k,"&",R,"¥¥")
    return x,rs

n = 100
D = C(n,2)
b = mat_times(D,[[i+1] for i in range(n)])
x_0 = [[1] for i in range(n)]
w,R3 = CG(D,b,x_0,10**(0))

n = 300
D = C(n,20)
b = mat_times(D,[[i+1] for i in range(n)])
x_0 = [[1] for i in range(n)]
w,R4 = CG(D,b,x_0,10**(-10))

n = len(R1)#結果の描画
m = len(R3)
s = [i for i in range(0,n)]
u = [i for i in range(0,m)]
x = [math.log(R1[i]) for i in range(0,n)]
y = [math.log(R3[i]) for i in range(0,m)]
plt.plot(u,y,label="CG")
plt.plot(s,x,linestyle="dashed",label="Jacobi")
plt.xlabel("k")
plt.ylabel("log(||Ax-b||)")
plt.legend()
plt.show()

n = len(R2)
m = len(R4)
s = [i for i in range(0,n)]
u = [i for i in range(0,m)]
x = [math.log(R2[i]) for i in range(0,n)]
y = [math.log(R4[i]) for i in range(0,m)]
plt.plot(u,y,label="CG")
plt.plot(s,x,linestyle="dashed",label="Jacobi")
plt.xlabel("k")
plt.ylabel("log(||Ax-b||)")
plt.legend()
plt.show()

E = C(100,2)#c=2の場合の条件数の推定
F = gauss_inverse(E)
x_0 = [[1] for i in range(100)]

power_method(E,x_0,10**(-10))

power_method(F,x_0,10**(-10))
