###############################################################################
#行列計算に用いる関数の準備

import math
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

#行列の積
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

#転置
def t(a):
    n = len(a)
    m = len(a[0])
    return [[a[i][j] for i in range(n)]for j in range(m)]
#単位行列
def I(n):
    return [[1 if j == i else 0 for j in range(n)]for i in range(n)]

#内積とノルム
def inner_product(x,y):
    return mat_times(t(x),y)[0][0]
def two_norm(x):
    return inner_product(x,x)**0.5

#行列の和・差・スカラー倍
def mat_sum(A,B):
    return [[A[i][j]+B[i][j] for j in range(len(A[0]))]for i in range(len(A))]
def mat_sub(A,B):
    return [[A[i][j]-B[i][j] for j in range(len(A[0]))]for i in range(len(A))]
def mat_sch(A,a):
    return [[A[i][j]*a for j in range(len(A[0]))]for i in range(len(A))]

#行列Aの最大固有値を初期ベクトルx_0から出発して許容誤差eで求めるべき乗法
def power_method(A,x_0,e):
    pre = 0
    new = 10000
    x = x_0
    while abs(new-pre) >= e*abs(pre):
        y = mat_times(A,x)
        pre = new
        new = inner_product(x,y)
        x = mat_sch(y,1/two_norm(y))
    return new

#余因子行列
def devided_matrix(A,i,j):
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
#行列式
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
#逆行列
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

###############################################################################
#適当な問題設定

A = [[2,3,1],[1,3,5]]
b = [[2],[-3]]
w_0 = [[0],[0],[0]]

#計算用の関数
def f(A,b,c,w):
    bAw = mat_sub(b,mat_times(A,w))
    return inner_product(bAw,bAw) + c*inner_product(w,w)
def gradient(A,B,c,w):
    m = len(A)
    n = len(A[0])
    b_1 = mat_times(t(A),b) # A^Tb
    b_2 = mat_sch(b_1,2) # 2A^Tb
    A_1 = mat_sum(mat_times(t(A),A),mat_sch(I(n),c)) # A^TA+cI
    A_2 = mat_sch(A_1,2) # 2(A^TA+cI)

    return mat_sub(mat_times(A_2,w),b_2)

###############################################################################
#Q1.

#steepest descent method
def SDM(A,b,c,w_0,e):
    #アルゴリズムで用いる行列などを事前に用意する．
    m = len(A)
    n = len(A[0]) #仮定より m<n
    b_1 = mat_times(t(A),b) # A^Tb
    b_2 = mat_sch(b_1,2) # 2A^Tb
    bb = inner_product(b,b)
    A_1 = mat_sum(mat_times(t(A),A),mat_sch(I(n),c)) # A^TA+cI
    A_2 = mat_sch(A_1,2) # 2(A^TA+cI)

    #アルゴリズム内での計算軽量化
    def F(w):
        return inner_product(w,mat_times(A_1,w)) - inner_product(w,b_2) + bb
    def Gradient(w):
        return mat_sub(mat_times(A_2,w),b_2)

    #Lを求める
    x_0 = [[1] for i in range(n)] #べき乗法の初期ベクトル．
    L = 2*power_method(A_1,x_0,10**(-6)) #L-平滑

    #最急降下法
    k = 0
    w = w_0
    g = Gradient(w)
    numbers = []
    values = []
    while two_norm(g) >= e:
        w = mat_sub(w,mat_sch(g,1/L))
        g = Gradient(w)
        k += 1
        numbers.append(k)
        values.append(F(w))

    #結果の出力用
    Q = 1
    WantToKnow = False
    WantToPlot = False

    #理論的な最適解
    if WantToKnow:
        if Q == 1:
            print("実行回数:"+str(k))
            print("数値解の勾配:"+str(two_norm(Gradient(w))))
        if Q == 2 or 3:
            print("λ="+str(c)+",実行回数:"+str(k))
        if det(A_1) != 0:
            w_d = mat_times(inversed(A_1),b_1)
            print("最適解との誤差:"+str(two_norm(mat_sub(w,w_d))))
            print("最適値との誤差:"+str(abs(F(w)-F(w_d))))
            print(F(w),F(w_d))
        else:
            print("最適解不明")

    #描画
    if WantToPlot:
        LogPlot = False #対数グラフ用
        label = "f(w_k)"
        if LogPlot:
            if det(A_1) != 0:
                values = [math.log(abs(values[i]-F(w_d))) if values[i] != F(w_d) else math.log(10**(-15)) for i in range(len(values))]
                label = "log|f(w_k)-(opt)|"
            else:
                values = [math.log(values[i]) for i in range(len(values))]
                label = "logf(w_k)"
        plt.plot(numbers,values,label="Constant Rule")
        plt.xlabel("k")
        plt.ylabel(label)
        plt.legend()
        plt.show()
    if Q == 2 or 3: #Q2,Q3での反復回数の比較のため
        return k
    else:
        return w

#SDM(A,b,0,w_0,10**(-6))
#SDM(A,b,1,w_0,10**(-6))
#SDM(A,b,10,w_0,10**(-6))
#SDM(A,b,100,w_0,10**(-6))

###############################################################################
#Q2.

#steepest descent method, Armijo則
def SDM_Armijo(A,b,c,w_0,e,a_0,eps,tau):
    #アルゴリズムで用いる行列などを事前に用意する．
    m = len(A)
    n = len(A[0]) #仮定より m<n
    b_1 = mat_times(t(A),b) # A^Tb
    b_2 = mat_sch(b_1,2) # 2A^Tb
    bb = inner_product(b,b)
    A_1 = mat_sum(mat_times(t(A),A),mat_sch(I(n),c)) # A^TA+cI
    A_2 = mat_sch(A_1,2) # 2(A^TA+cI)

    #アルゴリズム内での計算軽量化
    def F(w):
        return inner_product(w,mat_times(A_1,w)) - inner_product(w,b_2) + bb
    def Gradient(w):
        return mat_sub(mat_times(A_2,w),b_2)
    def condition(w,g,gg,l):
        d = a_0*tau**l
        c_1 = F(mat_sum(w,mat_sch(g,-d)))
        c_2 = -eps*d*gg
        return c_1 - c_2

    k = 0
    w = w_0
    g = Gradient(w)
    numbers = [0]
    values = []
    while two_norm(g) >= e:
        l = 0
        Fw = F(w)
        gg = inner_product(g,g)
        while condition(w,g,gg,l) > Fw:
            l += 1
        w = mat_sum(w,mat_sch(g,-a_0*tau**l))
        g = Gradient(w)
        k += 1

    #理論的な最適解
    WantToKnow = False
    if WantToKnow:
        print("λ="+str(c)+",実行回数:"+str(k))
        if det(A_1) != 0:
            w_d = mat_times(inversed(A_1),b_1)
            print("最適解との誤差:"+str(two_norm(mat_sub(w,w_d))))
            print("最適値との誤差"+str(abs(F(w)-F(w_d))))
        else:
            print("最適解不明")

    Q = 2
    if Q == 2:
        return k
    else:
        return w

#素直な最急降下法との比較
WantToCompare_Armijo = False
if WantToCompare_Armijo:
    lamb = []
    Cons = []
    Armijo = []
    for i in range(1001):
        lamb.append(i)
        Cons.append(SDM(A,b,i,w_0,10**(-6)))
        Armijo.append(SDM_Armijo(A,b,i,w_0,10**(-6),1,10**(-3),0.5))
    plt.plot(lamb,Cons,label="Constant Rule")
    plt.plot(lamb,Armijo,linestyle="dashed",label="Armijo Rule")
    plt.xlabel("λ")
    plt.ylabel("k")
    plt.legend()
    plt.show()

#パラメータを変えたArmijo則の比較
WantToCompare_Armijos = False
if WantToCompare_Armijos:
    lamb = []
    Armijo1 = []
    Armijo2 = []
    Armijo3 = []
    for i in range(501):
        lamb.append(i)
        Armijo1.append(SDM_Armijo(A,b,i,w_0,10**(-6),1,10**(-1),0.5))
        Armijo2.append(SDM_Armijo(A,b,i,w_0,10**(-6),1,10**(-3),0.5))
        Armijo3.append(SDM_Armijo(A,b,i,w_0,10**(-6),1,10**(-5),0.5))
        print(i )
    plt.plot(lamb,Armijo1,linestyle="dashed",label="ξ=0.1")
    plt.plot(lamb,Armijo2,label="ξ=0.001")
    plt.plot(lamb,Armijo3,linestyle="dotted",label="ξ=0.00001")
    plt.xlabel("λ")
    plt.ylabel("k")
    plt.legend()
    plt.show()

#SDM_Armijo(A,b,7,w_0,10**(-6),1,10**(-3),0.5)
#SDM_Armijo(A,b,20,w_0,10**(-6),1,10**(-3),0.5)
#SDM_Armijo(A,b,84,w_0,10**(-6),1,10**(-3),0.5)

#条件数の計算
CheckCondition = False
if CheckCondition:
    lamb = []
    cond = []
    for c in range(1001):
        n = len(A[0]) #仮定より m<n
        A_1 = mat_sum(mat_times(t(A),A),mat_sch(I(n),c)) # A^TA+cI
        x_0 = [[1] for i in range(n)]
        if det(A_1) != 0:
            Lmax = power_method(A_1,x_0,10**(-6))
            Lmin = power_method(inversed(A_1),x_0,10**(-6))
            lamb.append(c)
            cond.append((Lmax-Lmin)/(Lmax+Lmin))
    plt.plot(lamb,cond,label="Condition Number")
    plt.xlabel("λ")
    plt.ylabel("λmax/λmin")
    plt.legend()
    plt.show()

###############################################################################
#Q3.

#steepest descent method, Nesterovs加速
def SDM_Nesterov(A,b,c,w_0,e):
    #最アルゴリズムで用いる行列などを事前に用意する．
    m = len(A)
    n = len(A[0]) #仮定より m<n
    b_1 = mat_times(t(A),b) # A^Tb
    b_2 = mat_sch(b_1,2) # 2A^Tb
    A_1 = mat_sum(mat_times(t(A),A),mat_sch(I(n),c)) # A^TA+cI
    A_2 = mat_sch(A_1,2) # 2(A^TA+cI)
    bb = inner_product(b,b)

    #アルゴリズム内での計算軽量化
    def F(w):
        return inner_product(w,mat_times(A_1,w)) - inner_product(w,b_2) + bb
    def Gradient(w):
        return mat_sub(mat_times(A_2,w),b_2)

    #Lを求める
    x_0 = [[1] for i in range(n)] #べき乗法の初期ベクトル．
    L = 2*power_method(A_1,x_0,10**(-6)) #L-平滑

    k = 0
    w = w_0
    ww = w_0
    rho = 1
    g = Gradient(w)
    numbers = [0]
    values = []
    while two_norm(g) >= e:
        w_new = mat_sub(ww,mat_sch(g,1/L))
        dw = mat_sub(w_new,w)
        ww = mat_sum(w_new,mat_sch(dw,(2*rho-2)/(1+(1+4*rho**2)**0.5)))
        if F(w_new) >= F(w) or inner_product(g,dw) >= 0:
            rho = 1
        else:
            rho = (1+(1+4*rho**2)**0.5)/2
        w = w_new
        g = Gradient(ww)
        k += 1

    Q = 3
    WantToKnow = False

    #理論的な最適解
    if WantToKnow:
        if Q == 1:
            print("実行回数:"+str(k))
            print("数値解の勾配:"+str(two_norm(Gradient(w))))
        if Q == 2 or 3:
            print("λ="+str(c)+",実行回数:"+str(k))
        if det(A_1) != 0:
            w_d = mat_times(inversed(A_1),b_1)
            print("最適解との誤差:"+str(two_norm(mat_sub(w,w_d))))
            print("最適値との誤差:"+str(abs(F(w)-F(w_d))))
            print(F(w),F(w_d))
        else:
            print("最適解不明")

    if Q == 3:
        return k
    else:
        return w

#素直な最急降下法との比較
WantToCompare_Nesterov = False
if WantToCompare_Nesterov:
    lamb = []
    Cons = []
    Nesterov = []
    for i in range(1001):
        lamb.append(i)
        Cons.append(SDM(A,b,i,w_0,10**(-6)))
        Nesterov.append(SDM_Nesterov(A,b,i,w_0,10**(-6)))
    plt.plot(lamb,Cons,label="Constant Rule")
    plt.plot(lamb,Nesterov,linestyle="dashed",label="Nesterov Rule")
    plt.xlabel("λ")
    plt.ylabel("k")
    plt.legend()
    plt.show()
