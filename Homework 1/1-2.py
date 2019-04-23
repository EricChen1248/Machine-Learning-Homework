''' kernel trick '''


from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt

def polynomial(x1, x2):
    return (1 + np.dot(x1,x2))**2

def fit(x, y, kernel): 
    n = x.shape[0]
    # min       1/2 XT Q x  + p x
    # s.t.      gx <= h
    #           ax = b
    
    # min       1/2 ΣΣ α α y y z z - Σ α
    # s.t.      Σ y α = 0
    #           α >= 0

    # qnm = yn ym k(xn,xm)
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = kernel(x[i], x[j])

    Q = matrix(np.outer(y,y) * K)
    
    # -1 matrix of size n × 1
    p = matrix(np.ones((n))) * -1.
    # Negative Identity Matrix of size n × n
    G = matrix(-np.eye(n))
    # Vector of zero of size n × 1
    h = matrix(np.zeros(n))
    # y label in size of n × 1
    A = matrix(y, (1,n))
    # zero scalar
    b = matrix(0.)

    solvers.options['show_progress'] = False
    sol = solvers.qp(Q, p, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

def solve_hard_margin_kernel(x, y, kernel):
    # fit svm classifier
    alphas = fit(x, y, kernel)
    print("Alphas:\n")
    print(alphas)
    return alphas

def calc_equation(x, y, alphas, kernel, sv):
    n = x.shape[0]
    s = sv[0]
    b = 0
    for i in sv:
        b += y[s] - alphas[i] * y[i] * kernel(x[i], x[s])

    print("b:", b)

x = np.array([[1, 0], [0,1], [0,-1], [-1,-0], [0, 2], [0, -2], [-2, 0]]) * 1.
y = np.array([-1, -1, -1, 1, 1, 1, 1]) * 1.
# x = np.array([[-2, -2], [4,5], [4,-1], [6,-2], [10, -7], [10, 1], [10, 1]]) * 1.
# y = np.array([-1, -1, -1, 1, 1, 1, 1]) * 1.

# x = np.array([[4.,-1.], [6.,-2.]])
# y = np.array([-1., 1.])

alphas = solve_hard_margin_kernel(x, y, polynomial)
calc_equation(x, y, alphas, polynomial, range(1,6))

'''
alphas:
4.32455522e-09
7.03703704e-01
7.03703704e-01
8.88888891e-01
2.59259260e-01
2.59259260e-01
5.27081514e-10
'''