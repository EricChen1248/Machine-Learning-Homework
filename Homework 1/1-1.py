''' non linear transformation '''

from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt


'''
x1 = ( 1, 0) = -1 = (-2,-2)
x2 = ( 0, 1) = -1 = ( 4, 5)
x3 = ( 0,-1) = -1 = ( 4,-1)
x4 = (-1, 0) =  1 = ( 6,-2)
x5 = ( 0, 2) =  1 = (10,-7)
x6 = ( 0,-2) =  1 = (10, 1)
x7 = (-2, 0) =  1 = (10, 1)

z = 2(x2)^2 − 4(x1) + 2
z =  (x1)^2 − 2(x2) − 3
'''

def transform(x):
    return (2*x[1]*x[1] - 4 * x[0] + 2, x[0] * x[0] - 2 * x[1] - 3)

def fit(x, y): 
    n = x.shape[0]
    y = y.reshape(-1,1) * 1.

    # min       1/2 XT P x  + q x
    # s.t.      gx < h
    #           ax = b

    X_dash = y * np.array(x)
    H = np.dot(X_dash, X_dash.T) * 1.
    P = matrix(H)
    # -1 matrix of size n × 1
    q = matrix(-np.ones((n, 1)))
    # Negative Identity Matrix of size n × n
    G = matrix(-np.eye(n))
    # Vector of zero of size n × 1
    h = matrix(np.zeros(n))
    # y label in size of n × 1
    A = matrix(y.reshape(1, -1))
    # zero scalar
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = True
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

def plot_data_with_labels(x, y, ax):
    COLORS = ['red', 'blue']
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[y == unique[li]]
        ax.scatter(x_sub[:, 0], x_sub[:, 1], c = COLORS[li])


def plot_separator(ax, slope, intercept):
    x = np.arange(0, 10)
    ax.plot(x, x * slope + intercept, 'k-')


def plot_vertical(ax, x):
    ax.plot([x,x],[-10,5])

def solve_hard_margin(x, y, plot = False):
    # fit svm classifier
    alphas = fit(x, y)
    print(alphas)

    # get weights
    w = np.sum(alphas * y[:, None] * x, axis = 0)
    # get bias
    cond = (alphas > 1e-4).reshape(-1)
    b = y[cond] - np.dot(x[cond], w)
    bias = b[0]

    # normalize
    norm = np.linalg.norm(w)
    w, bias = w / norm, bias / norm
    
    slope = -w[0] / w[1]
    intercept = -bias / w[1]
    print(w[0], "* x +", w[1], "* y +", bias)

    # show data and w
    if plot:
        _, ax = plt.subplots()
        if abs(slope) > 10000:
            plot_vertical(ax, -bias)
        else:
            plot_separator(ax, slope, intercept)
        plot_data_with_labels(x, y, ax)
        plt.axis('equal')
        plt.show()

x = np.array([[1, 0], [0,1], [0,-1], [-1,-0], [0, 2], [0, -2], [-2, 0]]) * 1.
y = np.array([-1, -1, -1, 1, 1, 1, 1]) * 1.

t_x = np.array([transform(t) for t in x])
# x = np.array([[-2, -2], [4,-5], [4,-1], [6,-2], [10, -7], [10, 1], [10, 1]]) * 1.
# y = np.array([-1, -1, -1, 1, 1, 1, 1]) * 1.
print(t_x)
# x = np.array([[4.,-1.], [6.,-2.]])
# y = np.array([-1., 1.])

#solve_hard_margin(t_x, y, False)
solve_hard_margin(np.array([[-1,-1], [-1,1], [1,-2], [1, 2]]), np.array([1,1,-1,-1]), True)