import cv2
import random
import numpy as np
from numpy import sqrt, sum, abs, max, maximum, logspace, exp, log, log10, zeros
from numpy.linalg import norm
from numpy.random import randn, rand
import urllib
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import optimize
np.random.seed(0)

def estimate_lipschitz(g, x):
    y = rand(*x.shape)
    L = norm(g(x)-g(y))/norm(x-y)
    return L


def grad_descent(f, grad, x0, max_iters=10000, tol=1e-4):
    x_k = x0
    L = estimate_lipschitz(grad, x_k)
    step_size = 10 / L
    res = []
    res.append(norm(grad(x_k)))

    for i in range(max_iters):
        d = -grad(x_k)

        while (f(x_k + step_size * d) >= (f(x_k) + 0.1 * (np.sum((step_size * d) * (-d))))):
            step_size = step_size / 2
        x_k1 = x_k + step_size * d
        res.append(norm((-d)))

        if res[-1] < tol * res[0]:
            x = x_k
            break
        x_k = x_k1

    return x, res


def grad_descent_bb(f, grad, x0, max_iters=10000, tol=1e-4):
    x_k = x0
    L = estimate_lipschitz(grad, x0)
    step_size = L / 100
    x_k1 = x_k - step_size * grad(x_k)
    res = []
    res.append(norm(grad(x_k)))
    d = -grad(x_k)
    for i in range(max_iters):

        step_size = (np.sum((x_k1 - x_k) * (x_k1 - x_k))) / (np.sum((x_k1 - x_k) * (grad(x_k1) + d)))

        while (f(x_k + step_size * d) >= (f(x_k) + 0.1 * step_size * (np.sum((d) * (-d))))):
            step_size = step_size / 2

        x_k = x_k1
        d = -grad(x_k)
        res.append(norm(d))
        x_k1 = x_k + step_size * d

        if (res[-1] < tol * res[0]):
            x = x_k
            break

    return x, res

def FR(f,grad,x0,e=0.001):
    xcur = np.array(x0)
    n = len(x0)
    k = 0 # step1
    dk=grad(x0)
    prevgrad = 1
    pk = -1*dk
    res=[]
    res.append(norm(grad(x0)))
    while (np.linalg.norm(dk)>e): # step3
        if (k%n==0): # step4
            pk = -1*dk
        else:
            bk = (np.linalg.norm(dk)**2)/(np.linalg.norm(prevgrad)**2) # step5
            prevpk = pk
            pk = -1*dk + bk*prevpk # step6
        a = (optimize.minimize_scalar(lambda x: f(xcur+pk*x), bounds=(0,10)).x)
        xcur = xcur + a*pk #step8
        k=k+1 #step8
        prevgrad=dk
        dk=grad(xcur)
        res.append(norm(dk))
    return xcur,res #step10
