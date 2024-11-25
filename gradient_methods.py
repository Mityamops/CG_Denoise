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
from unifrom import dichotomy_method
def estimate_lipschitz(g, x):
    y = rand(*x.shape)
    L = norm(g(x)-g(y))/norm(x-y)
    return L


def grad_descent(f, grad, x0, max_iters=1000, tol=1e-4):
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

        #print(norm(d))
    x = x_k
    return x, res


def grad_descent_bb(f, grad, x0, max_iters=100, tol=1e-4):
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
    x=x_k
    return x, res



def HY(f, grad, x0, e=0.001, print_grad=0, print_dif=0, visualize=0, max_iter=500):
    xcur = np.array(x0)
    n = len(x0)
    k = 0
    dk = grad(x0)
    prevgrad = dk.copy()
    pk = -dk
    res = [np.linalg.norm(dk)]

    while (np.linalg.norm(dk) > e) and (k < max_iter):
        xprev = xcur
        if k % n == 0:
            pk = -dk
        else:
            yk = dk - prevgrad
            numerator = np.trace(dk.T @ yk) - np.trace(dk.T @ dk)
            denominator = np.trace(pk.T @ yk) - np.trace(prevgrad.T @ prevgrad)
            bk = numerator / denominator
            pk = -dk + bk * pk

        alpha = optimize.minimize_scalar(lambda a: f(xcur + a * pk), bounds=(0, 1)).x
        xcur = xcur + alpha * pk
        k += 1

        prevgrad = dk.copy()
        dk = grad(xcur)
        res.append(np.linalg.norm(dk))
        if print_dif == 1:
            print(np.linalg.norm(xcur - xprev))
        if print_grad == 1:
            print(np.linalg.norm(dk))
        if visualize != 0 and k % visualize == 0:
            plt.title(str(k) + " iteration")
            plt.imshow(xcur, cmap='gray')
            plt.show()

    return xcur, res



def CG(f,grad,x0,e=0.001,print_grad=0,print_dif=0,visualize=0,max_iter=500,method='FR'):
    xcur = np.array(x0)
    n = len(x0)
    k = 0
    dk = grad(x0)
    prevgrad = 1
    pk = -dk
    res = [np.linalg.norm(dk)]

    #while k < max_iter:
    while (np.linalg.norm(dk)>e):
        xprev = xcur
        if k % n == 0:
            pk = -dk
        else:
            if method=='FR':
                bk = np.linalg.norm(dk)**2 / np.linalg.norm(prevgrad)**2
            if method=='PR':
                bk = np.linalg.norm(dk.T @ (dk - prevgrad)) / np.linalg.norm(prevgrad.T @ prevgrad)
            if method=='DY':
                bk = np.linalg.norm(dk.T @ dk) / np.linalg.norm(pk.T  @ (dk - prevgrad))
            pk = -dk + bk * pk

        alpha = optimize.minimize_scalar(lambda a: f(xcur + a * pk), bounds=(0, 1)).x
        xcur = xcur + alpha * pk
        k += 1

        prevgrad = dk
        dk = grad(xcur)
        res.append(np.linalg.norm(dk))
        if print_dif==1:
            print(np.linalg.norm(xcur-xprev))
        if print_grad==1:
            print(np.linalg.norm(dk))
        if visualize!=0 and k%visualize==0:
            plt.title(str(k)+"iteration")
            plt.imshow(xcur, cmap='gray')
            plt.show()

    print(k)
    return xcur,res #step10