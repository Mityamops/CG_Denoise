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

# Gradient of the individual components of the objective
kernel_h = [[1,-1,0]]
kernel_v = [[1],[-1],[0]]

def gradh(x):
    """Discrete gradient/difference in horizontal direction"""
    return convolve2d(x,kernel_h, mode='same', boundary='wrap')
def gradv(x):
    """Discrete gradient/difference in vertical direction"""
    return convolve2d(x,kernel_v, mode='same', boundary='wrap')
def grad2d(x):
    """The full gradient operator: compute both x and y differences and return them all.  The x and y
    differences are stacked so that rval[0] is a 2D array of x differences, and rval[1] is the y differences."""
    return np.stack([gradh(x),gradv(x)])


def gradht(x):
    """Adjoint of gradh"""
    kernel_ht = [[0,-1,1]]
    return convolve2d(x,kernel_ht, mode='same', boundary='wrap')
def gradvt(x):
    """Adjoint of gradv"""
    kernel_vt = [[0],[-1],[1]]
    return convolve2d(x,kernel_vt, mode='same', boundary='wrap')
def divergence2d(x):
    "The method is the adjoint of grad2d."
    return gradht(x[0])+gradvt(x[1])


# Using the individual components to create the complete gradient of the objective

def h(z, eps=.01):
    """The hyperbolic approximation to L1"""
    return sum(sqrt(z*z+eps*eps).ravel())
def tv_denoise_objective(x,mu,b):
    return h(grad2d(x)) + 0.5*mu*norm(x-b)**2
def h_grad(z, eps=.01):
    """The gradient of h"""
    return z/sqrt(z*z+eps*eps)
def tv_denoise_grad(x,mu,b):
    """The gradient of the TV objective"""
    return divergence2d(h_grad(grad2d(x))) + mu*(x-b)


def add_salt_and_pepper_noise(image: np.ndarray, salt_prob: float, pepper_prob: float) -> np.ndarray:
    """Добавляет шум типа соль-перец на изображение.

    Args:
        image (np.ndarray): Исходное изображение.
        salt_prob (float): Вероятность появления соли.
        pepper_prob (float): Вероятность появления перца.

    Returns:
        np.ndarray: Зашумленное изображение.
    """
    noisy = np.copy(image)
    total_pixels = image.size

    # Добавление соли
    num_salt = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1]] = 1

    # Добавление перца
    num_pepper = np.ceil(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

if __name__ == '__main__':
    # Generate a noisy image
    #image = zeros((50, 50))
    #image[15:35, 15:35] = 1
    image=cv2.imread('Lena.png')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    m,n=image.shape

    noise = np.zeros(image.shape, np.uint8)
    cv2.randn(noise, 0, 50)

    image = cv2.add(image, noise)

    #image = image + 10 * randn(m, n)
    plt.title('Noisy image')
    plt.imshow(image, cmap='gray')
    plt.show()

    x0 = np.zeros(image.shape)
    f = lambda x: tv_denoise_objective(x, mu=1, b=image)
    grad = lambda x: tv_denoise_grad(x, mu=1, b=image)
    F_R,_= FR(f, grad, x0,0.001)
    x, res = grad_descent(f, grad, x0)
    plt.title("BB method")
    plt.imshow(x, cmap='gray')
    plt.show()

    plt.title("FR method")
    plt.imshow(F_R, cmap='gray')
    plt.show()

    psnr_BB = cv2.PSNR(image, x)
    psnr_FR = cv2.PSNR(image, F_R)
    psnr_FR_x = cv2.PSNR(x, F_R)
    print(psnr_BB,psnr_FR,psnr_FR_x )