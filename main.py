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
from TV_regularization import *
from gradient_methods import *

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

def generate_image(noise_level=0.1):
    image = zeros((50, 50))
    image[15:35, 15:35] = 1
    m, n = image.shape
    image = image + noise_level * randn(m, n)
    return image

def add_noise(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    noise = np.zeros(image.shape, np.uint8)
    cv2.randn(noise, 0, 50)

    image = cv2.add(image, noise)
    return image

if __name__ == '__main__':
    # Generate a noisy image

    image=cv2.imread('images/Lena.png')
    image=add_noise(image)

    #image=generate_image(0.2)

    plt.title('Noisy image')
    plt.imshow(image, cmap='gray')
    plt.show()
    '''
    x0 = np.zeros(image.shape)
    f = lambda x: tv_denoise_objective(x, mu=0.05, b=image)
    grad = lambda x: tv_denoise_grad(x, mu=0.05, b=image)
    F_R,_= FR(f, grad, x0,0.01)
    #x, res = grad_descent(f, grad, x0)
    #plt.title("BB method")
    #plt.imshow(x, cmap='gray')
    #plt.show()

    plt.title("FR method")
    plt.imshow(F_R, cmap='gray')
    plt.show()

    #psnr_BB = cv2.PSNR(image, x)
    #psnr_FR = cv2.PSNR(image, F_R)
    #psnr_FR_x = cv2.PSNR(x, F_R)
    #print(psnr_BB,psnr_FR,psnr_FR_x )
    
    '''

    for i in range(1,200):
        x0 = np.zeros(image.shape)
        f = lambda x: tv_denoise_objective(x, mu=i*0.01, b=image)
        grad = lambda x: tv_denoise_grad(x, mu=i*0.01, b=image)
        F_R, _ = FR(f, grad, x0, 0.01)
        plt.title("FR method,mu="+str(i*0.01))
        plt.imshow(F_R, cmap='gray')
        plt.show()
