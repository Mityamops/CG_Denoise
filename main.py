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
import datetime
np.random.seed(0)
from unifrom import dichotomy_method
from TV_regularization import *
from gradient_methods import *


def TV_reg(image,met,m=1):
    x0 = np.zeros(image.shape)
    f = lambda x: tv_denoise_objective(x, mu=m, b=image)
    grad = lambda x: tv_denoise_grad(x, mu=m, b=image)
    x,_= CG(f, grad, x0,0.001,method=met)
    plt.title("Denoised image")
    plt.imshow(x,cmap='gray')
    plt.show()
    return x
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

def generate_image(noise_level=0.01):
    image = zeros((200, 200))
    image[60:140, 60:140] = 1
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

    #image=cv2.imread('images/Lena.png')

    #image = cv2.imread('images/ips 677 p15 96h 5 bad (2).tif')
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image = cv2.equalizeHist(image)
    #image=cv2.resize(image,(200,150))


    #image=add_noise(image)
    #image=generate_image(1)

    image = zeros((100, 100)).astype(np.float_)
    image = cv2.circle(image, (80, 20), 10, 255, -1)
    image = cv2.rectangle(image, (68, 20), (60, 40), 255, -1)
    image = cv2.circle(image, (40, 80), 12, 255, -1)
    image = cv2.circle(image, (70, 80), 12, 255, -1)
    image = cv2.circle(image, (55, 60), 12, 255, -1)
    pts = np.array([[10, 90], [20, 30], [90, 20], [50, 10]], np.int32)
    cv2.polylines(image, [pts], True, 255, 1)
    plt.title('image')
    plt.imshow(image, cmap='gray')
    plt.show()
    gauss_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.randn(gauss_noise, 128, 20)
    gauss_noise = (gauss_noise * 0.03).astype(np.float_)
    image = cv2.add(255-image, gauss_noise).astype(np.uint8)

    plt.title('Noisy image')
    plt.imshow(image, cmap='gray')
    plt.show()

    plt.title('Noisy image')
    plt.imshow(image, cmap='gray')
    plt.show()



    start = datetime.datetime.now()

    F_R=TV_reg(image,'PR',m=1.2)
    #print(cv2.PSNR(image,F_R))
    finish = datetime.datetime.now()
    print('Время работы: ' + str(finish - start))




    #cv2.imwrite('images/ips 677 p15 96h 5 bad mu=0.08.png',F_R)



    #psnr_BB = cv2.PSNR(image, x)
    #psnr_FR = cv2.PSNR(image, F_R)
    #psnr_FR_x = cv2.PSNR(x, F_R)
    #print(psnr_BB,psnr_FR,psnr_FR_x )
    
