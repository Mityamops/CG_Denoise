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
from illumination_gradient import remove_illumination_gradient


def TV_reg(image,met='FR',m=1):
    x0 = np.zeros(image.shape)
    f = lambda x: tv_denoise_objective(x, mu=m, b=image)
    grad = lambda x: tv_denoise_grad(x, mu=m, b=image)
    x,_= CG(f, grad, x0,method=met,e=1e-4)
    print(f(x))
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
    image = np.zeros((200, 200), dtype=np.uint8)
    image[60:140, 60:140] = 255.0

    # Добавляем шум
    m, n = image.shape
    noise = np.random.randn(m, n) * noise_level
    image = image + noise

    cv2.imwrite('images/square.png',image)
    return image
def add_noise(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    noise = np.zeros(image.shape, np.uint8)
    cv2.randn(noise, 0, 50)

    image = cv2.add(image, noise)
    return image





if __name__ == '__main__':
    # Generate a noisy image

    image=cv2.imread('images/Lena_noise.png')

    #image = cv2.imread(r'C://Users/митя//PycharmProjects//CG_Denoise//images//Lena.png')
    #image = add_noise(image)

    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #image=cv2.resize(image,(400,300))



    #image=generate_image(10)

    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image = cv2.equalizeHist(image)


    

    plt.title('Noisy image')
    plt.imshow(image, cmap='gray')
    plt.show()



    start = datetime.datetime.now()

    F_R=TV_reg(image,'FR',m=0.05)
    #print(cv2.PSNR(image,F_R))
    finish = datetime.datetime.now()
    print('Время работы: ' + str(finish - start))

    plt.title("Denoised image")
    plt.imshow(F_R, cmap='gray')
    plt.show()

    print(cv2.PSNR(image,F_R.astype(np.uint8)))
