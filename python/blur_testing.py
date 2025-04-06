import cv2
import numpy as np
import math

def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2
    sum_val = 0.0
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            sum_val += kernel[i, j]
    kernel /= sum_val  # Нормализация ядра
    return kernel

def apply_gaussian_blur(image, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
    blurred_image = np.zeros_like(image)

    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            region = padded_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            blurred_image[i-pad_size, j-pad_size] = np.sum(region * kernel)

    return blurred_image


image=np.random.randint(0, 20, (10, 10)).astype(np.uint8)

new_image=cv2.GaussianBlur(image, (3,3),sigmaX=5)


img_out=apply_gaussian_blur(image, kernel_size=3, sigma=5)

print('image:',image,sep='\n')
print('cv2:',new_image,sep='\n')
print('my:',img_out,sep='\n')
