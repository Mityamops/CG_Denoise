import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from scipy.interpolate import RectBivariateSpline
def polynomial_fit(x, y, degree):
    """
    Функция для полиномиальной аппроксимации методом наименьших квадратов.
    :param x: Значения x (номера строк).
    :param y: Значения y (средние интенсивности или диапазоны интенсивностей).
    :param degree: Степень полинома.
    :return: Коэффициенты полинома.
    """
    coeffs = np.polyfit(x, y, degree)

    return np.poly1d(coeffs)



def remove_illumination_gradient(image, degree=3, axis=0):
    """
    Функция для удаления градиента освещения в изображении.
    :param image: Входное изображение в формате 16-бит.
    :param degree: Степень полинома для аппроксимации.
    :param axis: Ось для удаления градиента (0 - строки, 1 - столбцы).
    :return: Обработанное изображение.
    """
    # Преобразование изображения в формат float для удобства вычислений
    image = image.astype(np.float32)

    if axis == 0:
        # Построение наборов значений для аппроксимации по строкам
        lines = np.arange(image.shape[0])
        avg_intensities = np.mean(image, axis=1)
        intensity_ranges = np.max(image, axis=1) - np.min(image, axis=1)
    elif axis == 1:
        # Построение наборов значений для аппроксимации по столбцам
        lines = np.arange(image.shape[1])
        avg_intensities = np.mean(image, axis=0)
        intensity_ranges = np.max(image, axis=0) - np.min(image, axis=0)
    else:
        raise ValueError("axis должен быть 0 или 1")

    # Полиномиальная аппроксимация
    P_avg = polynomial_fit(lines, avg_intensities, degree)
    P_range = polynomial_fit(lines, intensity_ranges, degree)

    # Обработка каждой строки или столбца изображения
    if axis == 0:
        for i in range(image.shape[0]):
            range_val = P_range(i)
            avg_val = P_avg(i)
            min_val = avg_val - range_val / 2
            interval = range_val / 256

            # Преобразование интенсивностей
            image[i] = (image[i] - min_val) / interval
            image[i] = np.clip(image[i], 0, 255)
    elif axis == 1:
        for j in range(image.shape[1]):
            range_val = P_range(j)
            avg_val = P_avg(j)
            min_val = avg_val - range_val / 2
            interval = range_val / 256

            # Преобразование интенсивностей
            image[:, j] = (image[:, j] - min_val) / interval
            image[:, j] = np.clip(image[:, j], 0, 255)

    return image.astype(np.uint8)


def cart2polar(x, y):
    """
    Преобразование из декартовых координат в полярные.
    :param x: Координаты x.
    :param y: Координаты y.
    :return: Угол и радиус.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return phi, rho

def polar2cart(rho, phi):
    """
    Преобразование из полярных координат в декартовы.
    :param rho: Радиус.
    :param phi: Угол.
    :return: Координаты x и y.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def remove_illumination_gradient_radial(image, degree=3):
    """
    Функция для удаления градиента освещения в изображении по радиальным линиям.
    :param image: Входное изображение в формате 16-бит.
    :param degree: Степень полинома для аппроксимации.
    :return: Обработанное изображение.
    """
    # Преобразование изображения в формат float для удобства вычислений
    image = image.astype(np.float32)
    height, width = image.shape
    center_x, center_y = width // 2, height // 2

    # Преобразование в полярные координаты
    y, x = np.indices((height, width))
    phi, rho = cart2polar(x - center_x, y - center_y)
    diameter = 2 * rho  # Преобразование радиуса в диаметр

    # Списки для хранения данных для аппроксимации
    all_diameters = []
    all_avg_intensities = []
    all_intensity_ranges = []

    # Маска для уже использованных пикселей
    used_mask = np.zeros_like(image, dtype=bool)

    # Обработка каждого угла
    step = 10
    for angle in range(-180, 0, step):  # Увеличиваем шаг для ускорения
        # Выбор пикселей, соответствующих текущему углу и его симметричному углу
        angle_rad = np.deg2rad(angle)
        const_angle = step / 2
        mask = (phi >= angle_rad - np.deg2rad(const_angle)) & (phi < angle_rad + np.deg2rad(const_angle))
        mask |= (phi >= (angle_rad + np.pi - np.deg2rad(const_angle))) & (phi < (angle_rad + np.pi + np.deg2rad(const_angle)))

        # Добавление пикселей из центра
        center_radius = 5  # Радиус вокруг центра, который будет включен
        center_mask = (x - center_x)**2 + (y - center_y)**2 <= center_radius**2
        mask |= center_mask

        # Исключение уже использованных пикселей
        mask &= ~used_mask

        line_pixels = image[mask]
        line_diameter = diameter[mask]

        if len(line_pixels) == 0:
            continue

        # Средние интенсивности и диапазоны интенсивностей
        avg_intensities = np.array([np.mean(line_pixels[line_diameter == d]) for d in np.unique(line_diameter)])
        intensity_ranges = np.array([np.max(line_pixels[line_diameter == d]) - np.min(line_pixels[line_diameter == d]) for d in np.unique(line_diameter)])
        unique_diameter = np.unique(line_diameter)

        # Добавление данных для аппроксимации
        all_diameters.extend(unique_diameter)
        all_avg_intensities.extend(avg_intensities)
        all_intensity_ranges.extend(intensity_ranges)

        # Обновление маски для уже использованных пикселей
        used_mask |= mask

    # Преобразование списков в массивы
    all_diameters = np.array(all_diameters)
    all_avg_intensities = np.array(all_avg_intensities)
    all_intensity_ranges = np.array(all_intensity_ranges)

    # Полиномиальная аппроксимация
    P_avg = polynomial_fit(all_diameters, all_avg_intensities, degree)
    P_range = polynomial_fit(all_diameters, all_intensity_ranges, degree)

    # Обработка каждого пикселя на радиальной линии
    for i in range(height):
        for j in range(width):
            range_val = P_range(diameter[i, j])
            avg_val = P_avg(diameter[i, j])
            min_val = avg_val - range_val / 2
            interval = range_val / 256

            # Преобразование интенсивностей
            image[i, j] = (image[i, j] - min_val) / interval
            image[i, j] = np.clip(image[i, j], 0, 255)

    return image.astype(np.uint8)



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

def remove_illumination_gradient_lowpass(image, kernel_size=51):



    image = image.astype(np.float32)

    #low_pass_image = cv2.GaussianBlur(image, kernel_size,sigmaX=0)#если sigmaX=0: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    low_pass_image=apply_gaussian_blur(image, kernel_size, sigma=0.3*((kernel_size-1)*0.5 - 1) + 0.8)
    corrected_image = image - low_pass_image
    corrected_image = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX)

    return corrected_image.astype(np.uint8)

def build_graph(image, degree=3):
    image = image.astype(np.float32)

    # Построение наборов значений для аппроксимации
    lines = np.arange(image.shape[0])
    avg_intensities = np.mean(image, axis=1)
    intensity_ranges = np.max(image, axis=1) - np.min(image, axis=1)

    # Построение графика исходных данных
    plt.figure(figsize=(12, 6))
    plt.plot(lines, intensity_ranges, label='Intensity Range (I_max - I_min)')
    plt.plot(lines, avg_intensities, label='Average Intensity (I_avg)')
    plt.xlabel('Line Number')
    plt.ylabel('Intensity')
    plt.title('Intensity Differences vs. Line Number')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Построение графиков аппроксимированных данных и ошибок
    for i in range(1, degree + 1):
        # Полиномиальная аппроксимация
        P_avg = polynomial_fit(lines, avg_intensities, i)
        P_range = polynomial_fit(lines, intensity_ranges, i)

        # Рассчет ошибок аппроксимации
        avg_errors = avg_intensities - P_avg(lines)
        range_errors = intensity_ranges - P_range(lines)

        # Построение графика аппроксимированных данных
        plt.figure(figsize=(12, 6))
        plt.plot(lines, P_range(lines), label='Approximated Intensity Range', linestyle='--')
        plt.plot(lines, P_avg(lines), label='Approximated Average Intensity', linestyle='--')
        plt.xlabel('Line Number')
        plt.ylabel('Intensity')
        plt.title(f'Approximated Intensity Differences vs. Line Number (Degree {i})')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Построение графика ошибок аппроксимации
        plt.figure(figsize=(12, 6))
        plt.plot(lines, range_errors, label='Intensity Range Error', linestyle='--')
        plt.plot(lines, avg_errors, label='Average Intensity Error', linestyle='--')
        plt.xlabel('Line Number')
        plt.ylabel('Error')
        plt.title(f'Approximation Errors vs. Line Number (Degree {i})')
        plt.legend()
        plt.grid(True)
        plt.show()



def remove_illumination_gradient_2d(img):
    image = img.copy()

    height, width = image.shape
    block_x = 80  # Размер блока по оси X
    block_y = 40  # Размер блока по оси Y

    # Вычисление средних интенсивностей и диапазонов для каждого блока
    avg_intensities = np.zeros((height // block_y + 1, width // block_x + 1))
    ranges = np.zeros((height // block_y + 1, width // block_x + 1))

    for i in range(0, height, block_y):
        for j in range(0, width, block_x):
            block = image[i:i + block_y, j:j + block_x]
            if block.size > 0:  # Проверка, что блок не пустой
                avg_intensities[i // block_y, j // block_x] = np.mean(block)
                ranges[i // block_y, j // block_x] = np.max(block) - np.min(block)

    # Создание сетки для бивариантного сплайна
    x = np.arange(0, width, block_x)
    y = np.arange(0, height, block_y)

    # Построение бивариантного сплайна для средних интенсивностей и диапазонов
    spline_avg = RectBivariateSpline(y, x, avg_intensities[:len(y), :len(x)],s=0)
    spline_range = RectBivariateSpline(y, x, ranges[:len(y), :len(x)],s=0)

    # Трансформация изображения
    for i in range(height):
        for j in range(width):
            avg = spline_avg(i, j)[0][0]
            range_val = spline_range(i, j)[0][0]
            min_val = avg - range_val / 2
            interval = range_val / 256
            intensity = image[i, j]
            new_intensity = (intensity - min_val) / interval
            image[i, j] = np.clip(new_intensity, 0, 255)

    return image


def plot_histogram(image):
    plt.figure(figsize=(15, 10))
    plt.hist(image.ravel(), bins=256, color='purple', alpha=0.7)
    plt.title('Histogram of the Whole Image')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

# Пример использования
if __name__ == "__main__":
    # Загрузка изображения
    image_path = 'images/ips 677 p15 96h 5 bad (2).tif'
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image = cv2.resize(image, (500, 400))


    #processed_image = cv2.equalizeHist(image)
    # Удаление градиента освещения

    image_orig = cv2.equalizeHist(image.astype(np.uint8))

    processed_image = remove_illumination_gradient_2d(image_orig)


    image_new = cv2.equalizeHist(processed_image.astype(np.uint8))
    # Отображение результатов
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.show()

    plt.title('orig Image hist')
    plt.imshow(image_orig, cmap='gray')
    plt.show()

    plt.title('proccesed Image hist')
    plt.imshow(image_new, cmap='gray')
    plt.show()

    plt.title('Processed Image')
    plt.imshow(processed_image, cmap='gray')

    plot_histogram(processed_image)
    plot_histogram(image_orig)

    cv2.imwrite('images/report_il_grad_2d.png',processed_image.astype(np.uint8))

    cv2.imwrite('images/report_il_grad_hist_2d.png',image_orig.astype(np.uint8))
    plt.show()
