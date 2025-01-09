import numpy as np
import cv2
import matplotlib.pyplot as plt


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


def remove_illumination_gradient(image, degree=3):
    """
    Функция для удаления градиента освещения в изображении.
    :param image: Входное изображение в формате 16-бит.
    :param degree: Степень полинома для аппроксимации.
    :return: Обработанное изображение.
    """
    # Преобразование изображения в формат float для удобства вычислений
    image = image.astype(np.float32)

    # Построение наборов значений для аппроксимации
    lines = np.arange(image.shape[0])
    avg_intensities = np.mean(image, axis=1)
    intensity_ranges = np.max(image, axis=1) - np.min(image, axis=1)

    # Полиномиальная аппроксимация
    P_avg = polynomial_fit(lines, avg_intensities, degree)
    P_range = polynomial_fit(lines, intensity_ranges, degree)

    # Обработка каждой строки изображения
    for i in range(image.shape[0]):
        range_val = P_range(i)
        avg_val = P_avg(i)
        min_val = avg_val - range_val / 2
        interval = range_val / 256

        # Преобразование интенсивностей
        image[i] = (image[i] - min_val) / interval
        image[i] = np.clip(image[i], 0, 255)

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

    # Создание массива для хранения обработанного изображения
    processed_image = np.zeros_like(image)
    processed_mask = np.zeros_like(image, dtype=bool)  # Маска для отслеживания обработанных пикселей

    # Обработка каждого угла
    for angle in range(-180, 6, 6):  # Увеличиваем шаг для ускорения
        # Выбор пикселей, соответствующих текущему углу и его симметричному углу
        angle_rad = np.deg2rad(angle)
        const_angle=3
        mask = (phi >= angle_rad - np.deg2rad(const_angle)) & (phi < angle_rad + np.deg2rad(const_angle))
        mask |= (phi >= (angle_rad + np.pi - np.deg2rad(const_angle))) & (phi < (angle_rad + np.pi + np.deg2rad(const_angle)))
        mask &= ~processed_mask  # Исключаем уже обработанные пиксели

        line_pixels = image[mask]
        line_diameter = diameter[mask]

        if len(line_pixels) == 0:
            continue

        # Полиномиальная аппроксимация
        avg_intensities = np.array([np.mean(line_pixels[line_diameter == d]) for d in np.unique(line_diameter)])
        intensity_ranges = np.array([np.max(line_pixels[line_diameter == d]) - np.min(line_pixels[line_diameter == d]) for d in np.unique(line_diameter)])
        unique_diameter = np.unique(line_diameter)
        P_avg = polynomial_fit(unique_diameter, avg_intensities, degree)
        P_range = polynomial_fit(unique_diameter, intensity_ranges, degree)

        # Обработка каждого пикселя на радиальной линии
        for i in range(len(line_pixels)):
            range_val = P_range(line_diameter[i])
            avg_val = P_avg(line_diameter[i])
            min_val = avg_val - range_val / 2
            interval = range_val / 256

            # Преобразование интенсивностей
            line_pixels[i] = (line_pixels[i] - min_val) / interval
            line_pixels[i] = np.clip(line_pixels[i], 0, 255)

        # Обновление обработанного изображения
        processed_image[mask] = line_pixels
        processed_mask[mask] = True  # Помечаем обработанные пиксели

    return processed_image.astype(np.uint8)

def remove_illumination_gradient_lowpass(image, kernel_size=(51, 51)):



    image = image.astype(np.float32)

    low_pass_image = cv2.GaussianBlur(image, kernel_size,sigmaX=0)#если sigmaX=0: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
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


# Пример использования
if __name__ == "__main__":
    # Загрузка изображения
    image_path = 'images/ips F055 p28 72h 11.tif'
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (500, 400))


    #processed_image = cv2.equalizeHist(image)
    # Удаление градиента освещения

    #image_new=cv2.equalizeHist(image.astype(np.uint8))
    processed_image = remove_illumination_gradient_lowpass(image)
    #build_graph(image)

    # Отображение результатов
    #print(image.astype(np.uint8))
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.show()



    plt.title('Processed Image')
    plt.imshow(processed_image, cmap='gray')

    cv2.imwrite('images/processed_image_grad_rad.tif',processed_image)
    plt.show()
