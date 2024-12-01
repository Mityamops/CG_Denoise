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





# Пример использования
if __name__ == "__main__":
    # Загрузка изображения
    image_path = 'images/ips F055 p28 72h 11.tif'
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=image.astype(np.uint16)
    # Проверка, что изображение 16-битное
    #if image.dtype != np.uint16:
    #    raise ValueError("Изображение должно быть в формате 16-бит.")

    # Удаление градиента освещения
    processed_image = remove_illumination_gradient(image, degree=3)

    # Отображение результатов

    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.show()

    plt.title('Processed Image')
    plt.imshow(processed_image, cmap='gray')

    #cv2.imwrite('images/processed_image.tif',processed_image)
    plt.show()
