#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
#include <cmath>
#include "TV_regularization.h"
#include "gradient_methods.h"
#include "illumination_gradient.h"
using namespace cv;
using namespace std;

int denoise(int argc, char* argv[]) {
    if (argc != 6) { // Теперь ожидаем 6 аргументов: имя программы, входное изображение, выходное изображение, max_iters, tol, method
        cerr << "Usage: " << argv[0] << " input_image output_image max_iters tol method" << endl;
        cerr << "Available methods: FR (Fletcher-Reeves), PR (Polak-Ribiere)" << endl;
        return -1;
    }

    string input_filename = argv[1]; // Входной файл
    string output_filename = argv[2]; // Выходной файл

    // Преобразуем аргументы командной строки в числа
    int max_iters = atoi(argv[3]); // Максимальное число итераций
    double tol = atof(argv[4]); // Точность
    string method = argv[5]; // Метод оптимизации

    

    // Чтение изображения
    Mat image = imread(input_filename, IMREAD_GRAYSCALE);

    if (image.empty()) {
        cerr << "Error: Could not read input image!" << endl;
        return -1;
    }

    // Преобразование в тип CV_32F
    image.convertTo(image, CV_32F);

    // Инициализация начального приближения
    Mat x0 = Mat::zeros(image.size(), CV_32F);

    // Параметр регуляризации
    double mu = 0.05;

    // Лямбда-функции для целевой функции и её градиента
    auto objective = [&](const Mat& x) -> double {
        return tv_denoise_objective(x, mu, image);
        };

    auto gradient = [&](const Mat& x) -> Mat {
        return tv_denoise_grad(x, mu, image);
        };

    // Вызов метода CG с параметрами max_iters, tol и method
    Mat denoised_image = CG(objective, gradient, x0, method, max_iters, tol);
    cout << "Objective function value: " << objective(denoised_image) << endl;

    // Преобразование результата в формат CV_8U
    denoised_image.convertTo(denoised_image, CV_8U);

    // Сохранение результата в указанный выходной файл
    if (!imwrite(output_filename, denoised_image)) {
        cerr << "Error: Could not save output image!" << endl;
        return -1;
    }

    cout << "Denoised image saved to " << output_filename << endl;
    return 0;
}

int illum_grad() {
    // Загрузка изображения
    string imagePath = "C:/Users/митя/PycharmProjects/CG_Denoise/python/images/ipc 677 p16 MG (R) 12 bad.tif";
    Mat image = imread(imagePath, IMREAD_UNCHANGED);

    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    // Преобразуем в градации серого, если изображение цветное
    Mat grayImage;
    if (image.channels() > 1) {
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
        cout << "Image converted to grayscale." << endl;
    }
    else {
        grayImage = image;
        cout << "Image is already in grayscale." << endl;
    }

    // Нормализация изображения, если оно 16-битное
    Mat normalizedImage;
    if (grayImage.type() == CV_16U) {
        normalize(grayImage, normalizedImage, 0, 255, NORM_MINMAX, CV_8U);
    }
    else {
        grayImage.convertTo(normalizedImage, CV_8U);
    }

    // Удаление градиента освещения
    Mat processedImage = removeIlluminationGradient(normalizedImage);

    // Сохранение и отображение результата
    imwrite("inverted.png", processedImage);

    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", normalizedImage);

    namedWindow("Processed Image", WINDOW_NORMAL);
    imshow("Processed Image", processedImage);

    waitKey(0);
    return 0;
}

int main(int argc, char* argv[]) { 
    int a = denoise(argc, argv);
    //int a = illum_grad();
    return 0;
}

