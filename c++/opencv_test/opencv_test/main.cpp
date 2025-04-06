#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
#include <cmath>
#include "TV_regularization.h"
#include "gradient_methods.h"
using namespace cv;
using namespace std;

int main(int argc, char* argv[]) { 
    if (argc != 3) { // Проверка количества аргументов
        cerr << "Usage: " << argv[0] << " input_image output_image" << endl;
        return -1;
    }

    string input_filename = argv[1]; // Используем первый аргумент как входной файл
    string output_filename = argv[2]; // Второй аргумент как выходной файл
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

    Mat denoised_image = CG(objective, gradient, x0, "FR");
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