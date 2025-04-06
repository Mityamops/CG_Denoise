#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
#include <cmath>
#include "TV_regularization.h"
#include "gradient_methods.h"
using namespace cv;
using namespace std;



int main() {
    // Чтение изображения
    string filename = "C:/Users/митя/PycharmProjects/CG_Denoise/python/images/Lena_noise.png"; // Укажите путь к файлу
    Mat image = imread(filename, IMREAD_GRAYSCALE);

    if (image.empty()) {
        cerr << "Error: Could not read image!" << endl;
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

    Mat denoised_image = CG(objective, gradient, x0,"FR");
    cout << objective(denoised_image) << endl;
    // Преобразование результата в формат CV_8U
    denoised_image.convertTo(denoised_image, CV_8U);
    
    imwrite("denoised_image_gs.png", denoised_image);
    return 0;
}