#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
#include <cmath>
#include "TV_regularization.h"
#include "gradient_methods.h"
using namespace cv;
using namespace std;



int main() {
    // ������ �����������
    string filename = "C:/Users/����/PycharmProjects/CG_Denoise/python/images/Lena_noise.png"; // ������� ���� � �����
    Mat image = imread(filename, IMREAD_GRAYSCALE);

    if (image.empty()) {
        cerr << "Error: Could not read image!" << endl;
        return -1;
    }

    // �������������� � ��� CV_32F
    image.convertTo(image, CV_32F);

    // ������������� ���������� �����������
    Mat x0 = Mat::zeros(image.size(), CV_32F);

    // �������� �������������
    double mu = 0.05;

    // ������-������� ��� ������� ������� � � ���������
    auto objective = [&](const Mat& x) -> double {
        return tv_denoise_objective(x, mu, image);
        };

    auto gradient = [&](const Mat& x) -> Mat {
        return tv_denoise_grad(x, mu, image);
        };

    Mat denoised_image = CG(objective, gradient, x0,"FR");
    cout << objective(denoised_image) << endl;
    // �������������� ���������� � ������ CV_8U
    denoised_image.convertTo(denoised_image, CV_8U);
    
    // ���������� ����������
    imwrite("denoised_image_gs.png", denoised_image);

    return 0;
}