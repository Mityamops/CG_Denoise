#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
using namespace cv;
using namespace std;

// Ядра для градиентов
Mat kernel_h = (Mat_<float>(1, 3) << 1, -1, 0);
Mat kernel_v = (Mat_<float>(3, 1) << 1, -1, 0);
Mat kernel_ht = (Mat_<float>(1, 3) << 0, -1, 1);
Mat kernel_vt = (Mat_<float>(3, 1) << 0, -1, 1);

// Градиент в горизонтальном направлении
Mat gradh(const Mat& x) {
    Mat result;
    filter2D(x, result, CV_32F, kernel_h, Point(-1, -1), 0, BORDER_REFLECT); // Циклические граничные условия
    return result;
}

// Градиент в вертикальном направлении
Mat gradv(const Mat& x) {
    Mat result;
    filter2D(x, result, CV_32F, kernel_v, Point(-1, -1), 0, BORDER_REFLECT); // Циклические граничные условия
    return result;
}

// Полный градиент
vector<Mat> grad2d(const Mat& x) {
    vector<Mat> result;
    result.push_back(gradh(x)); // Горизонтальный градиент
    result.push_back(gradv(x)); // Вертикальный градиент
    return result;
}

// Сопряжённый горизонтальный градиент
Mat gradht(const Mat& x) {
    Mat result;
    filter2D(x, result, CV_32F, kernel_ht, Point(-1, -1), 0, BORDER_REFLECT); // Циклические граничные условия
    return result;
}

// Сопряжённый вертикальный градиент
Mat gradvt(const Mat& x) {
    Mat result;
    filter2D(x, result, CV_32F, kernel_vt, Point(-1, -1), 0, BORDER_REFLECT); // Циклические граничные условия
    return result;
}

// Дивергенция
Mat divergence2d(const vector<Mat>& x) {
    return gradht(x[0]) + gradvt(x[1]);
}

// Гиперболическая аппроксимация L1 нормы
double hyperbolic(const Mat& z, double eps = 0.01) {
    Mat temp;
    sqrt(z.mul(z) + eps * eps, temp); // Вычисление sqrt(z^2 + eps^2)
    return sum(temp)[0]; // Сумма всех элементов
}

// Целевая функция TV-денойзинга
double tv_denoise_objective(const Mat& x, double mu, const Mat& b) {
    vector<Mat> grad_x = grad2d(x); // Вычисление градиента
    double term1 = hyperbolic(grad_x[0]) + hyperbolic(grad_x[1]); // Гиперболическая аппроксимация
    double term2 = 0.5 * mu * norm(x - b, NORM_L2); // L2 норма

    return term1 + term2;
}

// Градиент гиперболической аппроксимации
Mat h_grad(const Mat& z, double eps = 0.01) {
    Mat temp;
    sqrt(z.mul(z) + eps * eps, temp); // Вычисление sqrt(z^2 + eps^2)
    return z / temp; // z / sqrt(z^2 + eps^2)
}

// Градиент целевой функции TV-денойзинга
Mat tv_denoise_grad(const Mat& x, double mu, const Mat& b) {
    vector<Mat> grad_x = grad2d(x); // Вычисление градиента
    Mat grad_h = h_grad(grad_x[0]); // Градиент по горизонтали
    Mat grad_v = h_grad(grad_x[1]); // Градиент по вертикали
    return divergence2d({ grad_h, grad_v }) + mu * (x - b); // Дивергенция + регуляризация
}
