#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
using namespace cv;
using namespace std;

// ���� ��� ����������
Mat kernel_h = (Mat_<float>(1, 3) << 1, -1, 0);
Mat kernel_v = (Mat_<float>(3, 1) << 1, -1, 0);
Mat kernel_ht = (Mat_<float>(1, 3) << 0, -1, 1);
Mat kernel_vt = (Mat_<float>(3, 1) << 0, -1, 1);

// �������� � �������������� �����������
Mat gradh(const Mat& x) {
    Mat result;
    filter2D(x, result, CV_32F, kernel_h, Point(-1, -1), 0, BORDER_REFLECT); // ����������� ��������� �������
    return result;
}

// �������� � ������������ �����������
Mat gradv(const Mat& x) {
    Mat result;
    filter2D(x, result, CV_32F, kernel_v, Point(-1, -1), 0, BORDER_REFLECT); // ����������� ��������� �������
    return result;
}

// ������ ��������
vector<Mat> grad2d(const Mat& x) {
    vector<Mat> result;
    result.push_back(gradh(x)); // �������������� ��������
    result.push_back(gradv(x)); // ������������ ��������
    return result;
}

// ���������� �������������� ��������
Mat gradht(const Mat& x) {
    Mat result;
    filter2D(x, result, CV_32F, kernel_ht, Point(-1, -1), 0, BORDER_REFLECT); // ����������� ��������� �������
    return result;
}

// ���������� ������������ ��������
Mat gradvt(const Mat& x) {
    Mat result;
    filter2D(x, result, CV_32F, kernel_vt, Point(-1, -1), 0, BORDER_REFLECT); // ����������� ��������� �������
    return result;
}

// �����������
Mat divergence2d(const vector<Mat>& x) {
    return gradht(x[0]) + gradvt(x[1]);
}

// ��������������� ������������� L1 �����
double hyperbolic(const Mat& z, double eps = 0.01) {
    Mat temp;
    sqrt(z.mul(z) + eps * eps, temp); // ���������� sqrt(z^2 + eps^2)
    return sum(temp)[0]; // ����� ���� ���������
}

// ������� ������� TV-����������
double tv_denoise_objective(const Mat& x, double mu, const Mat& b) {
    vector<Mat> grad_x = grad2d(x); // ���������� ���������
    double term1 = hyperbolic(grad_x[0]) + hyperbolic(grad_x[1]); // ��������������� �������������
    double term2 = 0.5 * mu * norm(x - b, NORM_L2); // L2 �����

    return term1 + term2;
}

// �������� ��������������� �������������
Mat h_grad(const Mat& z, double eps = 0.01) {
    Mat temp;
    sqrt(z.mul(z) + eps * eps, temp); // ���������� sqrt(z^2 + eps^2)
    return z / temp; // z / sqrt(z^2 + eps^2)
}

// �������� ������� ������� TV-����������
Mat tv_denoise_grad(const Mat& x, double mu, const Mat& b) {
    vector<Mat> grad_x = grad2d(x); // ���������� ���������
    Mat grad_h = h_grad(grad_x[0]); // �������� �� �����������
    Mat grad_v = h_grad(grad_x[1]); // �������� �� ���������
    return divergence2d({ grad_h, grad_v }) + mu * (x - b); // ����������� + �������������
}
