#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>

using namespace cv;
using namespace std;

// Определение ядер для градиента и его сопряженного оператора
extern Mat kernel_h;
extern Mat kernel_v;
extern Mat kernel_ht;
extern Mat kernel_vt;

Mat gradh(const Mat& x);
Mat gradv(const Mat& x);
vector<Mat> grad2d(const Mat& x);
Mat gradht(const Mat& x);
Mat gradvt(const Mat& x);
Mat divergence2d(const vector<Mat>& x);
double hyperbolic(const Mat& z, double eps);
double tv_denoise_objective(const Mat& x, double mu, const Mat& b);
Mat h_grad(const Mat& z, double eps);
Mat tv_denoise_grad(const Mat& x, double mu, const Mat& b);