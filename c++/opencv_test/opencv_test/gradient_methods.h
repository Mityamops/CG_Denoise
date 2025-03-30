#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
#include <cmath>
using namespace cv;
using namespace std;

double norm_2(const Mat& v);
double golden_section_search(
    const function<double(double)>& func,
    double a, double b, double tol = 1e-3
);
Mat CG(
    const function<double(const Mat&)>& f,
    const function<Mat(const Mat&)>& grad,
    const Mat& x0,
    const string& method = "FR",
    int max_iters = 10000,
    double tol = 1e-4
);