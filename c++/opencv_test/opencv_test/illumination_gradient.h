#ifndef ILLUMINATION_GRADIENT_H
#define ILLUMINATION_GRADIENT_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

cv::Mat removeIlluminationGradient(const cv::Mat& image, int degree = 3, int axis = 0);

#endif // ILLUMINATION_GRADIENT_H