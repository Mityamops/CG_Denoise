#include "illumination_gradient.h"
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace Eigen;

// ������� ��� �������������� ������������� ������� ���������� ���������
VectorXd polynomialFit(const VectorXd& x, const VectorXd& y, int degree) {
    int n = x.size();
    MatrixXd A(n, degree + 1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= degree; ++j) {
            A(i, j) = pow(x(i), j);
        }
    }
    VectorXd coeffs = A.householderQr().solve(y);
    return coeffs;
}

// �������� ������� ��� �������� ��������� ���������
Mat removeIlluminationGradient(const Mat& image, int degree, int axis) {
    Mat floatImage;
    image.convertTo(floatImage, CV_32F);

    int rows = floatImage.rows;
    int cols = floatImage.cols;

    // ��������� ������� ������������� � ���������
    vector<float> avgIntensities;
    vector<float> intensityRanges;
    vector<float> lines;

    if (axis == 0) { // ��������� �� �������
        for (int i = 0; i < rows; ++i) {
            Mat row = floatImage.row(i);
            Scalar meanVal = mean(row);
            double minVal, maxVal;
            minMaxLoc(row, &minVal, &maxVal);
            avgIntensities.push_back(meanVal[0]);
            intensityRanges.push_back(maxVal - minVal);
            lines.push_back(i);
        }
    }
    else if (axis == 1) { // ��������� �� ��������
        for (int j = 0; j < cols; ++j) {
            Mat col = floatImage.col(j);
            Scalar meanVal = mean(col);
            double minVal, maxVal;
            minMaxLoc(col, &minVal, &maxVal);
            avgIntensities.push_back(meanVal[0]);
            intensityRanges.push_back(maxVal - minVal);
            lines.push_back(j);
        }
    }
    else {
        throw invalid_argument("Axis must be 0 or 1");
    }

    // ����������� ������ � Eigen::VectorXd
    VectorXd x(lines.size());
    VectorXd yAvg(avgIntensities.size());
    VectorXd yRange(intensityRanges.size());

    for (size_t i = 0; i < lines.size(); ++i) {
        x(i) = lines[i];
        yAvg(i) = avgIntensities[i];
        yRange(i) = intensityRanges[i];
    }

    // �������������� �������������
    VectorXd coeffsAvg = polynomialFit(x, yAvg, degree);
    VectorXd coeffsRange = polynomialFit(x, yRange, degree);

    // ��������� �����������
    Mat processedImage = floatImage.clone();

    if (axis == 0) { // ��������� �����
        for (int i = 0; i < rows; ++i) {
            double rangeVal = 0.0, avgVal = 0.0;
            for (int k = 0; k <= degree; ++k) {
                rangeVal += coeffsRange(k) * pow(i, k);
                avgVal += coeffsAvg(k) * pow(i, k);
            }

            double minVal = avgVal - rangeVal / 2.0;
            double interval = rangeVal / 256.0;

            // ������ �� ����� ����������
            if (interval < 1e-6) {
                interval = 1.0;
            }

            for (int j = 0; j < cols; ++j) {
                float val = floatImage.at<float>(i, j);
                val = (val - minVal) / interval;
                val = max(0.0f, min(255.0f, val));
                processedImage.at<float>(i, j) = val;
            }
        }
    }
    else if (axis == 1) { // ��������� ��������
        for (int j = 0; j < cols; ++j) {
            double rangeVal = 0.0, avgVal = 0.0;
            for (int k = 0; k <= degree; ++k) {
                rangeVal += coeffsRange(k) * pow(j, k);
                avgVal += coeffsAvg(k) * pow(j, k);
            }

            double minVal = avgVal - rangeVal / 2.0;
            double interval = rangeVal / 256.0;

            // ������ �� ����� ����������
            if (interval < 1e-6) {
                interval = 1.0;
            }

            for (int i = 0; i < rows; ++i) {
                float val = floatImage.at<float>(i, j);
                val = (val - minVal) / interval;
                val = max(0.0f, min(255.0f, val));
                processedImage.at<float>(i, j) = val;
            }
        }
    }

    // ����������� ������� � 8-������ �����������
    Mat result;
    processedImage.convertTo(result, CV_8U);

    return result;
}