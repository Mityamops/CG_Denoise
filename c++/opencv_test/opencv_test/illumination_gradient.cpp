#include "illumination_gradient.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
using namespace Eigen;

// Функция для полиномиальной аппроксимации методом наименьших квадратов
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

// Основная функция для удаления градиента освещения
Mat removeIlluminationGradient(const Mat& image, int degree, int axis) {
    Mat floatImage;
    image.convertTo(floatImage, CV_32F);

    int rows = floatImage.rows;
    int cols = floatImage.cols;

    // Вычисляем средние интенсивности и диапазоны
    vector<float> avgIntensities;
    vector<float> intensityRanges;
    vector<float> lines;

    if (axis == 0) { // Обработка по строкам
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
    else if (axis == 1) { // Обработка по столбцам
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

    // Преобразуем данные в Eigen::VectorXd
    VectorXd x(lines.size());
    VectorXd yAvg(avgIntensities.size());
    VectorXd yRange(intensityRanges.size());

    for (size_t i = 0; i < lines.size(); ++i) {
        x(i) = lines[i];
        yAvg(i) = avgIntensities[i];
        yRange(i) = intensityRanges[i];
    }

    // Полиномиальная аппроксимация
    VectorXd coeffsAvg = polynomialFit(x, yAvg, degree);
    VectorXd coeffsRange = polynomialFit(x, yRange, degree);

    // Обработка изображения
    Mat processedImage = floatImage.clone();

    if (axis == 0) { // Обработка строк
        for (int i = 0; i < rows; ++i) {
            double rangeVal = 0.0, avgVal = 0.0;
            for (int k = 0; k <= degree; ++k) {
                rangeVal += coeffsRange(k) * pow(i, k);
                avgVal += coeffsAvg(k) * pow(i, k);
            }

            double minVal = avgVal - rangeVal / 2.0;
            double interval = rangeVal / 256.0;

            // Защита от малых интервалов
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
    else if (axis == 1) { // Обработка столбцов
        for (int j = 0; j < cols; ++j) {
            double rangeVal = 0.0, avgVal = 0.0;
            for (int k = 0; k <= degree; ++k) {
                rangeVal += coeffsRange(k) * pow(j, k);
                avgVal += coeffsAvg(k) * pow(j, k);
            }

            double minVal = avgVal - rangeVal / 2.0;
            double interval = rangeVal / 256.0;

            // Защита от малых интервалов
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

    // Преобразуем обратно в 8-битное изображение
    Mat result;
    processedImage.convertTo(result, CV_8U);

    return result;
}

double bilinearInterpolation(const vector<vector<double>>& grid, double x, double y) {
    int nx = grid.size();
    int ny = grid[0].size();

    if (nx == 0 || ny == 0) {
        return 0.0;
    }

    // Ограничение координат внутри границ сетки
    x = std::max(0.0, std::min(x, static_cast<double>(nx - 1)));
    y = std::max(0.0, std::min(y, static_cast<double>(ny - 1)));

    int ix = static_cast<int>(x);
    int iy = static_cast<int>(y);

    double dx = x - ix;
    double dy = y - iy;

    double v11 = grid[iy][ix];
    double v12 = grid[iy][min(ix + 1, nx - 1)];
    double v21 = grid[min(iy + 1, ny - 1)][ix];
    double v22 = grid[min(iy + 1, ny - 1)][min(ix + 1, nx - 1)];

    double v1 = v11 * (1 - dx) + v12 * dx;
    double v2 = v21 * (1 - dx) + v22 * dx;

    return v1 * (1 - dy) + v2 * dy;
}

// Главная функция для удаления градиента освещения
Mat removeIlluminationGradient2D(const Mat& image,int block_size) {
    Mat floatImage;
    image.convertTo(floatImage, CV_32F);

    int height = floatImage.rows;
    int width = floatImage.cols;

    // Параметры блоков
    int blockX = width / block_size;
    int blockY = height / block_size;

    // Расширение границ изображения с помощью зеркального отражения
    Mat paddedImage;
    copyMakeBorder(floatImage, paddedImage, blockY, blockY, blockX, blockX, BORDER_REFLECT);

    height = paddedImage.rows;
    width = paddedImage.cols;

    // Вычисление средних интенсивностей и диапазонов для каждого блока
    vector<vector<double>> avgIntensities(height / blockY, vector<double>(width / blockX, 0.0));
    vector<vector<double>> ranges(height / blockY, vector<double>(width / blockX, 0.0));

    for (int i = 0; i < height; i += blockY) {
        for (int j = 0; j < width; j += blockX) {
            Mat block = paddedImage(Rect(j, i, blockX, blockY));
            Scalar meanVal = mean(block);
            double minVal, maxVal;
            minMaxLoc(block, &minVal, &maxVal);
            avgIntensities[i / blockY][j / blockX] = meanVal[0];
            ranges[i / blockY][j / blockX] = maxVal - minVal;
        }
    }

    // Создание сетки для интерполяции
    vector<double> xGrid(width / blockX);
    vector<double> yGrid(height / blockY);

    for (int i = 0; i < height / blockY; ++i) {
        yGrid[i] = blockY * i + blockY / 2;
    }
    for (int j = 0; j < width / blockX; ++j) {
        xGrid[j] = blockX * j + blockX / 2;
    }

    // Трансформация изображения
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // Преобразование координат пикселей в координаты блоков
            double blockXCoord = static_cast<double>(j) ;
            double blockYCoord = static_cast<double>(i) ;

            // Интерполяция средней интенсивности и диапазона
            double avg = bilinearInterpolation(avgIntensities, blockXCoord, blockYCoord);
            double rangeVal = bilinearInterpolation(ranges, blockXCoord, blockYCoord);

            // Вычисление минимального значения и интервала
            double minVal = avg - rangeVal / 2.0;
            double interval = rangeVal / 256.0;

            // Защита от малых интервалов
            if (interval < 1e-6) {
                interval = 1.0;
            }

            // Корректировка интенсивности
            double intensity = paddedImage.at<float>(i, j);
            double newIntensity = (intensity - minVal) / interval;

            // Использование собственной функции clamp
            auto clamp = [](double value, double minVal, double maxVal) {
                return std::max(minVal, std::min(value, maxVal));
                };

            paddedImage.at<float>(i, j) = clamp(newIntensity, 0.0, 255.0);
        }
    }

    // Удаление расширенных границ
    Mat result = paddedImage(Range(blockY, height - blockY), Range(blockX, width - blockX));

    // Преобразование обратно в 8-битное изображение
    result.convertTo(result, CV_8U);

    return result;
}