#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
#include <cmath>
#include <algorithm>
#include "gradient_methods.h"
using namespace cv;
using namespace std;

// L2 норма вектора
double norm_2(const Mat& v) {
    return sqrt(v.dot(v));
}

// Метод золотого сечения для нахождения оптимального шага
double golden_section_search(
    const function<double(double)>& func,
    double a, double b, double tol 
) {
    const double phi = (sqrt(5) - 1) / 2; // Золотое сечение
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);

    while (abs(c - d) > tol) {
        if (func(c) < func(d)) {
            b = d;
        }
        else {
            a = c;
        }
        c = b - phi * (b - a);
        d = a + phi * (b - a);
    }

    return (a + b) / 2.0;
}



double brent_search(
    const std::function<double(double)>& func,
    double a, double b, double tol = 1e-3
) {
    const double golden_ratio = (std::sqrt(5.0) - 1.0) / 2.0; // Золотое сечение
    const double eps = 1e-10; // Защита от деления на ноль

    double x = a + golden_ratio * (b - a); // Первое пробное значение
    double w = x; // Точка с минимальным значением функции
    double v = w; // Предыдущая точка
    double fx = func(x); // Значение функции в x
    double fw = fx; // Минимальное значение функции
    double fv = fw;

    double d = b - a; // Расстояние между a и b
    double e = d; // Предыдущее расстояние

    while (std::abs(b - a) > tol) {
        double midpoint = (a + b) / 2.0;
        double tolerance = tol * std::abs(x) + eps;

        // Проверка условия остановки
        if (std::abs(x - midpoint) <= tolerance) {
            break;
        }

        double p = 0.0, q = 0.0, r = 0.0;
        double u = 0.0;

        if (std::abs(e) > tolerance) {
            // Параболическая интерполяция
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);

            if (q > 0.0) {
                p = -p;
            }
            q = std::abs(q);

            double e_temp = e;
            e = d;

            // Принятие параболической интерполяции
            if (std::abs(p) < std::abs(0.5 * q * e_temp) && p > q * (a - x) && p < q * (b - x)) {
                d = p / q;
                u = x + d;

                // Защита от выхода за границы
                if ((u - a) < tolerance || (b - u) < tolerance) {
                    d = (x < midpoint) ? tolerance : -tolerance;
                }
            }
            else {
                // Использование золотого сечения
                e = (x < midpoint) ? b - x : a - x;
                d = golden_ratio * e;
            }
        }
        else {
            // Использование золотого сечения
            e = (x < midpoint) ? b - x : a - x;
            d = golden_ratio * e;
        }

        // Выбор нового значения x
        u = (std::abs(d) >= tolerance) ? x + d : x + ((d > 0.0) ? tolerance : -tolerance);
        double fu = func(u);

        // Обновление границ
        if (fu <= fx) {
            if (u >= x) {
                a = x;
            }
            else {
                b = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        }
        else {
            if (u < x) {
                a = u;
            }
            else {
                b = u;
            }
            if (fu <= fw || w == x) {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            }
            else if (fu <= fv || v == x || v == w) {
                v = u;
                fv = fu;
            }
        }
    }

    return x;
}

Mat CG(
    const function<double(const Mat&)>& f,
    const function<Mat(const Mat&)>& grad,
    const Mat& x0,
    const string& method ,
    int max_iters,
    double tol
) {
    Mat xcur = x0.clone();
    Mat g_k = grad(x0); // Начальный градиент
    Mat pk = -g_k;      // Начальное направление
    Mat prevgrad = g_k.clone();
    Mat xprev;

    double step_size = 1.0;
    const double min_step_size = 1e-10;
    double f_prev = f(xcur);
    vector<double> residuals = { norm_2(g_k) };

    for (int k = 0; k < max_iters; ++k) {

        //cout << (f(xcur)) << endl;
        // Обновление направления поиска
        if (k % (x0.total()) == 0) { // Для методов с периодическим обнуление
            pk = -g_k;
        }
        else {
            Mat g_k_prev = prevgrad;
            Mat g_k_curr = grad(xcur);

            if (method == "FR") {
                double beta = g_k_curr.dot(g_k_curr) / (g_k_prev.dot(g_k_prev) + 1e-10);
                pk = -g_k_curr + beta * pk;
            }
            else if (method == "PR") {
                Mat yk = g_k_curr - g_k_prev;
                double beta = yk.dot(g_k_curr) / (g_k_prev.dot(g_k_prev) + 1e-10);
                pk = -g_k_curr + beta * pk;
            }
            else if (method == "DY") {
                double numerator = g_k_curr.dot(g_k_curr);
                Mat yk = g_k_curr - g_k_prev;
                double denominator = pk.dot(yk) + 1e-10;
                double beta = numerator / denominator;
                pk = -g_k_curr + beta * pk;
            }
            else if (method == "BKY" || method == "BKS" || method == "BKG") {

                Mat sk = step_size * pk;
                Mat yk = grad(xcur + sk) - grad(xcur);

                double f_prev_val = f(xcur);
                double f_curr_val = f(xcur + sk);

                if (method == "BKY") {
                    double numerator = (f_curr_val - f_prev_val - 0.5 * sk.dot(yk))
                        + yk.dot(g_k_curr) - sk.dot(g_k_prev); // g_k_prev = prevgrad
                    double denominator = sk.dot(yk) + 1e-10;
                    double beta = numerator / denominator;
                    pk = -g_k_curr + beta * sk;
                }
                else if (method == "BKS") {
                    double numerator = (f_curr_val - f_prev_val + 0.5 * sk.dot(g_k_prev))
                        + yk.dot(g_k_curr) - sk.dot(g_k_prev);
                    double denominator = sk.dot(yk) + 1e-10;
                    double beta = numerator / denominator;
                    pk = -g_k_curr + beta * sk;
                }
                else if (method == "BKG") {
                    double numerator = (f_curr_val - f_prev_val - 0.5 * step_size * g_k_prev.dot(g_k_prev))
                        + yk.dot(g_k_curr) - sk.dot(g_k_prev);
                    double denominator = sk.dot(yk) + 1e-10;
                    double beta = numerator / denominator;
                    pk = -g_k_curr + beta * sk;
                }
            }
            else {
                cerr << "Unknown method: " << method << endl;
                return xcur;
            }
        }

        // Линейный поиск
        auto line_func = [&](double a) { return f(xcur + a * pk); };
        step_size = golden_section_search(line_func, 0.0, 1000.0);

        if (step_size < min_step_size) {
            cerr << "Step size too small" << endl;
            return xcur;
        }

        xprev = xcur.clone();
        xcur = xcur + step_size * pk;

        // Обновление градиента и условий
        Mat g_k_new = grad(xcur);
        residuals.push_back(norm_2(g_k_new));
        
        if (k > 2 && (fabs(f(xcur) - f_prev) / (fabs(f(xcur)) + 1e-10) <= tol)) {
            cout << "Converged in " << k + 1 << " iterations" << endl;
            return xcur;
        }

        prevgrad = g_k;
        g_k = g_k_new;
        f_prev = f(xcur);
    }

    cout << "Reached max iterations" << endl;
    return xcur;
}
