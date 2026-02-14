#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
#include <cmath>
#include <algorithm>
#include "gradient_methods.h"
using namespace cv;
using namespace std;


double norm_2(const Mat& v) {
    return sqrt(v.dot(v));
}

// Метод золотого сечения для нахождения оптимального шага
double golden_section_search(
    const function<double(double)>& func,
    double a, double b, double tol 
) {
    const double phi = (sqrt(5) - 1) / 2;
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
    const string& method,
    int max_iters,
    double tol
) {
    Mat x = x0.clone();          // Текущая точка x_k
    Mat g = grad(x);             // Градиент в x_k
    Mat p = -g;                  // Начальное направление поиска
    double f_val = f(x);         // Значение функции в x_k

    const double min_step = 1e-10;
    const int restart_period = static_cast<int>(sqrt(x0.total())); // Период рестарта

    for (int k = 0; k < max_iters; ++k) {
        // === 1. Линейный поиск шага α_k ===
        auto phi = [&](double alpha) {
            return f(x + alpha * p);
            };
        double alpha = golden_section_search(phi, 0.0, 1.0, 1e-4);

        if (alpha < min_step) {
            cerr << "Warning: Step size too small (" << alpha << "), stopping at iteration " << k << endl;
            break;
        }

        // === 2. Обновление точки: x_{k+1} = x_k + α_k * p_k ===
        Mat x_new = x + alpha * p;
        Mat g_new = grad(x_new);      // Градиент в новой точке
        double f_new = f(x_new);      // Значение функции в новой точке

        // === 3. Вычисление векторов s_k и y_k ===
        Mat s = alpha * p;            // s_k = x_{k+1} - x_k
        Mat y = g_new - g;            // y_k = g_{k+1} - g_k

        // === 4. Проверка сходимости 
        double grad_norm = norm_2(g_new);
        if (k > 2) {
            double rel_change = fabs(f_new - f_val) / (fabs(f_val) + 1e-10);
            double abs_threshold = 1e-4 * fabs(f(x0)); 

            if (rel_change <= 1e-4 && fabs(f_new) <= abs_threshold) {
                cout << "Converged  at iteration " << k
                    << ": rel_change=" << rel_change << ", |f|=" << fabs(f_new) << endl;
                return x_new;
            }
        }

        // === 5. Вычисление коэффициента β_{k+1} ===
        double beta = 0.0;
        double sTy = s.dot(y);
        double sTg = s.dot(g);
        double yTg_new = y.dot(g_new);
        double g_norm2 = g.dot(g);

        if (fabs(sTy) < 1e-8) {
            beta = 0.0; 
           
        }
        else if (method == "FR") {
            beta = g_new.dot(g_new) / (g_norm2 + 1e-10);
        }
        else if (method == "PR") {
            beta = y.dot(g_new) / (g_norm2 + 1e-10);
        }
        else if (method == "DY") {
            beta = g_new.dot(g_new) / (p.dot(y) + 1e-10);
        }
        else if (method == "BKY") {
            double term1 = (f_new - f_val - 0.5 * sTy) / sTy;
            double term2 = yTg_new / sTy;
            double term3 = -sTg / sTy;
            beta = term1 + term2 + term3;
        }
        else if (method == "BKS") {
            double term1 = (f_new - f_val + 0.5 * sTg) / sTy;
            double term2 = yTg_new / sTy;
            double term3 = -sTg / sTy;
            beta = term1 + term2 + term3;
        }
        else if (method == "BKG") {
            double term1 = (f_new - f_val - 0.5 * alpha * g_norm2) / sTy;
            double term2 = yTg_new / sTy;
            double term3 = -sTg / sTy;
            beta = term1 + term2 + term3;
        }
        else {
            cerr << "Unknown method: " << method << endl;
            return x;
        }

        
        
        if (beta < 0.0) {
            beta = 0.0;
        }

        if (beta > 5.0) {
            beta = 5.0;
        }

        // === 6. Обновление направления поиска ===
        Mat p_new = -g_new + beta * p;

        if (g_new.dot(p_new) >= 0.0) {
            p_new = -g_new;
        }

        if (restart_period > 0 && (k + 1) % restart_period == 0) {
            p_new = -g_new;
        }

        // === 7. Обновление состояния для следующей итерации ===
        p = p_new;
        x = x_new;
        g = g_new;
        f_val = f_new;

        // Отладочный вывод (каждые 10 итераций)
        if (k % 10 == 0 || k == max_iters - 1) {
            cout << "Iter " << k << ": f=" << f_val << ", ||g||=" << grad_norm
                << ", alpha=" << alpha << ", beta=" << beta << endl;
        }
    }
    cout << "Reached max iterations (" << max_iters << ")" << endl;
    return x;
}
