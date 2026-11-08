#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace std;

template <typename T>
vector<vector<T>> gen_matrix(int n) {
    vector<vector<T>> matrix(n, vector<T>(n));
    mt19937 gen(42);
    uniform_real_distribution<double> dis(-100.0, 100.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = static_cast<T>(dis(gen));
        }
    }
    return matrix;
}

template <typename T>
vector<vector<T>> points_matrix_mul(const vector<vector<T>>& a, const vector<vector<T>>& b, int n) {
    vector<vector<T>> c(n, vector<T>(n, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

template <typename T>
vector<vector<T>> blocks_matrix_mul(const vector<vector<T>>& a, const vector<vector<T>>& b, int r, int n) {
    vector<vector<T>> c(n, vector<T>(n, 0));

    int Q = (n + r - 1) / r;

    for (int igl = 0; igl < Q; ++igl) {
        int ai_left = igl * r;
        int ai_right = min((igl + 1) * r, n);

        for (int kgl = 0; kgl < Q; ++kgl) {
            int ak_left = kgl * r;
            int ak_right = min((kgl + 1) * r, n);
            
            for (int jgl = 0; jgl < Q; ++jgl) {
                int bj_left = jgl * r;
                int bj_right = min((jgl + 1) * r, n);
                
                for (int i = ai_left; i < ai_right; ++i) {
                    for (int j = bj_left; j < bj_right; ++j) {
                        for (int k = ak_left; k < ak_right; ++k) {
                            c[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }

            }
        }
    }

    return c;
}

template <typename T>
bool check_correctness(const vector<vector<T>>& c1, const vector<vector<T>>& c2, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (abs(c1[i][j] - c2[i][j]) > 1e-5) return false;
        }
    }
    return true;
}

int main() {
    vector<int> N_values = {20, 64, 512, 1024, 2048};
    vector<int> r_values = {2, 5, 16, 20, 32, 64, 128};

    cout << fixed << setprecision(2);

    for (int n : N_values) {
        cout << "\n==========================================" << endl;
        cout << "Матрица N = " << n << "x" << n << endl;
        cout << "==========================================" << endl;

        auto a = gen_matrix<double>(n);
        auto b = gen_matrix<double>(n);

        auto start = chrono::high_resolution_clock::now();
        auto c_naive = points_matrix_mul(a, b, n);
        auto end = chrono::high_resolution_clock::now();
        double time_naive = chrono::duration<double, milli>(end - start).count();

        cout << "Точечный алгоритм (1): " << time_naive << " ms" << endl;
        cout << "------------------------------------------" << endl;

        for (int r : r_values) {
            if (r > n) continue;

            start = chrono::high_resolution_clock::now();
            auto c_block = blocks_matrix_mul(a, b, r, n);
            end = chrono::high_resolution_clock::now();
            double time_block = chrono::duration<double, milli>(end - start).count();

            bool ok = check_correctness(c_naive, c_block, n);

            cout << "Блочный (r = " << setw(3) << r << "): " 
                 << setw(8) << time_block << " ms | "
                 << (ok ? "OK" : "FAIL") << endl;
        }
    }

    return 0;
}