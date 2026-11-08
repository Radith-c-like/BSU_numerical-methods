#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <omp.h>

using namespace std;



vector<vector<int>> blocs_mul(const vector<vector<int>>&a, const vector<vector<int>>&b, int bc_size) {
    if (a.size() == 0 || b.size() == 0) {
        throw invalid_argument("matrix must exist and not be empty");
    }   

    int a_rows = a.size();
    int a_cols = a[0].size();
    int b_rows = b.size();
    int b_cols = b[0].size();

    if (a_cols != b_rows) {
        throw invalid_argument("matrix sizes are not available for multiplication");
    }

    auto res = vector<vector<int>>(a_rows, vector<int>(b_cols, 0));
    int Q = (a_rows + bc_size - 1) / bc_size; // кол-во блоков в строке
    #pragma omp parallel for
    for (int igl = 0; igl < Q; ++igl) {
        int ai_left = igl * bc_size; // координата строки матрицы а в блоке g_c
        int ai_right = min(ai_left + bc_size, a_rows);
        
        for (int kgl = 0; kgl < Q; ++kgl) {
            int ak_left = kgl * bc_size;
            int ak_right = min(ak_left + bc_size, a_cols);

            for (int jgl = 0; jgl < Q; ++jgl) {
                int bj_left = jgl * bc_size;
                int bj_right = min(bj_left + bc_size, b_cols);

                for (int i = ai_left; i < ai_right; ++i) {
                    for (int k = ak_left; k < ak_right; ++k) {
                        auto aik = a[i][k];
                        for (int j = bj_left; j < bj_right; ++j) {
                            res[i][j] += aik * b[k][j];
                        }
                    }
                }
            }
        }
    }
    return res;
}


vector<vector<int>> blocs_mul_collapse(const vector<vector<int>>& a, const vector<vector<int>>& b, int bc_size) {
    if (a.size() == 0 || b.size() == 0) {
        throw invalid_argument("matrix must exist and not be empty");
    }   

    int a_rows = a.size();
    int a_cols = a[0].size();
    int b_rows = b.size();
    int b_cols = b[0].size();

    if (a_cols != b_rows) {
        throw invalid_argument("matrix sizes are not available for multiplication");
    }

    auto res = vector<vector<int>>(a_rows, vector<int>(b_cols, 0));
    int Q = (a_rows + bc_size - 1) / bc_size; 

    #pragma omp parallel for collapse(2)
    for (int igl = 0; igl < Q; ++igl) {
        for (int jgl = 0; jgl < Q; ++jgl) {            
            for (int kgl = 0; kgl < Q; ++kgl) {
                int ai_left = igl * bc_size; 
                int ai_right = min(ai_left + bc_size, a_rows);
                int ak_left = kgl * bc_size;
                int ak_right = min(ak_left + bc_size, a_cols);
                int bj_left = jgl * bc_size;
                int bj_right = min(bj_left + bc_size, b_cols);

                for (int i = ai_left; i < ai_right; ++i) {
                    for (int k = ak_left; k < ak_right; ++k) {
                        int aik = a[i][k]; 
                        for (int j = bj_left; j < bj_right; ++j) {
                            res[i][j] += aik * b[k][j];
                        }
                    }
                }
            }
        }
    }
    return res;
}

int main() {
    const int N = 2000;
    const int r = 100;
    vector<int> thread_counts = {1, 2, 4, 6, 8};

    cout << "Generating matrices " << N << "x" << N << "..." << endl;
    mt19937 gen(42);
    uniform_int_distribution<> dist(1, 10);
    
    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = dist(gen);
            B[i][j] = dist(gen);
        }
    }
    cout << "Matrices are ready. Starting tests...\n" << endl;

    // ИСПРАВЛЕНО: Тестируем оба метода в одном цикле
    for (int p : thread_counts) {
        omp_set_num_threads(p); 
        
        // --- Тест 1: Только внешний цикл ---
        double start_time_1 = omp_get_wtime();
        vector<vector<int>> C1 = blocs_mul(A, B, r);
        double end_time_1 = omp_get_wtime();
        
        // --- Тест 2: Collapse(2) ---
        double start_time_2 = omp_get_wtime();
        vector<vector<int>> C2 = blocs_mul_collapse(A, B, r);
        double end_time_2 = omp_get_wtime();
        
        // Вывод
        cout << "Threads: " << p << " | "
             << "Outer loop: " << (end_time_1 - start_time_1) << " sec (C[0][0]=" << C1[0][0] << ") | "
             << "Collapse(2): " << (end_time_2 - start_time_2) << " sec (C[0][0]=" << C2[0][0] << ")" << endl;
    }

    return 0;
}