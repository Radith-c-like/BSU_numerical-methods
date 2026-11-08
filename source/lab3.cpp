#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

using namespace std;


double vector_norm(const vector<double>& a) {
    double sum = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

vector<double> vector_matrix_mul(const vector<vector<double>>& matrix, const vector<double>& vec) {
    int n_rows = matrix.size();
    vector<double> res(n_rows, 0.0);

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < matrix[i].size(); ++j) {
            res[i] += matrix[i][j] * vec[j];
        }
    }
    return res;
}

vector<double> gauss_solver(const vector<vector<double>>& coef, const vector<double>& value) {
    auto A = coef;
    auto b = value;

    if (A.size() == 0 || b.size() == 0 || A.size() != b.size()) {
        throw invalid_argument("Invalid Argument");
    }

    int n_rows = A.size();
    int n_cols = A[0].size();


    for (int k = 0; k < n_rows; ++k) {
        if (abs(A[k][k]) < 1e-12) {
            cout << "Warning: Zero pivot encountered in naive Gauss step" << k;
        }

        for (int i = k + 1; i < n_rows; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k + 1; j < n_cols; ++j) {
                A[i][j] -= factor * A[k][j]; 
            }

            b[i] -= factor * b[k];
        }
    }

    vector<double> x(n_cols,0.0);

    for (int i = n_rows - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n_cols; ++j) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    } 

    return x;
}


vector<double> gaus_with_perm(const vector<vector<double>>& coef, const vector<double>& val) {
    auto A = coef;
    auto b = val;

    if (A.size() == 0 || b.size() == 0 || A.size() != b.size()) {
        throw invalid_argument("Invalid Argument");

    }

    int n_rows = A.size();
    int n_cols = A[0].size();

    for (int k = 0; k < n_rows; ++k) {
        int max_row = k;
        for (int i = k + 1; i < n_rows; ++i) {
            if (abs(A[i][k]) > abs(A[max_row][k])) {
                max_row = i;
            }
        }

        swap(A[k], A[max_row]);
        swap(b[k], b[max_row]);

        
        for (int i = k + 1; i < n_rows; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k + 1; j < n_cols; ++j) {
                A[i][j] -= factor * A[k][j]; 
            }

            b[i] -= factor * b[k];
        }
    }
    auto x = vector<double>(n_cols, 0.0); 
    for (int k = n_rows-1; k >= 0; --k) {
        double sum = 0.0;
        for (int j = k + 1; j < n_cols; ++j) {
            sum += A[k][j] * x[j];
        }
        x[k] = (b[k] - sum) / A[k][k];
    }

    return x;

}


void print_results(const string& method_name, const vector<double>& x_approx, const vector<double>& x_exact, 
                   const vector<vector<double>>& A_orig, const vector<double>& b_orig, double time_sec) {
    int n = A_orig.size();
    
    // 1. Первые 5 координат
    cout << "[" << method_name << "]" << endl;
    cout << "First 5 coordinates of x*: ";
    for (int i = 0; i < 5 && i < n; ++i) {
        cout << x_approx[i] << " ";
    }
    cout << endl;

    // 2. Норма вектора невязки ||Ax* - b||
    vector<double> Ax = vector_matrix_mul(A_orig, x_approx);
    vector<double> residual(n);
    for (int i = 0; i < n; ++i) {
        residual[i] = Ax[i] - b_orig[i];
    }
    double residual_norm = vector_norm(residual);
    cout << "Residual norm ||Ax* - b||: " << scientific << residual_norm << defaultfloat << endl;

    // 3. Относительная погрешность ||x* - x_exact|| / ||x_exact||
    vector<double> diff(n);
    for (int i = 0; i < n; ++i) {
        diff[i] = x_approx[i] - x_exact[i];
    }
    double rel_error = vector_norm(diff) / vector_norm(x_exact);
    cout << "Relative error: " << scientific << rel_error << defaultfloat << endl;

    // 4. Время
    cout << "Time: " << fixed << setprecision(3) << time_sec << " sec" << endl;
    cout << "--------------------------------------------------\n";
}



int main() {
    const int n = 2000; 
    const int m = 23;

    cout << "Starting Lab 3 with n = " << n << ", m = " << m << "\n\n";

    
    mt19937 gen(42);
    uniform_real_distribution<double> dist(-100.0, 100.0);

    
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> x_exact(n);
    
    for (int i = 0; i < n; ++i) {
        x_exact[i] = m + i; // x = (m, m+1, ..., n+m-1)
        for (int j = 0; j < n; ++j) {
            A[i][j] = dist(gen);
        }
    }

    
    cout << "Generating matrix and computing b = A * x..." << endl;
    vector<double> b = vector_matrix_mul(A, x_exact);
    cout << "Done.\n" << endl;

    
    auto A_copy1 = A;
    auto b_copy1 = b;
    
    auto start1 = chrono::high_resolution_clock::now();
    vector<double> x_naive = gauss_solver(A_copy1, b_copy1);
    auto end1 = chrono::high_resolution_clock::now();
    
    chrono::duration<double> time1 = end1 - start1;
    print_results("Gauss WITHOUT pivoting", x_naive, x_exact, A, b, time1.count());


    
    vector<vector<double>> A_copy2 = A;
    vector<double> b_copy2 = b;
    
    auto start2 = chrono::high_resolution_clock::now();
    vector<double> x_pivot = gaus_with_perm(A_copy2, b_copy2);
    auto end2 = chrono::high_resolution_clock::now();
    
    chrono::duration<double> time2 = end2 - start2;
    print_results("Gauss WITH pivoting", x_pivot, x_exact, A, b, time2.count());

    return 0;
}