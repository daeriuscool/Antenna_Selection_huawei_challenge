#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense> // Подключение библиотеки Eigen

using namespace Eigen;
using namespace std;
using cd = complex<double>;

// Генерация матрицы V
MatrixXcd generate_V(int N, int L) {
    MatrixXcd V(N, L);
    mt19937 gen(42); // Зафиксированный seed для повторяемости (как в np.random)
    normal_distribution<double> dist(0.0, 1.0);

    // Заполнение нормальным распределением (вещественная и мнимая части)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < L; ++j) {
            V(i, j) = cd(dist(gen), dist(gen));
        }
    }

    // Нормализация столбцов (column_norms)
    for (int j = 0; j < L; ++j) {
        double col_norm = V.col(j).norm();
        V.col(j) /= col_norm;
    }

    // Нормализация по максимальной мощности антенны (max row norm)
    double max_norm = 0;
    for (int i = 0; i < N; ++i) {
        max_norm = max(max_norm, V.row(i).norm());
    }
    V /= max_norm;

    return V;
}

// Целевая функция
double calc_general_obj(const MatrixXcd& V_eq, double sigma) {
    int L = V_eq.rows();
    // matrix = V_eq @ V_eq.conj().T + sigma * I
    MatrixXcd matrix = V_eq * V_eq.adjoint() + MatrixXcd::Identity(L, L) * sigma;
    // Возвращаем вещественную часть определителя
    return matrix.determinant().real();
}

int main() {
    int N = 1000;
    int L = 8;
    double P = 1.0;
    double sigma = 1.0;
    int K_off = N / 4;

    MatrixXcd V = generate_V(N, L);

    // powers = np.sum(np.abs(V)**2, axis=1)
    vector<double> powers(N);
    for (int i = 0; i < N; ++i) {
        powers[i] = V.row(i).squaredNorm();
    }

    // M = np.einsum('ni,nj->nij', V.conj(), V)
    // В C++ храним это как вектор из N матриц (L x L)
    vector<MatrixXcd> M(N, MatrixXcd(L, L));
    for (int i = 0; i < N; ++i) {
        VectorXcd v_row = V.row(i).transpose();
        // Внешнее произведение: v_row.conj() * v_row.T
        M[i] = v_row.conjugate() * v_row.transpose();
    }

    // ==========================================
    // Эвристика H1
    // ==========================================
    vector<int> sorted_indices(N);
    iota(sorted_indices.begin(), sorted_indices.end(), 0);
    sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
        return powers[a] < powers[b];
        });

    vector<bool> h1_active_mask(N, true);
    for (int i = 0; i < K_off; ++i) {
        h1_active_mask[sorted_indices[i]] = false;
    }

    double max_p_h1 = 0;
    MatrixXcd sum_M_h1 = MatrixXcd::Zero(L, L);
    for (int i = 0; i < N; ++i) {
        if (h1_active_mask[i]) {
            max_p_h1 = max(max_p_h1, powers[i]);
            sum_M_h1 += M[i];
        }
    }
    double z_h1 = sqrt(P / max_p_h1);
    double score_h1 = calc_general_obj(z_h1 * sum_M_h1, sigma);
    cout << "H1 Оценка: " << score_h1 << "\n";

    // ==========================================
    // Эвристика H2
    // ==========================================
    vector<cd> interference_terms(N);
    for (int i = 0; i < N; ++i) {
        interference_terms[i] = conj(V(i, 0)) * V(i, 1);
    }

    vector<bool> h2_active_mask(N, true);
    for (int step = 0; step < K_off; ++step) {
        cd current_I_total(0.0, 0.0);
        vector<int> active_indices;
        active_indices.reserve(N);

        for (int i = 0; i < N; ++i) {
            if (h2_active_mask[i]) {
                current_I_total += interference_terms[i];
                active_indices.push_back(i);
            }
        }

        double min_mag = numeric_limits<double>::infinity();
        int best_antenna_to_drop = -1;

        for (int idx : active_indices) {
            cd test_I = current_I_total - interference_terms[idx];
            double mag = abs(test_I);
            if (mag < min_mag) {
                min_mag = mag;
                best_antenna_to_drop = idx;
            }
        }
        h2_active_mask[best_antenna_to_drop] = false;
    }

    double max_p_h2 = 0;
    MatrixXcd sum_M_h2 = MatrixXcd::Zero(L, L);
    for (int i = 0; i < N; ++i) {
        if (h2_active_mask[i]) {
            max_p_h2 = max(max_p_h2, powers[i]);
            sum_M_h2 += M[i];
        }
    }
    double z_h2 = sqrt(P / max_p_h2);
    double score_h2 = calc_general_obj(z_h2 * sum_M_h2, sigma);
    cout << "H2 Оценка: " << score_h2 << "\n";

    double best_heuristic_score = max(score_h1, score_h2);

    // ==========================================
    // Жадный алгоритм (Greedy Search)
    // ==========================================
    auto start_time = chrono::high_resolution_clock::now();

    vector<bool> active_mask(N, true);
    MatrixXcd current_sum_matrix = MatrixXcd::Zero(L, L);
    for (int i = 0; i < N; ++i) {
        current_sum_matrix += M[i];
    }

    double score_greedy = 0;

    for (int step = 0; step < K_off; ++step) {
        double best_obj = -numeric_limits<double>::infinity();
        int best_antenna_to_drop = -1;

        // Поиск top-1 и top-2 самых мощных активных антенн (Оптимизация, как в Python)
        int top1_idx = -1, top2_idx = -1;
        double top1_power = -1.0, top2_power = -1.0;

        for (int i = 0; i < N; ++i) {
            if (active_mask[i]) {
                if (powers[i] > top1_power) {
                    top2_power = top1_power;
                    top2_idx = top1_idx;
                    top1_power = powers[i];
                    top1_idx = i;
                }
                else if (powers[i] > top2_power) {
                    top2_power = powers[i];
                    top2_idx = i;
                }
            }
        }

        for (int idx = 0; idx < N; ++idx) {
            if (!active_mask[idx]) continue;

            double max_power_test = (idx == top1_idx) ? top2_power : top1_power;
            double z_test = sqrt(P / max_power_test);

            MatrixXcd test_matrix = current_sum_matrix - M[idx];
            MatrixXcd V_eq_test = z_test * test_matrix;

            double obj = calc_general_obj(V_eq_test, sigma);

            if (obj > best_obj) {
                best_obj = obj;
                best_antenna_to_drop = idx;
            }
        }

        active_mask[best_antenna_to_drop] = false;
        current_sum_matrix -= M[best_antenna_to_drop];
        score_greedy = best_obj;
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> time_taken = end_time - start_time;

    cout << "Время работы жадного: " << time_taken.count() << "s. Оценка: " << score_greedy << "\n";

    double improvement = ((score_greedy - best_heuristic_score) / abs(best_heuristic_score)) * 100.0;
    cout << "Улучшение: " << improvement << "%\n";

    return 0;
}