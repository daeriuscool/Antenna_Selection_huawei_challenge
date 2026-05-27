import numpy as np
from numpy.random import normal
import time


def generate_V(N, L):
    V = normal(size=(N,L)) + 1j * normal(size=(N,L)) 
    column_norms = np.linalg.norm(V, axis=0)
    V /= column_norms 
    antenna_max = np.max(np.linalg.norm(V, axis=1))
    V /= antenna_max 
    return V


def calc_general_obj(V_eq, sigma):
    matrix = V_eq @ V_eq.conj().T + sigma * np.eye(V_eq.shape[0])
    return np.real(np.linalg.det(matrix))

if __name__ == "__main__":
    N = 1000
    L = 8
    P = 1
    sigma = 1.0
    K_off = N // 4

    V = generate_V(N, L)

    powers = np.sum(np.abs(V)**2, axis=1) 
    M = np.einsum('ni,nj->nij', V.conj(), V)

    weakest_indices = np.argsort(powers)[:K_off]
    h1_active_mask = np.ones(N, dtype=bool)
    h1_active_mask[weakest_indices] = False
    
    max_p_h1 = np.max(powers[h1_active_mask])
    z_h1 = np.sqrt(P / max_p_h1)
    V_eq_h1 = z_h1 * np.sum(M[h1_active_mask], axis=0)
    score_h1 = calc_general_obj(V_eq_h1, sigma)
    
    print(f"H1 Оценка: {score_h1}")

    interference_terms = V[:, 0].conj() * V[:, 1]
    
    h2_active_mask = np.ones(N, dtype=bool)
    
    for _ in range(K_off):
        active_indices = np.where(h2_active_mask)[0]
        current_I_total = np.sum(interference_terms[h2_active_mask])
        
        test_I = current_I_total - interference_terms[active_indices]
        magnitudes = np.abs(test_I)
        
        best_local_idx = np.argmin(magnitudes)
        best_antenna_to_drop = active_indices[best_local_idx]
        
        h2_active_mask[best_antenna_to_drop] = False

    max_p_h2 = np.max(powers[h2_active_mask])
    z_h2 = np.sqrt(P / max_p_h2)
    V_eq_h2 = z_h2 * np.sum(M[h2_active_mask], axis=0)
    score_h2 = calc_general_obj(V_eq_h2, sigma)
    
    print(f"H2 Оценка: {score_h2}")
    
    best_heuristic_score = max(score_h1, score_h2)

    start_time = time.time()

    active_mask = np.ones(N, dtype=bool)
    current_sum_matrix = np.sum(M, axis=0)

    for step in range(K_off):
        best_obj = -np.inf
        best_antenna_to_drop = -1
        
        active_indices = np.where(active_mask)[0]
        active_powers = powers[active_indices]
        
        top2_local_indices = np.argpartition(active_powers, -2)[-2:]
        if active_powers[top2_local_indices[0]] < active_powers[top2_local_indices[1]]:
            top2_local_indices = top2_local_indices[::-1]
            
        top1_idx = active_indices[top2_local_indices[0]]
        top2_idx = active_indices[top2_local_indices[1]]
        top1_power = powers[top1_idx]
        top2_power = powers[top2_idx]

        for idx in active_indices:
            max_power_test = top2_power if idx == top1_idx else top1_power
            z_test = np.sqrt(P / max_power_test)
            
            test_matrix = current_sum_matrix - M[idx]
            V_eq_test = z_test * test_matrix
            
            obj = calc_general_obj(V_eq_test, sigma)
            
            if obj > best_obj:
                best_obj = obj
                best_antenna_to_drop = idx
                
        active_mask[best_antenna_to_drop] = False
        current_sum_matrix -= M[best_antenna_to_drop]
        
    score_greedy = best_obj
    time_taken = time.time() - start_time
    print(f"Время работы жадного: {time_taken}. Оценка: {score_greedy}")

    improvement = ((score_greedy - best_heuristic_score) / abs(best_heuristic_score)) * 100
    
    print(f"Улучшение: {improvement:.2f}%")