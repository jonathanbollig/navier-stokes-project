# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 15:22:10 2026

@author: Jan
"""

import numpy as np

import conversions as conv
import derivatives as deriv



def norm_L2(field: np.array) -> float:
    return np.sqrt(1 / (field.shape[0] * field.shape[1]) * np.cumsum(np.square(field))[-1])

def calc_new_pressure(F: np.array, G: np.array, P: np.array, delta_x: float, delta_y: float, 
                      delta_t: float, omega: float, epsilon: float, N_max: int = 100) -> np.array:
    # Convert F and G to actual grid points since they refer to different coordinate systems:
    F_grid: np.array = conv.U_like_to_grid(F)
    G_grid: np.array = conv.V_like_to_grid(G)
    RHS: np.array = 1 / delta_t * (deriv.lin_x(F_grid, delta_x) + deriv.lin_y(G_grid, delta_y))
    
    # Convert RHS-array to P-grid since it referred to actual grid:
    RHS = conv.P_like_from_grid(RHS)
    
    P_it: np.array = P.copy()
    P_0_norm: float = norm_L2(P)
    residual_norm: float = epsilon * P_0_norm + 1
    n = 0
    
    while residual_norm >= epsilon * P_0_norm and n < N_max:
        P_new: np.array = np.zeros_like(P_it)
        
        P_sum: np.array = (P_it[1:-1, 2:] + P_it[1:-1, :-2]) / delta_x**2 + (P_it[2:, 1:-1] + P_it[:-2, 1:-1]) / delta_y**2
        P_new[1:-1, 1:-1] = (1 - omega) * P_it[1:-1, 1:-1] + omega / (2 * (1 / delta_x**2 + 1 / delta_y**2)) * (P_sum - RHS)
        
        # Set boundary values:
        P_new[0, :], P_new[-1, :] = P_it[1, :], P_it[-2, :]
        P_new[:, 0], P_new[:, -1] = P_it[:, 1], P_it[:, -2]
        
        residual: np.array = deriv.lin_x(P_new, delta_x, 2) + deriv.lin_y(P_new, delta_y, 2) - RHS
        residual_norm = norm_L2(residual)
        
        P_it = P_new
        n = n + 1

    return P_it