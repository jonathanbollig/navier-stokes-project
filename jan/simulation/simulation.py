# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 19:55:12 2026

@author: Jan
"""

# Staggered grid:

#   •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
#       |       |       |       |       |       |       | 
#   ↑ - 0 - ↑ - ╬ - ↑ - ╬ - ↑ - ╬ - ↑ - ╬ - ↑ - ╬ - ↑ - ╬ - ↑
#       |       |       |       |       |       |       |
#   •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
#       |       |       |       |       |       |       | 
#   ↑ - ╬ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ╬ - ↑
#       |       |       |       |       |       |       |
#   •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
#       |       |       |       |       |       |       | 
#   ↑ - ╬ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ╬ - ↑
#       |       |       |       |       |       |       |
#   •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
#       |       |       |       |       |       |       | 
#   ↑ - ╬ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ╬ - ↑
#       |       |       |       |       |       |       |
#   •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
#       |       |       |       |       |       |       | 
#   ↑ - ╬ - ↑ - ╬ - ↑ - ╬ - ↑ - ╬ - ↑ - ╬ - ↑ - ╬ - ↑ - ╬ - ↑
#       |       |       |       |       |       |       |
#   •   →   •   →   •   →   •   →   •   →   •   →   •   →   •

#   * u is saved at locations marked by "→" -> need (N_y + 1) by N_x nodes
#   * v is saved at locations marked by "↑" -> need N_y by (N_x + 1) nodes
#   * p is saved at locations marked by "•" -> need (N_y + 1) by (N_x + 1) nodes
#   * grid points on the boundary are marked by "╬"


import numpy as np

import conversions as conv
import derivatives as deriv
import pressure as pres
import plotting as plot



def apply_boundary_conditions(x_vel: float, U: np.array, V: np.array, P: np.array) -> None:
    # U-component (no-slip, except upper boundary where u = x_vel):
    U[:, 0], U[:, -1] = 0, 0
    U[0, :] = 2 * x_vel * np.ones_like(U[0, :]) - U[1, :]
    U[-1, :] = -1 * U[-2, :]
    
    # V-component (no-slip):
    V[0, :], V[-1, :] = 0, 0
    V[:, 0] = -1 * V[:, 1]
    V[:, -1] = -1 * V[:, -2]
    
    # Pressure (no-slip):
    P[:, 0], P[:, -1] = -1 * P[:, 1], -1 * P[:, -2]
    P[0, :], P[-1, :] = -1 * P[1, :], -1 * P[-2, :]
    
    
def calc_timestep(delta_x: float, delta_y: float, u_max: float, v_max: float, Re: float, 
                  tau: float) -> float:
    if tau <= 0 or tau > 1:
        raise ValueError("tau must be from (0, 1]")
    
    Re_cond: float = Re / (2 * (delta_x**-2 + delta_y**-2))
    x_cond: float = delta_x / np.abs(u_max)
    y_cond: float = delta_y / np.abs(v_max)
    
    return tau * np.min([cond for cond in [Re_cond, x_cond, y_cond] if cond != 0.0])


def calc_F_and_G(U: np.array, V: np.array, delta_x: float, delta_y: float, delta_t: float,
                 Re: float) -> tuple[np.array, np.array]:
    # Assume (the change in) g_x is negligible:
    F: np.array = U.copy()
    F[1:-1, 1:-1] = F[1:-1, 1:-1] + delta_t / Re * (deriv.lin_x(U, delta_x, 2) + deriv.lin_y(U, delta_y, 2))
    F[1:-1, 1:-1] = F[1:-1, 1:-1] - delta_t * deriv.nonlin_x(U, V, delta_x, delta_y, delta_t, deriv.NonlinType.SQUARE)
    
    mixed_deriv: np.array = deriv.nonlin_y(U, V, delta_x, delta_y, delta_t, deriv.NonlinType.MIXED)
    F[1:-1, 1:-1] = F[1:-1, 1:-1] - delta_t * conv.U_like_from_grid(mixed_deriv)
    
    # Assume (the change in) g_y is negligible:
    G: np.array = V.copy()
    G[1:-1, 1:-1] = G[1:-1, 1:-1] + delta_t / Re * (deriv.lin_x(V, delta_x, 2) + deriv.lin_y(V, delta_y, 2))
    G[1:-1, 1:-1] = G[1:-1, 1:-1] - delta_t * deriv.nonlin_y(U, V, delta_x, delta_y, delta_t, deriv.NonlinType.SQUARE)
    
    mixed_deriv = deriv.nonlin_x(U, V, delta_x, delta_y, delta_t, deriv.NonlinType.MIXED) 
    G[1:-1, 1:-1] = G[1:-1, 1:-1] - delta_t * conv.V_like_from_grid(mixed_deriv)
    
    return F, G    
    

def lid_driven_cavity_simulation(domain_size: list[float], grid_size: list[int], x_vel: float, 
                                 Re: float, tau: float, omega: float, epsilon: float, T_max: float,
                                 N_max_P: int = 100) -> tuple[list[np.array], list[float]]:
    # Domain size:
    a: float = domain_size[0]
    b: float = domain_size[1]
    
    # Grid parameters:
    N_x: int = grid_size[0]
    N_y: int = grid_size[1]
    
    delta_x: float = a / N_x
    delta_y: float = b / N_y
    
    # Initializing field arrays:
    U: np.array = np.zeros(shape = (N_y + 1, N_x))
    V: np.array = np.zeros(shape = (N_y, N_x + 1))
    P: np.array = np.zeros(shape = (N_y + 1, N_x + 1))
    
    # Iteration variables:
    t = 0
    n = 0
    
    U_sol: list[np.array] = []
    V_sol: list[np.array] = []
    P_sol: list[np.array] = []
    t_sol: list[float] = [0]
    
    # Print a message at certain timesteps to track progress:
    print_times: list[float] = np.arange(0, int(T_max + 1), 0.25)
    print_index: int = 0
    
    while t < T_max:
        apply_boundary_conditions(x_vel, U, V, P)
        
        delta_t: float = calc_timestep(delta_x, delta_y, np.max(U), np.max(V), Re, tau)
        
        F, G = calc_F_and_G(U, V, delta_x, delta_y, delta_t, Re)
        P = pres.calc_new_pressure(F, G, P, delta_x, delta_y, delta_t, omega, epsilon, N_max_P)
        
        U[1:-1, 1:-1] = F[1:-1, 1:-1] - delta_t * conv.U_like_from_grid(conv.P_like_to_grid(deriv.lin_x(P, delta_x)))
        V[1:-1, 1:-1] = G[1:-1, 1:-1] - delta_t * conv.V_like_from_grid(conv.P_like_to_grid(deriv.lin_y(P, delta_y)))
        
        t = t + delta_t
        n = n + 1
        
        U_sol.append(U.copy())
        V_sol.append(V.copy())
        P_sol.append(P.copy())
        t_sol.append(t)
        
        # Print a message at certain timesteps to track progress:
        if t >= print_times[print_index]:
            print("t = " + str(np.round(t, 2)))
            
            print_index = print_index + 1
        
    return U_sol, V_sol, P_sol, t_sol


def main():
    domain_size: list[float] = [1, 1]
    grid_size: list[int] = [50, 50]
    x_vel: float = 2
    Re: float = 500
    tau: float = 0.5
    omega: float = 1
    epsilon: float = 0.01
    T_max: float = 5
    # N_max_P: int = 100
    
    solutions = lid_driven_cavity_simulation(domain_size, grid_size, x_vel, Re, tau, omega, epsilon, T_max)
    
    animation = plot.animate_solution(solutions, domain_size[0], domain_size[1])
    # animation.save('animation.gif', writer = 'pillow', fps = 20)
    
    return animation
        


if __name__ == '__main__':
    animation = main()