# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 19:55:12 2026

@author: common (based on Jan's code)
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
import pickle
import os
import conversions as conv
import derivatives as deriv
import pressure as pres
import plotting as plot

class navier_stokes_simulation:
    def __init__(self, xn: int, yn: int, len_x: float, len_y: float, x_vel: float,
                    Re: float, tau: float, omega: float, epsilon: float):
            # Domain size:
            self.xn: int = xn
            self.yn: int = yn
            
            # Grid parameters:
            self.len_x: float = len_x
            self.len_y: float = len_y
            
            self.delta_x: float = self.len_x / self.xn
            self.delta_y: float = self.len_y / self.yn
            
            # Fluid parameters:
            self.Re: float = Re
            self.tau: float = tau
            self.omega: float = omega
            self.epsilon: float = epsilon
            self.x_vel: float = x_vel  # velocity of the lid (top boundary)
            
            # Initializing field arrays:
            self.U: np.array = np.zeros(shape = (self.yn + 1, self.xn))
            self.V: np.array = np.zeros(shape = (self.yn, self.xn + 1))
            self.P: np.array = np.zeros(shape = (self.yn + 1, self.xn + 1))
            
            # History storage:
            self.u_history: list[np.array] = []
            self.v_history: list[np.array] = []
            self.p_history: list[np.array] = []
            self.t_history: list[float] = []
    
    def calc_timestep(self) -> float:
        if self.tau <= 0 or self.tau > 1:
            raise ValueError("tau must be from (0, 1]")
        
        try:
            Re_cond: float = self.Re / (2 * (self.delta_x**-2 + self.delta_y**-2))
        except ZeroDivisionError:
            Re_cond = np.inf
        try:
            x_cond: float = self.delta_x / np.abs(np.max(self.U))
        except ZeroDivisionError:
            x_cond = np.inf
        try:
            y_cond: float = self.delta_y / np.abs(np.max(self.V))
        except ZeroDivisionError:
            y_cond = np.inf
        
        return self.tau * np.min([cond for cond in [Re_cond, x_cond, y_cond] if cond != 0.0])
    
    def calc_F_and_G(self, delta_t: float) -> tuple[np.array, np.array]:
        # Assume (the change in) g_x is negligible:
        F: np.array = self.U.copy()
        F[1:-1, 1:-1] = F[1:-1, 1:-1] + delta_t / self.Re * (deriv.lin_x(self.U, self.delta_x, 2) + deriv.lin_y(self.U, self.delta_y, 2))
        F[1:-1, 1:-1] = F[1:-1, 1:-1] - delta_t * deriv.nonlin_x(self.U, self.V, self.delta_x, self.delta_y, delta_t, deriv.NonlinType.SQUARE)
        
        mixed_deriv: np.array = deriv.nonlin_y(self.U, self.V, self.delta_x, self.delta_y, delta_t, deriv.NonlinType.MIXED)
        F[1:-1, 1:-1] = F[1:-1, 1:-1] - delta_t * conv.U_like_from_grid(mixed_deriv)
        
        # Assume (the change in) g_y is negligible:
        G: np.array = self.V.copy()
        G[1:-1, 1:-1] = G[1:-1, 1:-1] + delta_t / self.Re * (deriv.lin_x(self.V, self.delta_x, 2) + deriv.lin_y(self.V, self.delta_y, 2))
        G[1:-1, 1:-1] = G[1:-1, 1:-1] - delta_t * deriv.nonlin_y(self.U, self.V, self.delta_x, self.delta_y, delta_t, deriv.NonlinType.SQUARE)
        
        mixed_deriv = deriv.nonlin_x(self.U, self.V, self.delta_x, self.delta_y, delta_t, deriv.NonlinType.MIXED) 
        G[1:-1, 1:-1] = G[1:-1, 1:-1] - delta_t * conv.V_like_from_grid(mixed_deriv)
        
        return F, G
    
    def apply_boundary_conditions(self,) -> None:
        # U-component (no-slip, except upper boundary where u = x_vel):
        self.U[:, 0], self.U[:, -1] = 0, 0
        self.U[0, :] = 2 * self.x_vel * np.ones_like(self.U[0, :]) - self.U[1, :]
        self.U[-1, :] = -1 * self.U[-2, :]
        
        # V-component (no-slip):
        self.V[0, :], self.V[-1, :] = 0, 0
        self.V[:, 0] = -1 * self.V[:, 1]
        self.V[:, -1] = -1 * self.V[:, -2]
        
        # Pressure (no-slip):
        self.P[:, 0], self.P[:, -1] = -1 * self.P[:, 1], -1 * self.P[:, -2]
        self.P[0, :], self.P[-1, :] = -1 * self.P[1, :], -1 * self.P[-2, :]
        
    def iterate(self, t_end: float, N_max_P: int = 100) -> None:
        # Print a message at certain timesteps to track progress:
        print_times: list[float] = np.linspace(0, t_end, 20)
        print_index: int = 0

        t: float = 0
        while t < t_end:
            self.apply_boundary_conditions()
            delta_t: float = self.calc_timestep()

            F, G = self.calc_F_and_G(delta_t)
            self.P = pres.calc_new_pressure(F, G, self.P, self.delta_x, self.delta_y, delta_t, self.omega, self.epsilon, N_max_P)

            self.U[1:-1, 1:-1] = F[1:-1, 1:-1] - delta_t * conv.U_like_from_grid(conv.P_like_to_grid(deriv.lin_x(self.P, self.delta_x)))
            self.V[1:-1, 1:-1] = G[1:-1, 1:-1] - delta_t * conv.V_like_from_grid(conv.P_like_to_grid(deriv.lin_y(self.P, self.delta_y)))

            t = t + delta_t

            self.u_history.append(self.U.copy())
            self.v_history.append(self.V.copy())
            self.p_history.append(self.P.copy())
            self.t_history.append(t)

            # Print a message at certain timesteps to track progress:
            if t >= print_times[print_index]:
                print(f"Passed t = {print_times[print_index]:.2f} / {t_end:.2f}")    
                print_index = print_index + 1
    
    def save(self, filename: str) -> None:
        """Save the complete simulation object to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


if __name__ == '__main__':
    # permanent values:
    tau: float = 1
    omega: float = 1
    epsilon: float = 0.01
    x_vel: float = 2

    # variables:
    nx: int = 70
    ny: int = 70
    len_x: float = 1
    len_y: float = 1
    Re: float = 500
    T_max: float = 7
    
    filename: str = f"lid_driven_nx{nx}_ny{ny}_re{Re}_t{int(T_max*1000)}.pkl"
    
    # Check if file exists
    if os.path.exists(filename):
        print(f"Loading simulation from {filename}...")
        with open(filename, 'rb') as f:
            simulation = pickle.load(f)
    else:
        print(f"Running new simulation...")
        simulation = navier_stokes_simulation(nx, ny, len_x, len_y, x_vel, Re, tau, omega, epsilon)
        simulation.iterate(t_end=T_max)
        simulation.save(filename)
        print(f"Simulation saved to {filename}")
    
    plot_log_vel = True # False # enable logarithmic scaling of velocity vectors
    quiver_scale = 14   # 8     # adjust length of plotted arrows (smaller -> longer)
    
    animation = plot.animate_simulation(simulation, quiver_scale, plot_log_vel)
