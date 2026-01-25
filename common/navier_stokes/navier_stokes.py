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
import plotting as plot

from typing import Literal
from numba import jit

@jit(nopython=True)
def sor_solver_fast(P_it, RHS, omega, delta_x, delta_y, yn, xn, N_max, epsilon, P_0_norm):
    dx2 = delta_x**2
    dy2 = delta_y**2
    n = 0
    residual = np.zeros_like(P_it)

    residual_norm = epsilon * P_0_norm + 1.0 
    
    while residual_norm > epsilon * P_0_norm and n < N_max:
        
        for j in range(1, yn):
            for i in range(1, xn):
                term_x = (P_it[j, i+1] + P_it[j, i-1]) / dx2
                term_y = (P_it[j+1, i] + P_it[j-1, i]) / dy2
                P_it[j, i] = (1 - omega) * P_it[j, i] + omega / (2 * (1/dx2 + 1/dy2)) * (term_x + term_y - RHS[j, i])

        P_it[0, :], P_it[-1, :] = P_it[1, :], P_it[-2, :]
        P_it[:, 0], P_it[:, -1] = P_it[:, 1], P_it[:, -2]

        max_res = 0

        for j in range(1, yn):
            for i in range(1, xn):
                residual = (P_it[j, i+1] - 2*P_it[j,i] + P_it[j, i-1])/dx2 + (P_it[j+1, i] - 2*P_it[j,i] + P_it[j-1, i])/dy2 - RHS[j, i]

                if abs(residual) > max_res:
                    max_res = abs(residual)
        residual_norm = max_res

        n +=1
    return P_it

def norm_L2(field: np.ndarray) -> float:
    return np.sqrt(1 / (field.shape[0] * field.shape[1]) * np.cumsum(np.square(field))[-1])

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
            self.U: np.ndarray = np.zeros(shape = (self.yn + 1, self.xn))
            self.V: np.ndarray = np.zeros(shape = (self.yn, self.xn + 1))
            self.P: np.ndarray = np.zeros(shape = (self.yn + 1, self.xn + 1))
            
            # History storage:
            self.u_history: list[np.ndarray] = []
            self.v_history: list[np.ndarray] = []
            self.p_history: list[np.ndarray] = []
            self.t_history: list[float] = []
            self.sparceify_factor: int = 1  # default no sparcification

            # to be filled later
            self.boundary_type = None
            self.boxes = []
        
    def set_boundary_type_and_boxes(self, type_: str, boxes: list) -> None:
        self.boundary_type = type_
        self.boxes = boxes

    def calc_timestep(self) -> float:
        if self.tau <= 0 or self.tau > 1:
            raise ValueError("tau must be from (0, 1]")
        
        try:
            Re_cond: float = self.Re / (2 * (self.delta_x**-2 + self.delta_y**-2))
        except ZeroDivisionError:
            Re_cond = np.inf
        try:
            x_cond: float = self.delta_x / np.max(np.abs(self.U))
        except ZeroDivisionError:
            x_cond = np.inf
        try:
            y_cond: float = self.delta_y / np.max(np.abs(self.V))
        except ZeroDivisionError:
            y_cond = np.inf
        
        return self.tau * np.min([Re_cond, x_cond, y_cond])  # removed check if cond != 0.0 since I don't think it should occur.
    
    def calc_F_and_G(self, delta_t: float) -> tuple[np.ndarray, np.ndarray]:
        # Assume (the change in) g_x is negligible:
        F: np.ndarray = self.U.copy()
        F[1:-1, 1:-1] = F[1:-1, 1:-1] + delta_t / self.Re * (deriv.lin_x(self.U, self.delta_x, 2) + deriv.lin_y(self.U, self.delta_y, 2))
        F[1:-1, 1:-1] = F[1:-1, 1:-1] - delta_t * deriv.nonlin_x(self.U, self.V, self.delta_x, self.delta_y, delta_t, deriv.NonlinType.SQUARE)
        
        mixed_deriv: np.ndarray = deriv.nonlin_y(self.U, self.V, self.delta_x, self.delta_y, delta_t, deriv.NonlinType.MIXED)
        F[1:-1, 1:-1] = F[1:-1, 1:-1] - delta_t * conv.U_like_from_grid(mixed_deriv)
        
        # Assume (the change in) g_y is negligible:
        G: np.ndarray = self.V.copy()
        G[1:-1, 1:-1] = G[1:-1, 1:-1] + delta_t / self.Re * (deriv.lin_x(self.V, self.delta_x, 2) + deriv.lin_y(self.V, self.delta_y, 2))
        G[1:-1, 1:-1] = G[1:-1, 1:-1] - delta_t * deriv.nonlin_y(self.U, self.V, self.delta_x, self.delta_y, delta_t, deriv.NonlinType.SQUARE)
        
        mixed_deriv = deriv.nonlin_x(self.U, self.V, self.delta_x, self.delta_y, delta_t, deriv.NonlinType.MIXED) 
        G[1:-1, 1:-1] = G[1:-1, 1:-1] - delta_t * conv.V_like_from_grid(mixed_deriv)
        
        return F, G
    
    def calc_pressure(self, F: np.ndarray, G: np.ndarray, delta_t: float, N_max: int = 100) -> None:
        # Convert F and G to actual grid points since they refer to different coordinate systems:
        F_grid: np.ndarray = conv.U_like_to_grid(F)
        G_grid: np.ndarray = conv.V_like_to_grid(G)
        P_it: np.ndarray = self.P.copy()
        RHS = np.zeros_like(P_it)

        # deriv_x = deriv.lin_x(F_grid, self.delta_x)
        # deriv_y = deriv.lin_y(G_grid, self.delta_y)
        
        # RHS[1:-2, 1:-2]: np.ndarray = 1 / delta_t * (deriv.lin_x(F_grid, self.delta_x) + deriv.lin_y(G_grid, self.delta_y))
        
        # Compute derivative of F and G on the pressure grid
        # F has shape (yn+1, xn), take derivative in x direction at pressure points
        deriv_F = np.zeros((self.yn + 1, self.xn + 1))
        deriv_F[:, 1:-1] = (F[:, 1:] - F[:, :-1]) / self.delta_x
        deriv_F[:, 0] = (F[:, 0] - 0) / self.delta_x      # assume F=0 at left
        deriv_F[:, -1] = (0 - F[:, -1]) / self.delta_x    # assume F=0 at right
    
        # G has shape (yn, xn+1), take derivative in y direction at pressure points
        deriv_G = np.zeros((self.yn + 1, self.xn + 1))
        deriv_G[1:-1, :] = (G[1:, :] - G[:-1, :]) / self.delta_y
        deriv_G[0, :] = (G[0, :] - 0) / self.delta_y      # assume G=0 at bottom
        deriv_G[-1, :] = (0 - G[-1, :]) / self.delta_y    # assume G=0 at top
    
        
        RHS = (1 / delta_t) * (deriv_F + deriv_G)
    
        # Convert RHS-array to P-grid since it referred to actual grid:
        #RHS = conv.P_like_from_grid(RHS)

        P_0_norm: float = norm_L2(self.P)
        residual_norm: float = self.epsilon * P_0_norm + 1
        n = 0
        
        #while residual_norm >= self.epsilon * P_0_norm and n < N_max:
            # P_new: np.ndarray = np.zeros_like(P_it)
            
            # P_sum: np.ndarray = (P_it[1:-1, 2:] + P_it[1:-1, :-2]) / self.delta_x**2 + (P_it[2:, 1:-1] + P_it[:-2, 1:-1]) / self.delta_y**2
            # P_new[1:-1, 1:-1] = (1 - self.omega) * P_it[1:-1, 1:-1] + self.omega / (2 * (1 / self.delta_x**2 + 1 / self.delta_y**2)) * (P_sum - RHS)
            
            # # Set boundary values:
            # P_new[0, :], P_new[-1, :] = P_it[1, :], P_it[-2, :]
            # P_new[:, 0], P_new[:, -1] = P_it[:, 1], P_it[:, -2]
            
            # residual: np.ndarray = deriv.lin_x(P_new, self.delta_x, 2) + deriv.lin_y(P_new, self.delta_y, 2) - RHS
            # residual_norm = norm_L2(residual)
            
            # P_it = P_new
            # n = n + 1

        P_0_norm = norm_L2(self.P)
        if P_0_norm == 0.0:
            P_0_norm = 1.0

        self.P = sor_solver_fast(
            self.P.copy(),
            RHS,
            self.omega,
            self.delta_x,
            self.delta_y,
            self.yn+1,
            self.xn+1,
            N_max,
            self.epsilon,
            P_0_norm
            )

    
    def apply_boundary_condition(self, side: Literal['left', 'right', 'top', 'bottom'],
                                 U_val: float = 0, V_val: float = 0,
                                 start_val=0, end_val=-1, terminate_rekursion=False) -> None:
        a, b = start_val, end_val  # shorthand
        
        # trick to first apply normal boundary condition to other parts
        # for some reason the simulation worked without this part. Not sure if that was coincidence, but to be sure added this part.
        if not terminate_rekursion:   
            self.apply_boundary_condition(side, 0, 0, 0, a, True) 
            self.apply_boundary_condition(side, 0, 0, b, -1, True) 

        if side == 'left':
            self.U[a:b, 0] = U_val
            self.V[a:b, 0] = V_val*2 - self.V[a:b, 1]
        elif side == 'right':
            self.U[a:b, -1] = U_val
            self.V[a:b, -1] = V_val*2 - self.V[a:b, -2]
        elif side == 'top':
            self.U[0, a:b] = U_val*2 - self.U[1, a:b]
            self.V[0, a:b] = V_val
        elif side == 'bottom':
            self.U[-1, a:b] = U_val*2 - self.U[-2, a:b]
            self.V[-1, a:b] = V_val
        else:
            raise ValueError("side must be one of 'left', 'right', 'top', 'bottom'")

    def apply_boundary_conditions(self) -> None:
        if self.boundary_type == "lid":
            self.apply_boundary_condition('left')
            self.apply_boundary_condition('right')
            self.apply_boundary_condition('top', U_val=self.x_vel)
            self.apply_boundary_condition('bottom')
        elif self.boundary_type == "lid_floor":
            self.apply_boundary_condition('left')
            self.apply_boundary_condition('right')
            self.apply_boundary_condition('top', U_val=self.x_vel)
            self.apply_boundary_condition('bottom', U_val=self.x_vel)
        elif self.boundary_type == "channel":
            self.apply_boundary_condition('left', U_val=self.x_vel)
            self.apply_boundary_condition('right', U_val=self.x_vel)
            self.apply_boundary_condition('top')
            self.apply_boundary_condition('bottom')
        elif self.boundary_type == "s_channel":
            self.apply_boundary_condition('left', U_val=self.x_vel, start_val=int(self.yn/2))
            self.apply_boundary_condition('right', U_val=self.x_vel, end_val=int(self.yn/2))
            self.apply_boundary_condition('top')
            self.apply_boundary_condition('bottom')
        elif self.boundary_type == "voided_channel":
            self.apply_boundary_condition('left', U_val=self.x_vel)
            self.apply_boundary_condition('right', U_val=self.x_vel)
            self.apply_boundary_condition('top', U_val=self.x_vel)
            self.apply_boundary_condition('bottom', U_val=self.x_vel)
        for box in self.boxes:
            self.apply_box_boundary(box_start_x=int(box[0]), box_end_x=int(box[1]),
                                    box_start_y=int(box[2]), box_end_y=int(box[3]))

    def apply_box_boundary(self, box_start_x: int, box_end_x: int, box_start_y: int, box_end_y: int) -> None:
        # Apply no-slip boundary conditions around a rectangular box defined by the given grid indices.
        
        print("Applying box boundary:", box_start_x, box_end_x, box_start_y, box_end_y)
        box_end_x = box_end_x
        box_end_y = box_end_y 
        box_start_x = box_start_x
        box_start_y = box_start_y
        self.U[box_start_y:box_end_y, box_start_x:box_end_x] = 0  # Left side
        self.V[box_start_y:box_end_y, box_start_x:box_end_x] = 0  # Top side

    def iterate(self, t_end: float, N_max_P: int = 100) -> None:
        # Print a message at certain timesteps to track progress:
        print_times: np.ndarray = np.linspace(0, t_end, 20)
        print_index: int = 0

        t: float = 0
        while t < t_end:
            self.apply_boundary_conditions()
            delta_t: float = self.calc_timestep()

            F, G = self.calc_F_and_G(delta_t)
            self.calc_pressure(F, G, delta_t, N_max_P)

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
    
    def sparcify_history(self, factor: int) -> None:
        self.u_history = self.u_history[::factor]
        self.v_history = self.v_history[::factor]
        self.p_history = self.p_history[::factor]
        self.t_history = self.t_history[::factor]
        self.sparceify_factor = factor

    def keys(self) -> list:
        """Return list of attribute names for dict-like access compatibility."""
        return list(self.__dict__.keys())

    def save(self, filename: str) -> None:
        """Save the complete simulation object to a file using pickle."""
        # check file does not already exist. If yes, prompt user for overwrite permission
        if os.path.exists(filename):
            response = input(f"File {filename} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Save operation cancelled.")
                return
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


if __name__ == '__main__':
    # permanent values:
    tau: float = 1
    omega: float = 1.7
    epsilon: float = 0.01
    x_vel: float = 2
    N_max_P: int = 100

    # variables:
    nx: int = 60
    ny: int = 60
    len_x: float = 1
    len_y: float = 1
    Re: float = 4000
    T_max: float = 20
    type_: str = "lid"
    boxes: list = [[nx/4, nx*2/4, ny*3/8, ny*5/8]]  # list of boxes defined by [start_x, end_x, start_y, end_y] in grid indices
    addon: str = "box3"  # for filename uniqueness

    """
    current types:
    "lid"      : standard lid-driven cavity
    "lid_floor": both top and bottom walls move with lid velocity
    "channel"  : left and right walls move with x_vel velocity
    "s_channel": left wall moves with x_vel in upper half, right wall moves with x_vel in lower half
    "voided_channel": all four walls move with x_vel velocity
    """
    
    filename: str = f"{type_}_nx{nx}_ny{ny}_re{Re}_t{int(T_max*1000)}{addon}.pkl"
    # Check if file exists
    if os.path.exists(filename):
        print(f"Loading simulation from {filename}...")
        with open(filename, 'rb') as f:
            simulation = pickle.load(f)
    else:
        print("Running new simulation...")
        simulation = navier_stokes_simulation(nx, ny, len_x, len_y, x_vel, Re, tau, omega, epsilon)
        simulation.set_boundary_type_and_boxes(type_, boxes)
        simulation.iterate(T_max, N_max_P=N_max_P)
        simulation.sparcify_history(factor=5)  # save every 10th timestep only
        simulation.save(filename)
        print(f"Simulation saved to {filename}")
    
    plot_log_vel = True # False # enable logarithmic scaling of velocity vectors
    quiver_scale = 30   # 8     # adjust length of plotted arrows (smaller -> longer)
    animation = plot.animate_simulation(simulation, quiver_scale, plot_log_vel, frame_skip=2, arrow_skip=2, plot_field="pressure")
    # plot.streamlines_and_magnitudes(simulation, [T_max], [len_x, len_y])
