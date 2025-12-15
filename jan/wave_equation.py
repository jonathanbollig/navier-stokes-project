# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 12:41:16 2025

@author: reich
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np



def solve_wave_equation(L: float, step: float, timestep: float, end_time: float,
                        boundary_values: list[float], initial_state: callable, 
                        initial_deriv: callable, c: float = 1) -> np.array:
    def initialize_solutions(U: np.array, Pi: np.array, boundary_values: list[float], 
                             initial_state: callable, initial_deriv: callable, 
                             coords: np.array) -> np.array:
        U[0, :] = boundary_values[0]
        U[:, -1] = boundary_values[1]
        U[-1, :] = boundary_values[2]
        U[:, 0] = boundary_values[3]
        
        for i in range(N):
            for j in range(N):
                U[j + 1, i + 1] = initial_state(coords[i], coords[j])
                Pi[j, i] = initial_deriv(coords[i], coords[j])
        
    
    N: int = int(np.round(L / step) + 1)
    coords: np.array = np.linspace(0, L, N)
    
    U: np.array = np.zeros(shape = (N + 2, N + 2))
    U_sols: list[np.array] = []
    
    Pi: np.array = np.zeros(shape = (N, N))
    Pi_change: np.array = np.zeros(shape = (N, N))
    
    initialize_solutions(U, Pi, boundary_values, initial_state, initial_deriv, coords)
    t: float = 0

    while t <= end_time:        
        Pi_change = c**2 * ((U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / step**2
                            + (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / step**2)
        
        Pi = Pi + timestep * Pi_change
        U[1:-1, 1:-1] = U[1:-1, 1:-1] + timestep * Pi
        
        U_sols.append(U[1:-1, 1:-1].copy())
    
        t = t + timestep
    
    return coords, U_sols


def plot_solutions(X_num: np.array, Y_num: np.array, U_num_plot: list[np.array],
                   X_ana: np.array = None, Y_ana: np.array = None, U_ana_plot: list[np.array] = None):
    fig = plt.figure(figsize = (10, 6))
    ax_num = fig.add_subplot(1, 2, 1, projection = '3d')
    
    surf_num = ax_num.plot_surface(X_num, Y_num, U_num_plot[0], cmap = 'viridis')
    
    ax_num.set_xlabel('x')
    ax_num.set_ylabel('y')
    ax_num.set_zlabel('u')
    
    ax_num.set_title(f'Numerical solution with {X_num.shape[0]} by {X_num.shape[0]} grid points')
    
    ax_num.set_zlim(-1, 1)
    
    if U_ana_plot != None:
        ax_ana = fig.add_subplot(1, 2, 2, projection = '3d')
        
        surf_ana = ax_ana.plot_surface(X_ana, Y_ana, U_ana_plot[0], cmap = 'viridis')
        
        ax_ana.set_xlabel('x')
        ax_ana.set_ylabel('y')
        ax_ana.set_zlabel('u')
        
        ax_ana.set_title('Analytical solution')
        
        ax_ana.set_zlim(-1, 1)
    
    def update(frame: int):
        nonlocal surf_num
        
        surf_num.remove()
        
        U_num: np.array = U_num_plot[frame]
        surf_num = ax_num.plot_surface(X_num, Y_num, U_num, cmap = 'viridis')
        
        if U_ana_plot != None:
            nonlocal surf_ana
            
            surf_ana.remove()
            
            U_ana: np.array = U_ana_plot[frame]
            surf_ana = ax_ana.plot_surface(X_ana, Y_ana, U_ana, cmap = 'viridis')
        
            return surf_num, surf_ana
        
        return surf_num
        
    
    animation = FuncAnimation(fig, update, frames = len(U_num_plot), interval = 25, blit = False)
    
    plt.tight_layout(pad = 3.0)
    plt.show()
    
    return animation
    
    

def main() -> None:
    def analytical_solution(t: float, L: float, c: float) -> np.array:
        coords: np.array = np.linspace(0, L, 101)
        X, Y = np.meshgrid(coords, coords)
        
        return np.sin(np.pi * X / L) * np.sin(np.pi * Y / L) * np.cos(2**(1 / 2) * np.pi * c * t / L)
    
    
    def initial_state(x: float, y: float, L: float) -> float:
        return np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
        
    
    L: float = 1
    c: float = 1
    step: float = 0.005
    timestep: float = 0.001
    end_time: float = 2**(1 / 2) * c / L * 2
    boundary_values: list[float] = [0, 0, 0, 0]
    
    coords, U_num = solve_wave_equation(L, step, timestep, end_time, boundary_values,
                                         lambda x, y: initial_state(x, y, L), lambda x, y: 0, c)
    
    U_num_plot: np.array = U_num[::(int(end_time / (100 * timestep)))]
    U_ana_plot: list[np.array] = [analytical_solution(i *  end_time / 100, L, c) for i in range(len(U_num_plot))]

    X_num, Y_num = np.meshgrid(coords, coords)
    X_ana, Y_ana = np.meshgrid(np.linspace(0, L, 101), np.linspace(0, L, 101))
    
    animation = plot_solutions(X_num, Y_num, U_num_plot, X_ana, Y_ana, U_ana_plot)
    
    return animation
    
    
if __name__ == '__main__':
    animation = main()