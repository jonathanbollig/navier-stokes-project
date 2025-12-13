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
    count: int = 0

    while t <= end_time:        
        Pi_change = c**2 * ((U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / step**2
                            + (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / step**2)
        
        Pi = Pi + timestep * Pi_change
        U[1:-1, 1:-1] = U[1:-1, 1:-1] + timestep * Pi
        
        # if count % int(end_time / (100 * timestep)) == 0:
        U_sols.append(U[1:-1, 1:-1].copy())
    
        t = t + timestep
        count = count + 1
    
    return coords, U_sols
    
    

def main() -> None:
    def initial_state(L: float, x: float, y: float) -> float:
        return np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
        
    
    L: float = 1
    c: float = 1
    step: float = 0.005
    timestep: float = 0.001
    end_time: float = 2**(1 / 2) * c / L * 2
    boundary_values: list[float] = [0, 0, 0, 0]
    
    coords, U_sols = solve_wave_equation(L, step, timestep, end_time, boundary_values,
                                         lambda x, y: initial_state(L, x, y), lambda x, y: 0, c)
    
    U_sols_plot: np.array = U_sols[::(int(end_time / (100 * timestep)))]
    
    X, Y = np.meshgrid(coords, coords)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    
    ax.set_zlim(-1, 1)
    
    ax.plot_surface(X, Y, U_sols_plot[0], cmap = 'viridis')
    
    def update(frame: int):
        ax.clear()
        
        U: np.array = U_sols_plot[frame]
        surf = ax.plot_surface(X, Y, U, cmap = 'viridis')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        
        ax.set_zlim(-1, 1)
        
        return surf,
        
    
    animation = FuncAnimation(fig, update, frames = len(U_sols_plot), interval = 10, blit = False)
    
    plt.show()
    
    return animation
    
    
    
if __name__ == '__main__':
    animation = main()