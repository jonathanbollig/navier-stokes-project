# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 12:41:16 2025

@author: reich & Jonathan
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def solve_wave_equation(step: float, timestep: float, end_time: float, initial_state: np.array, 
                        initial_deriv: np.array, c: float = 1) -> np.array:

    U: np.array = initial_state
    Pi: np.array = initial_deriv
    Pi_change: np.array = np.zeros_like(Pi)
    
    U_sols: list[np.array] = []
    for _ in range(int(end_time/timestep)):
        Pi_change = c**2 * ((U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / step**2
                            + (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / step**2)
        
        Pi = Pi + timestep * Pi_change
        U[1:-1, 1:-1] = U[1:-1, 1:-1] + timestep * Pi
        U_sols.append(U[1:-1, 1:-1].copy())

    return np.array(U_sols)
    
def initial_state(L: float, x: np.array, y: np.array, boundary_values: list[float]) -> np.array:
    U = np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
    U = np.pad(U, 1, constant_values=((boundary_values[0], boundary_values[1]), (boundary_values[2], boundary_values[3])))
    return U

def animate_U(U_sols, X, Y):    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_zlim(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    
    ax.plot_surface(X, Y, U_sols[0], cmap = 'viridis')
    
    def update(frame: int):
        ax.clear()
        
        U: np.array = U_sols[frame]
        surf = ax.plot_surface(X, Y, U, cmap = 'viridis')        
        ax.set_zlim(-1, 1)    
        return surf,
    
    animation = FuncAnimation(fig, update, frames = len(U_sols), interval = 10, blit = False)
    plt.show()   
    return animation

def main() -> None:
    L: float = 1
    c: float = 1
    step: float = 0.005
    timestep: float = 0.001
    end_time: float = 2**(1 / 2) * c / L * 2
    boundary_values: list[float] = [0, 0, 0, 0]
    
    N: int = int(np.round(L / step) + 1)
    coords: np.array = np.linspace(0, L, N)
    X, Y = np.meshgrid(coords, coords)

    U_init: np.array = initial_state(L, X, Y, boundary_values)
    dU_dt_init: np.array = np.zeros(shape=(N, N))

    U_sols = solve_wave_equation(step, timestep, end_time, U_init, dU_dt_init, c)
    U_sols_plot: np.array = U_sols[::(int(end_time / (100 * timestep)))]
    animate_U(U_sols_plot, X, Y) 
    
if __name__ == '__main__':
    animation = main()