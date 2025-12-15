# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 12:41:16 2025

@author: reich & Jonathan
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import plotly.graph_objects as go
import matplotlib.animation as animation

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
    
def wavefunc(X, Y, L, c, t):
    return np.sin(np.pi * X / L) * np.sin(np.pi * Y / L) * np.cos(2**(1 / 2) * np.pi * c * t / L)

def initial_state(L: float, x: np.array, y: np.array, boundary_values: list[float], c: float = 1) -> np.array:
    U = wavefunc(x, y, L, c, 0)
    U = np.pad(U, 1, constant_values=((boundary_values[0], boundary_values[1]), (boundary_values[2], boundary_values[3])))
    return U

def animate_U(U_sols, X, Y, U_ana=None, r_c_stride = 10):
    """
    Disclaimer: Slight refactor by Gemini, most work done by students
    """

    if U_ana is not None:
        fig = plt.figure(figsize=(10, 4))
        ax_num = fig.add_subplot(1, 2, 1, projection='3d')
        ax_ana = fig.add_subplot(1, 2, 2, projection='3d')
    else:
        fig = plt.figure(figsize=(6, 4))
        ax_num = fig.add_subplot(111, projection='3d')
        ax_ana = None

    ax_num.set_title('Numerical solution')
    ax_num.set_xlabel('x')
    ax_num.set_ylabel('y')
    ax_num.set_zlabel('u')
    ax_num.set_zlim(-1, 1)

    if ax_ana is not None:
        ax_ana.set_title('Analytical solution')
        ax_ana.set_xlabel('x')
        ax_ana.set_ylabel('y')
        ax_ana.set_zlabel('u')
        ax_ana.set_zlim(-1, 1)

    ims = []
    r_c_stride = 5

    if U_ana is not None:
        ax_ana = fig.add_subplot(1, 2, 2, projection = '3d')
        
        ax_ana.set_xlabel('x')
        ax_ana.set_ylabel('y')
        ax_ana.set_zlabel('u')
        
        ax_ana.set_title('Analytical solution')
        
        ax_ana.set_zlim(-1, 1)

    ims = []

    if U_ana is not None:
        for U_num, U_exact in zip(U_sols, U_ana):
            surf_num = ax_num.plot_surface(
                X, Y, U_num,
                cmap='viridis',
                rstride=r_c_stride,
                cstride=r_c_stride
            )

            surf_ana = ax_ana.plot_surface(
                X, Y, U_exact,
                cmap='viridis',
                rstride=r_c_stride,
                cstride=r_c_stride
            )

            ims.append([surf_num, surf_ana])
    else:
        for U_num in U_sols:
            surf_num = ax_num.plot_surface(
                X, Y, U_num,
                cmap='viridis',
                rstride=r_c_stride,
                cstride=r_c_stride
            )

            ims.append([surf_num])

    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=False, repeat_delay=1000)
    
    plt.show()
    return ani

def animate_U_plotly(U_sols, X, Y, frames_per_second=20, divider: int = 10):
    """
    Disclaimer: This was mainly written by Gemini, and eddited by Jonathan
    """
    U_sols = U_sols[:, ::divider, ::divider]
    X = X[::divider, ::divider]
    Y = Y[::divider, ::divider]
    # Create the data structure for the first frame
    data = [go.Surface(z=U_sols[0], x=X, y=Y, colorscale='Viridis')]
    
    # Create frames
    frames = []
    for U in U_sols:
        frame = go.Frame(data=[go.Surface(z=U, x=X, y=Y, colorscale='Viridis')],
                         name=str(len(frames)))
        frames.append(frame)

    # Create the layout with animation controls
    layout = go.Layout(
        scene=dict(zaxis=dict(range=[-1, 1])),
        updatemenus=[{
            'buttons': [
                {'args': [None, {'frame': {'duration': 1000/frames_per_second, 'redraw': True},
                                 'fromcurrent': True, 'transition': {'duration': 0}}],
                 'label': 'Play',
                 'method': 'animate'}
            ],
            'type': 'buttons',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        # ... add more layout configs (titles, etc.)
    )

    fig = go.Figure(data=data, layout=layout, frames=frames)
    fig.show() # Opens in your browser/notebook

if __name__ == '__main__':
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
    times = np.linspace(0, int(end_time/timestep)*timestep, int(end_time/timestep))
    times_plot = times[::(int(end_time / (100 * timestep)))]
    U_analytical = wavefunc(X[np.newaxis, :, :], Y[np.newaxis, :, :], L, c, times_plot[:, np.newaxis, np.newaxis])
    animate_U(U_sols_plot, X, Y, U_analytical) 
    # animate_U_plotly(U_sols_plot, X, Y)
