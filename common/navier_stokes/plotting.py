# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 19:52:56 2026

@author: Jan
"""

import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from navier_stokes2 import navier_stokes_simulation
import conversions as conv

def animate_simulation(sim: navier_stokes_simulation, 
                     quiver_scale: float = None, plot_log_vel: bool = False,
                     log_vel_exp: float = 2, frame_skip: int = None) -> ani.FuncAnimation:
    if frame_skip == None:
        # frame_skip: int = len(solutions[0]) / 100
        frame_skip = 20
    
    # Initialize solution arrays:
    U_sol = sim.u_history[::frame_skip]
    V_sol = sim.v_history[::frame_skip]
    P_sol = sim.p_history[::frame_skip]
    t_sol = sim.t_history[::frame_skip]
    
    for i in range(len(U_sol)):
        U_sol[i] = conv.U_like_to_grid(U_sol[i])
        V_sol[i] = conv.V_like_to_grid(V_sol[i])
        P_sol[i] = conv.P_like_to_grid(P_sol[i])
        
        # Convert velocity vectors to logarithmic scale for better visualization:
        if plot_log_vel:
            M: np.array = np.sqrt(U_sol[i]**2 + V_sol[i]**2)
            
            # Add small value epsilon to avoid log(0):
            epsilon = 1e-10
            log_M: np.array = np.log2(M + epsilon)
            
            # Normalize log magnitudes to a positive scale for better visualization:
            log_M_norm = (log_M - log_M.min()) / (log_M.max() - log_M.min())
            
            # Apply power law to amplify difference between longest and shortest arrows:
            gamma: float = log_vel_exp
            
            # Scale U and V by normalized log magnitude keeping direction:
            U_sol[i] = (U_sol[i] / (M + epsilon)) * log_M_norm**gamma
            V_sol[i] = (V_sol[i] / (M + epsilon)) * log_M_norm**gamma
    
    N_y, N_x = P_sol[0].shape
    
    X, Y = np.meshgrid(np.linspace(0, sim.len_x, N_x), np.linspace(0, sim.len_y, N_y))
    
    fig, ax = plt.subplots(figsize = (7, 7))
    ax.set_xlim(0, sim.len_x)
    ax.set_ylim(sim.len_y, 0) # invert y-limits so that plot is right side up
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Grid: ({N_x}, {N_y})\n" + f"t = {np.round(t_sol[0], 3)}")

    # Initial image plot for pressure field:
    im = ax.imshow(np.flipud(P_sol[0]), extent = (0, sim.len_x, 0, sim.len_y), origin = 'lower')

    # Initial quiver plot for velocity field:
    skip = int(np.round(3 * N_x / 50))  # reduce number of arrows for clarity and performance
    quiv = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], U_sol[0][::skip, ::skip], -V_sol[0][::skip, ::skip], 
                     color = 'white', scale_units = 'xy', scale = quiver_scale)
    
    # Colorbar for pressure values:
    cbar = fig.colorbar(im, ax = ax)
    cbar.set_label('Pressure')

    def update(frame):        
        im.set_data(P_sol[frame])
        quiv.set_UVC(U_sol[frame][::skip, ::skip], -V_sol[frame][::skip, ::skip])
        
        if frame % 5 == 0:
            ax.set_title(f"Grid: ({N_x}, {N_y})\n" + f"t = {np.round(t_sol[frame], 2)}")
        
        return [im, quiv]
    
    animation = ani.FuncAnimation(fig, update, frames = len(U_sol), interval = 50)
    
    plt.show()
    
    return animation