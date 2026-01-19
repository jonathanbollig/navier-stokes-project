# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 19:52:56 2026

@author: Jan
"""

from matplotlib import colormaps
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from typing import Any, Optional

from navier_stokes import navier_stokes_simulation
import conversions as conv

def draw_boxes(ax: plt.Axes, sim: navier_stokes_simulation) -> None:  # type: ignore
    """
    Draw black rectangles on the axes for each box in sim.boxes.
    
    Parameters:
    -----------
    ax : plt.Axes
        The matplotlib axes to draw on
    sim : navier_stokes_simulation
        The simulation object containing boxes and grid information
    """
    if not hasattr(sim, 'boxes') or not sim.boxes:
        return
    
    # Convert grid indices to physical coordinates
    dx = sim.len_x / sim.xn
    dy = sim.len_y / sim.yn
    
    for box in sim.boxes:
        # box format: [start_x, end_x, start_y, end_y] in grid indices
        x_start = box[0] * dx
        x_end = box[1] * dx
        y_start = box[2] * dy
        y_end = box[3] * dy
        
        width = x_end - x_start
        height = y_end - y_start
        
        # Add a black filled rectangle
        rect = plt.Rectangle((x_start, y_start), width, height,   # type: ignore
                            facecolor='black', edgecolor='black', 
                            linewidth=0, zorder=10)
        ax.add_patch(rect)

def animate_simulation(sim: navier_stokes_simulation, 
                     quiver_scale: Optional[float] = None, plot_log_vel: bool = False,
                     log_vel_exp: float = 2, frame_skip: Optional[int] = None, save: Optional[str] = None,
                     plot_field: str = 'velocity', arrow_skip: Optional[int] = None) -> ani.FuncAnimation:
    if frame_skip is None:
        # frame_skip: int = len(solutions[0]) / 100
        frame_skip = 20
    
    # Initialize solution arrays:
    U_sol = sim.u_history[::frame_skip]
    V_sol = sim.v_history[::frame_skip]
    P_sol = sim.p_history[::frame_skip]
    t_sol = sim.t_history[::frame_skip]
    
    # Compute velocity magnitude arrays:
    M_sol = []
    
    for i in range(len(U_sol)):
        U_sol[i] = conv.U_like_to_grid(U_sol[i])
        V_sol[i] = conv.V_like_to_grid(V_sol[i])
        P_sol[i] = conv.P_like_to_grid(P_sol[i])
        
        # Calculate velocity magnitude:
        M_sol.append(np.sqrt(U_sol[i]**2 + V_sol[i]**2))
        # Calculate velocity magnitude:
        M_sol.append(np.sqrt(U_sol[i]**2 + V_sol[i]**2))
        
        # Convert velocity vectors to logarithmic scale for better visualization:
        if plot_log_vel:
            
            M: np.ndarray = np.sqrt(U_sol[i]**2 + V_sol[i]**2)
            
            # Add small value epsilon to avoid log(0):
            epsilon = 1e-10
            log_M: np.ndarray = np.log2(M + epsilon)
            
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
    ax.set_aspect('equal', adjustable='box')  # maintain aspect ratio
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Grid: ({N_x}, {N_y})\n" + f"t = {np.round(t_sol[0], 3)}")

    # Choose field to plot based on plot_field parameter:
    if plot_field == 'pressure':
        field_data = P_sol
        cmap = "seismic"
        field_label = 'Pressure'
    else:  # default to velocity magnitude
        field_data = M_sol
        cmap = "viridis"
        field_label = 'Velocity Magnitude'

    # Initial image plot for the chosen field:
    im = ax.imshow(np.flipud(field_data[0]), extent = (0, sim.len_x, 0, sim.len_y), origin = 'lower', cmap=cmap)

    # Initial quiver plot for velocity field:
    if arrow_skip is None:
        skip = max(1, int(np.round(N_x / 25)))  # show more arrows by default
    else:
        skip = arrow_skip
    quiv = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], U_sol[0][::skip, ::skip], -V_sol[0][::skip, ::skip], 
                     color = 'black', scale_units = 'xy', scale = quiver_scale)
    
    # Colorbar for the plotted field:
    cbar = fig.colorbar(im, ax = ax)
    cbar.set_label(field_label)
    
    # Draw boxes (obstacles/boundaries):
    draw_boxes(ax, sim)

    def update(frame):        
        im.set_data(field_data[frame])
        quiv.set_UVC(U_sol[frame][::skip, ::skip], -V_sol[frame][::skip, ::skip])
        
        if frame % 5 == 0:
            ax.set_title(f"Grid: ({N_x}, {N_y})\n" + f"t = {np.round(t_sol[frame], 2)}")
        
        return [im, quiv]
    
    animation = ani.FuncAnimation(fig, update, frames = len(U_sol), interval = 50)
    
    if save is not None:
        animation.save(save, writer='pillow', fps=20)
        print(f"Animation saved to {save}")
    
    plt.show()
    
    return animation


def streamlines_and_magnitudes(sim: navier_stokes_simulation, plot_times: list[float],
                               domain_size: list[float], plot_params: dict = {}, 
                               save_params: Optional[dict] = None) -> None:
    # Initialize solution arrays:
    U_sol = sim.u_history
    V_sol = sim.v_history
    t_sol = sim.t_history
    
    for i in range(len(U_sol)):
        U_sol[i] = conv.U_like_to_grid(U_sol[i])
        V_sol[i] = conv.V_like_to_grid(V_sol[i])
    
    # Find the indices closest to the requested plot times:
    t_sol_array = np.array(t_sol)
    plot_time_indices = [np.argmin(np.abs(t_sol_array - t)) for t in plot_times]
        
    # Plot streamlines and velocity magnitudes of all requested time steps:    
    a, b = domain_size
    N_x, N_y = U_sol[0].shape
    
    X, Y = np.meshgrid(np.linspace(0, a, N_x), np.linspace(0, b, N_y))
    for plot_index in plot_time_indices:
        U: np.ndarray = U_sol[plot_index]
        V: np.ndarray = V_sol[plot_index]
        t: float = t_sol[plot_index]
        
        M: np.ndarray = np.sqrt(np.square(U) + np.square(V))
        
        fig, ax = plt.subplots(figsize = plot_params.get('figsize', (7, 7)))
        ax.set_aspect('equal', adjustable='box')  # maintain aspect ratio
        
        # Contour plot of velocity magnitude:
        print(X.shape, Y.shape, M.shape)
        contour = ax.contourf(X, Y, M, 
                     levels = plot_params.get('contour levels', 50), 
                     cmap = plot_params.get('cmap', colormaps['jet']))
        
        # Add colorbar for velocity magnitude:
        fig.colorbar(contour, ax=ax, label='Velocity Magnitude')
        
        # Streamplot of stream lines:
        ax.streamplot(X, Y, U, V, 
                       color = plot_params.get('streamline color', 'white'), 
                       density = plot_params.get('streamline density', 1.5), 
                       linewidth = plot_params.get('streamline linewidth', 0.7))
        
        ax.set_xlim(0, a)
        ax.set_ylim(b, 0) # invert y-limits so that plot is right side up
        
        ax.set_title(f"Grid: ({N_x}, {N_y})\n" + f"t = {t:.2f}")
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Draw boxes (obstacles/boundaries):
        draw_boxes(ax, sim)
        
        if save_params != None:
            title: str = save_params['title'] + f'_t{t:.1f}.png'
            plt.savefig(title, dpi = save_params.get('dpi', 200))
        
        plt.show()