# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 19:52:56 2026

@author: Jan
"""

from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from typing import Any, Optional, TYPE_CHECKING

plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=17)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize

"""
When calling navier_stokes from another file, it would throw an error due to circular imports.
To avoid this ChatGPT suggested using TYPE_CHECKING to only import for type hints. This fixed the issue.
"""
if TYPE_CHECKING:
    from navier_stokes import navier_stokes_simulation

import conversions as conv
def draw_circles(ax: plt.Axes, sim) -> None:
    """
    Draw black circles on the axes for each circle obstacle in sim.circle.
    
    Parameters:
    -----------
    ax : plt.Axes
        The matplotlib axes to draw on
    sim : navier_stokes_simulation
        The simulation object containing circles and grid information
    """
    if not hasattr(sim, 'circle') or not sim.circle:
        return
    # Convert grid indices to physical coordinates
    dx = sim.len_x / sim.xn
    dy = sim.len_y / sim.yn
    
    for circle in sim.circle:
        x_center = circle[0] * dx
        y_center = circle[1] * dy
        radius = circle[2] * dx  # Convert grid units to physical units
        
        # Add a red circle
        circ = plt.Circle((x_center, y_center), radius,
                         facecolor='black', edgecolor='black',
                         linewidth=0, zorder=10)
        ax.add_patch(circ)

def draw_boxes(ax: plt.Axes, sim: "navier_stokes_simulation") -> None:  # type: ignore
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

def animate_simulation(sim: "navier_stokes_simulation", 
                     quiver_scale: Optional[float] = None, plot_log_vel: bool = True,
                     log_vel_exp: float = 2, frame_skip: Optional[int] = None, save: Optional[str] = None,
                     plot_field: str = 'velocity', arrow_skip: Optional[int] = None,
                     v_max: Optional[float] = None) -> ani.FuncAnimation:
    if frame_skip is None:
        # frame_skip: int = len(solutions[0]) / 100
        frame_skip = 1
    
    # Initialize solution arrays (deep copy to avoid modifying original history):
    U_sol = sim.u_history[::frame_skip]
    V_sol = sim.v_history[::frame_skip]
    P_sol = sim.p_history[::frame_skip]
    t_sol = sim.t_history[::frame_skip]

    for i in range(len(U_sol)):
        U_sol[i] = conv.U_like_to_grid(U_sol[i])
        V_sol[i] = conv.V_like_to_grid(V_sol[i])
        P_sol[i] = conv.P_like_to_grid(P_sol[i])

    U_sol = np.array(U_sol)
    V_sol = np.array(V_sol)
    P_sol = np.array(P_sol)
    t_sol = np.array(t_sol)
    M_sol = np.sqrt(U_sol**2 + V_sol**2)
    
    if v_max is not None:
        M_sol = np.clip(M_sol, 0, v_max)
            
    if plot_log_vel:
        # Add small value epsilon to avoid log(0):
        epsilon = 1e-1
        log_M: np.ndarray = np.log2(M_sol + epsilon)
        # Normalize log magnitudes to a positive scale for better visualization:
        log_M_norm = (log_M - log_M.min()) / (log_M.max() - log_M.min())
        
        # Apply power law to amplify difference between longest and shortest arrows:
        gamma: float = log_vel_exp
        
        # Scale U and V by normalized log magnitude keeping direction:
        U_sol = (U_sol / (M_sol + epsilon)) * log_M_norm**gamma
        V_sol = (V_sol / (M_sol + epsilon)) * log_M_norm**gamma

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
    im = ax.imshow(np.flipud(field_data[0]), extent = (0, sim.len_x, 0, sim.len_y), origin = 'lower', cmap=cmap, vmin=0, vmax=v_max)

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
    draw_circles(ax, sim)

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
    
    # Show the animation and return it. Do not stop/close the animation/figure
    # immediately: closing the figure prevents the notebook widget backend from
    # rendering the animation. Let the notebook retain the figure so the
    # interactive backend can display and control the animation.
    plt.show()
    return animation


def streamlines_and_magnitudes(sim: "navier_stokes_simulation", plot_times: Optional[list[float]] = None,
                               domain_size: Optional[list[float]] = None, plot_params: dict = {}, 
                               save_params: Optional[dict] = None, 
                               v_max = None, plot=True, colorbar=False, 
                               correct_u=0) -> None:
    # Initialize solution arrays (deep copy to avoid modifying original history):
    U_sol = [u.copy() for u in sim.u_history]
    V_sol = [v.copy() for v in sim.v_history]
    t_sol = sim.t_history
    
    for i in range(len(U_sol)):
        U_sol[i] = conv.U_like_to_grid(U_sol[i])
        V_sol[i] = conv.V_like_to_grid(V_sol[i])
    
    # Find the indices closest to the requested plot times:
    t_sol_array = np.array(t_sol)
    if plot_times is None:
        plot_time_indices = [len(t_sol)-1]
    else:
        plot_time_indices = [np.argmin(np.abs(t_sol_array - t)) for t in plot_times]
        
    # Plot streamlines and velocity magnitudes of all requested time steps:    
    if domain_size is None:
        a: float = sim.len_x
        b: float = sim.len_y
    else:
        a: float = domain_size[0]
        b: float = domain_size[1]
    N_x, N_y = U_sol[0].shape
    
    X, Y = np.meshgrid(np.linspace(0, a, N_x), np.linspace(0, b, N_y))
    for plot_index in plot_time_indices:
        U: np.ndarray = U_sol[plot_index]
        V: np.ndarray = V_sol[plot_index]
        t: float = t_sol[plot_index]
        U = U-correct_u

        M: np.ndarray = np.sqrt(np.square(U) + np.square(V))
        if v_max is not None:
            M = np.clip(M, 0, v_max)
        
        if colorbar:
            fig, ax = plt.subplots(figsize = plot_params.get('figsize', (8, 7)))
        else:
            fig, ax = plt.subplots(figsize = plot_params.get('figsize', (7, 7)))
        ax.set_aspect('equal', adjustable='box')  # maintain aspect ratio
        
        # Contour plot of velocity magnitude:
        bounds = np.linspace(0, 2, plot_params.get('contour levels', 200))
        print(X.shape, Y.shape, M.shape)
        contour = ax.contourf(X, Y, M, 
                     levels = bounds, 
                     cmap = plot_params.get('cmap', colormaps['jet']), vmax = v_max)
        
        if colorbar:
            # Add colorbar for velocity magnitude:
            divider = make_axes_locatable(ax)
            tick_marks = np.arange(0, 2.1, 0.2)
            cax = divider.append_axes("right", size="5%", pad=0.5)
            fig.colorbar(contour, cax=cax, label='Velocity Magnitude', ticks=tick_marks)
        
        # Streamplot of stream lines:
        ax.streamplot(X, Y, U, V, 
                       color = plot_params.get('streamline color', 'white'), 
                       density = plot_params.get('streamline density', 1.5), 
                       linewidth = plot_params.get('streamline linewidth', 0.7))
        
        ax.set_xlim(0, a)
        ax.set_ylim(b, 0) # invert y-limits so that plot is right side up
        
        # ax.set_title(f"Re={sim.Re}\nt = {t:.1f}")
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Draw boxes (obstacles/boundaries):
        draw_circles(ax, sim)
        draw_boxes(ax, sim)
        fig.tight_layout()
        if save_params != None:
            title: str = save_params['title'] + f'_t{t:.1f}.png'
            plt.savefig(title, dpi = save_params.get('dpi', 200), bbox_inches='tight', pad_inches=0.1)
        
        if plot:    
            plt.show()
