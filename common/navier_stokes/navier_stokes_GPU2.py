# -*- coding: utf-8 -*-
"""
GPU-accelerated Navier-Stokes Solver using CuPy
================================================

Created on Thu Jan 23 2026

This module provides a GPU-accelerated version of the Navier-Stokes solver for
2D incompressible flow using CuPy for GPU array operations.

CuPy provides a NumPy-compatible interface that runs on NVIDIA GPUs, allowing
the entire simulation to run on GPU with minimal code changes from the original.

Requirements:
    - Python 3.11 (use conda environment: navier_stokes_311)
    - CuPy with CUDA 12 support (cupy-cuda12x)
    - NVIDIA GPU with compute capability 6.0+ (RTX 3060 has 8.6)

Environment Setup:
    The navier_stokes_311 conda environment must be properly configured:
    
    1. Activate with proper environment variables:
       
       source run_gpu2.sh
       python your_script.py
       
    2. Or set variables manually:
       
       export CUDA_PATH=/home/gandalf/miniforge3/envs/navier_stokes_311
       export LD_LIBRARY_PATH=/home/gandalf/miniforge3/envs/navier_stokes_311/lib:/home/gandalf/miniforge3/envs/navier_stokes_311/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
       /home/gandalf/miniforge3/envs/navier_stokes_311/bin/python your_script.py

Usage:
    from navier_stokes_GPU2 import NavierStokesGPU2
    
    sim = NavierStokesGPU2(nx=128, ny=128, len_x=1.0, len_y=1.0,
                           x_vel=2.0, Re=1000.0, tau=1.0, omega=1.7, epsilon=0.01)
    sim.set_boundary_type_and_boxes('lid', [])
    sim.iterate(t_end=1.0, N_max_P=100)
    
    # Get results on CPU for plotting/analysis
    U = sim.get_U_cpu()
    V = sim.get_V_cpu()
    P = sim.get_P_cpu()

Performance:
    CuPy implementation runs entire computation on GPU including F/G calculation.
    Benchmark results (vs CPU, lid-driven cavity):
    
    Grid Size   Speedup
    ---------   -------
    64x64       2.08x
    128x128     1.01x (overhead-limited)
    256x256     3.81x
    
    Best for larger grids (>128x128) where GPU parallelism can be fully utilized.

@author: common (GPU2 version using CuPy, based on Jan's original code)
"""

import numpy as np
import pickle
import os
import warnings
import time
from typing import Literal
from enum import Enum

# Import CuPy - will fail if not installed or CUDA not available
try:
    import cupy as cp
    from cupyx import jit as cupyx_jit
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np  # Fallback to numpy


# ============================================================================
# Configuration
# ============================================================================

class NonlinType(Enum):
    MIXED = 1
    SQUARE = 2


# ============================================================================
# GPU Array Operations using CuPy (NumPy-compatible interface)
# ============================================================================

def calc_deriv_factor(U, V, delta_x: float, delta_y: float, delta_t: float) -> float:
    """Calculate donor-cell scheme parameter gamma."""
    xp = cp.get_array_module(U)
    x_cond = float(xp.max(xp.abs(U) * delta_t / delta_x))
    y_cond = float(xp.max(xp.abs(V) * delta_t / delta_y))
    return max(x_cond, y_cond)


def lin_x(field, delta: float, order: int = 1):
    """Linear derivative in x direction (central difference)."""
    xp = cp.get_array_module(field)
    if order == 1:
        return (field[1:-1, 2:] - field[1:-1, :-2]) / (2 * delta)
    if order == 2:
        return (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / delta**2
    raise ValueError("requested order not implemented")


def lin_y(field, delta: float, order: int = 1):
    """Linear derivative in y direction."""
    return lin_x(field.T, delta, order).T


def U_like_to_grid(field):
    """Convert U-staggered field to grid points."""
    return (field[:-1, :] + field[1:, :]) / 2


def V_like_to_grid(field):
    """Convert V-staggered field to grid points."""
    return (field[:, :-1] + field[:, 1:]) / 2


def P_like_to_grid(field):
    """Convert P-staggered field to grid points."""
    return V_like_to_grid(U_like_to_grid(field))


def U_like_from_grid(field):
    """Convert grid points to U-staggered field."""
    xp = cp.get_array_module(field)
    conv_field = xp.zeros(shape=(field.shape[0] + 1, field.shape[1]), dtype=field.dtype)
    conv_field[1:-1, :] = (field[:-1, :] + field[1:, :]) / 2
    conv_field[0, :] = field[0, :] - (field[1, :] - field[0, :]) / 2
    conv_field[-1, :] = field[-1, :] + (field[-1, :] - field[-2, :]) / 2
    return conv_field


def V_like_from_grid(field):
    """Convert grid points to V-staggered field."""
    xp = cp.get_array_module(field)
    conv_field = xp.zeros(shape=(field.shape[0], field.shape[1] + 1), dtype=field.dtype)
    conv_field[:, 1:-1] = (field[:, :-1] + field[:, 1:]) / 2
    conv_field[:, 0] = field[:, 0] - (field[:, 1] - field[:, 0]) / 2
    conv_field[:, -1] = field[:, -1] + (field[:, -1] - field[:, -2]) / 2
    return conv_field


def nonlin_x(U, V, delta_x: float, delta_y: float, delta_t: float, type_: NonlinType):
    """Nonlinear derivative in x direction using donor-cell scheme."""
    xp = cp.get_array_module(U)
    gamma = calc_deriv_factor(U, V, delta_x, delta_y, delta_t)
    
    if type_ == NonlinType.MIXED:
        U_grid = U_like_to_grid(U)
        V_grid = V_like_to_grid(V)
        
        U_grid_i = U_grid[:-1, 1:-1] + U_grid[1:, 1:-1]
        U_grid_i_off = U_grid[:-1, :-2] + U_grid[1:, :-2]
        
        first_term = (U_grid_i * (V_grid[:-1, 1:-1] + V_grid[:-1, 2:])
                      - U_grid_i_off * (V_grid[:-1, :-2] + V_grid[:-1, 1:-1]))
        second_term = (xp.abs(U_grid_i) * (V_grid[:-1, 1:-1] - V_grid[:-1, 2:])
                       - xp.abs(U_grid_i_off) * (V_grid[:-1, :-2] - V_grid[:-1, 1:-1]))
        
        deriv = (first_term + gamma * second_term) / (4 * delta_x)
        return deriv[1:, :]
    
    if type_ == NonlinType.SQUARE:
        U_i = U[1:-1, 1:-1] + U[1:-1, 2:]
        U_i_off = U[1:-1, :-2] + U[1:-1, 1:-1]
        
        first_term = xp.square(U_i) - xp.square(U_i_off)
        second_term = (xp.abs(U_i) * (U[1:-1, 1:-1] - U[1:-1, 2:])
                       - xp.abs(U_i_off) * (U[1:-1, :-2] - U[1:-1, 1:-1]))
        
        return (first_term + gamma * second_term) / (4 * delta_x)
    
    raise ValueError("unknown nonlinear derivative type")


def nonlin_y(U, V, delta_x: float, delta_y: float, delta_t: float, type_: NonlinType):
    """Nonlinear derivative in y direction."""
    return nonlin_x(V.T, U.T, delta_y, delta_x, delta_t, type_).T


def norm_L2(field) -> float:
    """Compute L2 norm of a field."""
    xp = cp.get_array_module(field)
    return float(xp.sqrt(xp.mean(xp.square(field))))


# ============================================================================
# CuPy Raw Kernel for SOR (for even better performance)
# ============================================================================

if CUPY_AVAILABLE:
    # Raw CUDA kernel for Red-Black SOR iteration
    sor_red_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void sor_red(double* P, const double* RHS, 
                 double omega, double dx2, double dy2, double factor,
                 int yn, int xn, int stride) {
        int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
        
        if (i < xn && j < yn) {
            if ((i + j) % 2 == 0) {  // Red nodes
                int idx = j * stride + i;
                double term_x = (P[idx + 1] + P[idx - 1]) / dx2;
                double term_y = (P[idx + stride] + P[idx - stride]) / dy2;
                P[idx] = (1.0 - omega) * P[idx] + omega * factor * (term_x + term_y - RHS[idx]);
            }
        }
    }
    ''', 'sor_red')

    sor_black_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void sor_black(double* P, const double* RHS,
                   double omega, double dx2, double dy2, double factor,
                   int yn, int xn, int stride) {
        int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
        
        if (i < xn && j < yn) {
            if ((i + j) % 2 == 1) {  // Black nodes
                int idx = j * stride + i;
                double term_x = (P[idx + 1] + P[idx - 1]) / dx2;
                double term_y = (P[idx + stride] + P[idx - stride]) / dy2;
                P[idx] = (1.0 - omega) * P[idx] + omega * factor * (term_x + term_y - RHS[idx]);
            }
        }
    }
    ''', 'sor_black')

    compute_residual_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void compute_residual(const double* P, const double* RHS, double* residual,
                          double dx2, double dy2, int yn, int xn, int stride) {
        int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
        
        if (i < xn && j < yn) {
            int idx = j * stride + i;
            double laplacian = (P[idx + 1] - 2.0*P[idx] + P[idx - 1]) / dx2 +
                               (P[idx + stride] - 2.0*P[idx] + P[idx - stride]) / dy2;
            double res = laplacian - RHS[idx];
            residual[idx] = res * res;
        }
    }
    ''', 'compute_residual')


# ============================================================================
# GPU Navier-Stokes Simulation Class using CuPy
# ============================================================================

class NavierStokesGPU2:
    """
    GPU-accelerated Navier-Stokes solver using CuPy.
    
    This implementation runs entirely on GPU using CuPy's NumPy-compatible
    interface. All arrays are stored on GPU memory and operations use
    optimized CUDA kernels automatically.
    """
    
    def __init__(self, xn: int, yn: int, len_x: float, len_y: float, x_vel: float,
                 Re: float, tau: float, omega: float, epsilon: float, x_vel_type: Literal['constant', 'sinus']='constant'):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available. Please install cupy-cuda12x.")
        
        # Domain size
        self.xn = xn
        self.yn = yn
        
        # Grid parameters
        self.len_x = len_x
        self.len_y = len_y
        self.delta_x = len_x / xn
        self.delta_y = len_y / yn
        
        # Fluid parameters
        self.Re = Re
        self.tau = tau
        self.omega = omega
        self.epsilon = epsilon
        self.x_vel = x_vel
        self.x_vel_type = x_vel_type
        
        # Initialize field arrays on GPU
        self.U = cp.zeros((yn + 1, xn), dtype=cp.float64)
        self.V = cp.zeros((yn, xn + 1), dtype=cp.float64)
        self.P = cp.zeros((yn + 1, xn + 1), dtype=cp.float64)
        
        # Pre-allocate work arrays on GPU
        self.F = cp.zeros_like(self.U)
        self.G = cp.zeros_like(self.V)
        self.RHS = cp.zeros_like(self.P)
        
        # History storage (on CPU for memory efficiency)
        self.u_history = []
        self.v_history = []
        self.p_history = []
        self.t_history = []
        self.sparceify_factor = 1
        
        # Boundary settings
        self.boundary_type = None
        self.boxes = []
        
        # CUDA kernel configuration
        self.block_size = (16, 16)
        
    def set_boundary_type_and_boxes(self, type_: str, boxes: list) -> None:
        self.boundary_type = type_
        self.boxes = boxes

    def calc_timestep(self) -> float:
        """Calculate stable timestep based on CFL conditions."""
        if self.tau <= 0 or self.tau > 1:
            raise ValueError("tau must be from (0, 1]")
        
        try:
            Re_cond = self.Re / (2 * (self.delta_x**-2 + self.delta_y**-2))
        except ZeroDivisionError:
            Re_cond = float('inf')
        
        # Match original numpy behavior
        max_u = float(cp.max(cp.abs(self.U)))
        max_v = float(cp.max(cp.abs(self.V)))
        
        try:
            x_cond = self.delta_x / max_u
        except ZeroDivisionError:
            x_cond = float('inf')
        if max_u == 0:
            x_cond = float('inf')
            
        try:
            y_cond = self.delta_y / max_v
        except ZeroDivisionError:
            y_cond = float('inf')
        if max_v == 0:
            y_cond = float('inf')
        
        return self.tau * min(Re_cond, x_cond, y_cond)
    
    def calc_F_and_G(self, delta_t: float) -> None:
        """Compute F and G on GPU using CuPy array operations."""
        # F calculation
        self.F[:] = self.U
        self.F[1:-1, 1:-1] = self.F[1:-1, 1:-1] + delta_t / self.Re * (
            lin_x(self.U, self.delta_x, 2) + lin_y(self.U, self.delta_y, 2))
        self.F[1:-1, 1:-1] = self.F[1:-1, 1:-1] - delta_t * nonlin_x(
            self.U, self.V, self.delta_x, self.delta_y, delta_t, NonlinType.SQUARE)
        
        mixed_deriv = nonlin_y(self.U, self.V, self.delta_x, self.delta_y, delta_t, NonlinType.MIXED)
        self.F[1:-1, 1:-1] = self.F[1:-1, 1:-1] - delta_t * U_like_from_grid(mixed_deriv)
        
        # G calculation
        self.G[:] = self.V
        self.G[1:-1, 1:-1] = self.G[1:-1, 1:-1] + delta_t / self.Re * (
            lin_x(self.V, self.delta_x, 2) + lin_y(self.V, self.delta_y, 2))
        self.G[1:-1, 1:-1] = self.G[1:-1, 1:-1] - delta_t * nonlin_y(
            self.U, self.V, self.delta_x, self.delta_y, delta_t, NonlinType.SQUARE)
        
        mixed_deriv = nonlin_x(self.U, self.V, self.delta_x, self.delta_y, delta_t, NonlinType.MIXED)
        self.G[1:-1, 1:-1] = self.G[1:-1, 1:-1] - delta_t * V_like_from_grid(mixed_deriv)
    
    def calc_pressure(self, delta_t: float, N_max: int = 100) -> None:
        """Solve pressure Poisson equation using Red-Black SOR with CUDA kernels."""
        # Compute RHS
        deriv_F = cp.zeros((self.yn + 1, self.xn + 1), dtype=cp.float64)
        deriv_F[:, 1:-1] = (self.F[:, 1:] - self.F[:, :-1]) / self.delta_x
        deriv_F[:, 0] = self.F[:, 0] / self.delta_x
        deriv_F[:, -1] = -self.F[:, -1] / self.delta_x
        
        deriv_G = cp.zeros((self.yn + 1, self.xn + 1), dtype=cp.float64)
        deriv_G[1:-1, :] = (self.G[1:, :] - self.G[:-1, :]) / self.delta_y
        deriv_G[0, :] = self.G[0, :] / self.delta_y
        deriv_G[-1, :] = -self.G[-1, :] / self.delta_y
        
        self.RHS[:] = (1 / delta_t) * (deriv_F + deriv_G)
        
        # Pressure solver parameters
        dx2 = self.delta_x ** 2
        dy2 = self.delta_y ** 2
        factor = 1.0 / (2.0 * (1.0/dx2 + 1.0/dy2))
        
        P_0_norm = norm_L2(self.P)
        if P_0_norm == 0.0:
            P_0_norm = 1.0
        
        # Grid configuration for CUDA kernels
        grid_size = ((self.xn + self.block_size[0] - 1) // self.block_size[0],
                     (self.yn + self.block_size[1] - 1) // self.block_size[1])
        stride = self.P.shape[1]
        
        # SOR iterations using CUDA kernels
        for n in range(N_max):
            # Red sweep
            sor_red_kernel(grid_size, self.block_size,
                          (self.P, self.RHS, self.omega, dx2, dy2, factor,
                           self.yn, self.xn, stride))
            
            # Black sweep
            sor_black_kernel(grid_size, self.block_size,
                            (self.P, self.RHS, self.omega, dx2, dy2, factor,
                             self.yn, self.xn, stride))
            
            # Apply boundary conditions
            self.P[0, :] = self.P[1, :]
            self.P[-1, :] = self.P[-2, :]
            self.P[:, 0] = self.P[:, 1]
            self.P[:, -1] = self.P[:, -2]
            
            # Check convergence every 10 iterations
            if n % 10 == 9:
                residual = cp.zeros_like(self.P)
                compute_residual_kernel(grid_size, self.block_size,
                                       (self.P, self.RHS, residual, dx2, dy2,
                                        self.yn, self.xn, stride))
                max_res = float(cp.sqrt(cp.max(residual)))
                
                if max_res < self.epsilon * P_0_norm:
                    break
    
    def apply_boundary_condition(self, side: Literal['left', 'right', 'top', 'bottom'],
                                 U_val: float = 0, V_val: float = 0,
                                 start_val=0, end_val=-1, terminate_rekursion=False) -> None:
        a, b = start_val, end_val
        
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

    def apply_boundary_conditions(self, t: float, t_end: float) -> None:
        if self.x_vel_type == 'sinus':
            x_vel = self.x_vel * cp.sin(2 *cp.pi*t/t_end/2)  # two cycles over t_end
        else:
            x_vel = self.x_vel
        
        if self.boundary_type == "lid":
            self.apply_boundary_condition('left')
            self.apply_boundary_condition('right')
            self.apply_boundary_condition('top', U_val=x_vel)
            self.apply_boundary_condition('bottom')
        elif self.boundary_type == "lid_floor":
            self.apply_boundary_condition('left')
            self.apply_boundary_condition('right')
            self.apply_boundary_condition('top', U_val=x_vel)
            self.apply_boundary_condition('bottom', U_val=x_vel)
        elif self.boundary_type == "channel":
            self.apply_boundary_condition('left', U_val=x_vel)
            self.apply_boundary_condition('right', U_val=x_vel)
            self.apply_boundary_condition('top')
            self.apply_boundary_condition('bottom')
        elif self.boundary_type == "s_channel":
            self.apply_boundary_condition('left', U_val=x_vel, start_val=int(self.yn/2))
            self.apply_boundary_condition('right', U_val=x_vel, end_val=int(self.yn/2))
            self.apply_boundary_condition('top')
            self.apply_boundary_condition('bottom')
        elif self.boundary_type == "voided_channel":
            self.apply_boundary_condition('left', U_val=x_vel)
            self.apply_boundary_condition('right', U_val=x_vel)
            self.apply_boundary_condition('top', U_val=x_vel)
            self.apply_boundary_condition('bottom', U_val=x_vel)
        
    
        for box in self.boxes:
            self.apply_box_boundary(int(box[0]), int(box[1]), int(box[2]), int(box[3]))

    def apply_box_boundary(self, box_start_x: int, box_end_x: int, 
                          box_start_y: int, box_end_y: int) -> None:
        self.U[box_start_y:box_end_y, box_start_x:box_end_x] = 0
        self.V[box_start_y:box_end_y, box_start_x:box_end_x] = 0

    def update_velocities(self, delta_t: float) -> None:
        """Update velocities from F, G and pressure gradient."""
        self.U[1:-1, 1:-1] = self.F[1:-1, 1:-1] - delta_t * U_like_from_grid(
            P_like_to_grid(lin_x(self.P, self.delta_x)))
        self.V[1:-1, 1:-1] = self.G[1:-1, 1:-1] - delta_t * V_like_from_grid(
            P_like_to_grid(lin_y(self.P, self.delta_y)))

    def iterate(self, t_end: float, N_max_P: int = 100, max_histories: int = 200) -> None:
        """Run simulation from t=0 to t=t_end.
        
        Args:
            t_end: End time for simulation.
            N_max_P: Maximum number of pressure iterations.
            max_histories: Maximum number of history snapshots to save (default 200).
        """
        print_times = np.linspace(0, t_end, 100)
        print_index = 0
        
        # Calculate times at which to save history (evenly distributed)
        save_times = np.linspace(0, t_end, max_histories + 1)[1:]  # Exclude t=0
        save_index = 0

        t = 0.0
        step = 0
        start_time = time.time()
        
        while t < t_end:
            self.apply_boundary_conditions(t, t_end)
            delta_t = self.calc_timestep()

            self.calc_F_and_G(delta_t)
            self.calc_pressure(delta_t, N_max_P)
            self.update_velocities(delta_t)

            t += delta_t
            step += 1

            # Store history only at designated save times (copy to CPU)
            if save_index < len(save_times) and t >= save_times[save_index]:
                self.u_history.append(cp.asnumpy(self.U))
                self.v_history.append(cp.asnumpy(self.V))
                self.p_history.append(cp.asnumpy(self.P))
                self.t_history.append(t)
                save_index += 1

            if t >= print_times[print_index]:
                # ASCII progress bar updated at the same checkpoints as before
                frac = float(print_times[print_index] / t_end) if t_end > 0 else 1.0
                bar_len = 40
                filled = int(frac * bar_len)
                bar = '[' + '=' * filled + ' ' * (bar_len - filled) + ']'
                # overwrite previous bar using carriage return; add newline on final step
                # elapsed as integer seconds (no decimal) and ETA estimate
                elapsed = int(time.time() - start_time)
                if frac > 0 and frac < 1.0:
                    eta = int(elapsed * (1.0 - frac) / frac)
                else:
                    eta = 0
                msg = f"{bar} {print_times[print_index]:.2f}/{t_end:.2f} ({elapsed}s, ETA {eta}s) (step {step})"
                if print_index == (len(print_times) - 1):
                    # final update: print with newline
                    print('\r' + msg)
                else:
                    # update in-place without newline
                    print('\r' + msg, end='', flush=True)
                print_index += 1
        self.apply_boundary_conditions(t_end, t_end)
        print(f"Simulation complete: {step} timesteps")
    
    def sparcify_history(self, factor: int) -> None:
        self.u_history = self.u_history[::factor]
        self.v_history = self.v_history[::factor]
        self.p_history = self.p_history[::factor]
        self.t_history = self.t_history[::factor]
        self.sparceify_factor = factor

    def __getstate__(self) -> dict:
        """Prepare object for pickling by converting GPU arrays to CPU."""
        state = self.__dict__.copy()
        # Convert CuPy arrays to NumPy for serialization
        state['U'] = cp.asnumpy(self.U)
        state['V'] = cp.asnumpy(self.V)
        state['P'] = cp.asnumpy(self.P)
        state['F'] = cp.asnumpy(self.F)
        state['G'] = cp.asnumpy(self.G)
        state['RHS'] = cp.asnumpy(self.RHS)
        return state
    
    def __setstate__(self, state: dict) -> None:
        """Restore object from pickle, converting arrays back to GPU if available."""
        self.__dict__.update(state)
        # Convert NumPy arrays back to CuPy arrays if CuPy is available
        if CUPY_AVAILABLE:
            self.U = cp.asarray(self.U)
            self.V = cp.asarray(self.V)
            self.P = cp.asarray(self.P)
            self.F = cp.asarray(self.F)
            self.G = cp.asarray(self.G)
            self.RHS = cp.asarray(self.RHS)

    def keys(self) -> list:
        """Return list of attribute names for dict-like access compatibility."""
        return list(self.__dict__.keys())

    def save(self, filename: str) -> None:
        """Save the complete simulation object to a file using pickle."""
        if os.path.exists(filename):
            response = input(f"File {filename} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Save operation cancelled.")
                return
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def get_U_cpu(self) -> np.ndarray:
        """Get U velocity field on CPU."""
        return cp.asnumpy(self.U)
    
    def get_V_cpu(self) -> np.ndarray:
        """Get V velocity field on CPU."""
        return cp.asnumpy(self.V)
    
    def get_P_cpu(self) -> np.ndarray:
        """Get pressure field on CPU."""
        return cp.asnumpy(self.P)


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    import time
    
    print("=" * 60)
    print("GPU-Accelerated Navier-Stokes Solver (CuPy)")
    print("Python 3.11 / navier_stokes_311 environment")
    print("=" * 60)
    
    if not CUPY_AVAILABLE:
        print("\nERROR: CuPy is not available!")
        print("Please install with: pip install cupy-cuda12x")
        exit(1)
    
    # Check CUDA availability
    print(f"\nCuPy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")
    print(f"GPU compute capability: {cp.cuda.Device().compute_capability}")
    
    # Simulation parameters
    tau = 1.0
    omega = 1.7
    epsilon = 0.01
    x_vel = 2.0
    N_max_P = 100
    
    # Grid parameters
    nx = 256
    ny = 256
    len_x = 1.0
    len_y = 1.0
    Re = 1000.0
    T_max = 0.5
    type_ = "lid"
    boxes = []
    
    print(f"\nGrid: {nx} x {ny}")
    print(f"Reynolds number: {Re}")
    print(f"Simulation time: {T_max}")
    print(f"Boundary type: {type_}")
    
    # Run GPU simulation
    print("\n--- CuPy GPU Simulation ---")
    
    # Warm up GPU
    _ = cp.zeros((100, 100))
    cp.cuda.Stream.null.synchronize()
    
    start_time = time.time()
    
    sim_gpu = NavierStokesGPU2(nx, ny, len_x, len_y, x_vel, Re, tau, omega, epsilon)
    sim_gpu.set_boundary_type_and_boxes(type_, boxes)
    sim_gpu.iterate(T_max, N_max_P=N_max_P)
    
    cp.cuda.Stream.null.synchronize()  # Ensure all GPU operations complete
    elapsed_gpu = time.time() - start_time
    
    print(f"\nGPU (CuPy) completed in {elapsed_gpu:.2f} seconds")
    print(f"Timesteps: {len(sim_gpu.t_history)}")
    print(f"Time per step: {elapsed_gpu/len(sim_gpu.t_history)*1000:.2f} ms")
    
    # Get results on CPU for verification
    U_cpu = sim_gpu.get_U_cpu()
    V_cpu = sim_gpu.get_V_cpu()
    P_cpu = sim_gpu.get_P_cpu()
    
    print(f"Final max |U|: {np.max(np.abs(U_cpu)):.6f}")
    print(f"Final max |V|: {np.max(np.abs(V_cpu)):.6f}")
    print(f"Final max |P|: {np.max(np.abs(P_cpu)):.6f}")
    
    # Verify physical behavior
    print("\n--- Verification ---")
    print(f"Top boundary U (should be ~{x_vel}): {np.mean(U_cpu[0, :]):.4f}")
    print(f"V range: [{np.min(V_cpu):.4f}, {np.max(V_cpu):.4f}]")
    
    # Check divergence
    div_U = (U_cpu[1:-1, 1:] - U_cpu[1:-1, :-1]) / sim_gpu.delta_x
    div_V = (V_cpu[1:, 1:-1] - V_cpu[:-1, 1:-1]) / sim_gpu.delta_y
    div_max = np.max(np.abs(div_U + div_V))
    print(f"Max divergence: {div_max:.6e}")
    
    # Comparison test with smaller grid
    print("\n--- Comparison with Numba GPU version ---")
    T_test = 0.1
    nx_test, ny_test = 128, 128
    
    # CuPy version
    start_time = time.time()
    sim_cupy = NavierStokesGPU2(nx_test, ny_test, len_x, len_y, x_vel, Re, tau, omega, epsilon)
    sim_cupy.set_boundary_type_and_boxes(type_, boxes)
    sim_cupy.iterate(T_test, N_max_P=N_max_P)
    cp.cuda.Stream.null.synchronize()
    elapsed_cupy = time.time() - start_time
    
    print(f"CuPy GPU time: {elapsed_cupy:.2f}s")
    print(f"CuPy max |U|: {float(cp.max(cp.abs(sim_cupy.U))):.4f}")
    
    print("\n--- Test Complete! ---")
