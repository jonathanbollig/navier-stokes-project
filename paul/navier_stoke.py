import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import numpy as np

class NavierStokesSolver:
    def __init__(self, nx, ny, len_x, len_y, Re, gx=0.0, gy=0.0):
        self.nx, self.ny = nx, ny
        self.dx = len_x / nx
        self.dy = len_y / ny
        self.Re = Re
        self.gx, self.gy = gx, gy
        self.gamma = 0.9 # weighting factor (eq. 38)
        
        # arrays with ghost cells
        self.u = np.zeros((nx + 2, ny + 2))
        self.v = np.zeros((nx + 2, ny + 2))
        self.p = np.zeros((nx + 2, ny + 2))
        self.F = np.zeros_like(self.u)
        self.G = np.zeros_like(self.v)
        
    def apply_boundary_conditions(self):
        # no-slip (eq. 15, 16)
        # lid driven cavity
        self.u[0, :] = 0
        self.u[-1, :] = 0
        self.u[:, 0] = -self.u[:, 1]
        self.u[:, -1] = 2 - self.u[:, -2]

        self.v[0, :] = -self.v[1, :]
        self.v[-1, :] = -self.v[-2, :]
        self.v[:, 0] = 0  
        self.v[:, -1] = 0



    def calculate_F_G(self, dt):
        u, v = self.u, self.v
        dx, dy = self.dx, self.dy
        Re = self.Re
        gamma = self.gamma

        F = u.copy()
        G = v.copy()

        u_center = u[1:-1, 1:-1]
        u_right  = u[2:, 1:-1]
        u_left   = u[:-2, 1:-1]
        u_top    = u[1:-1, 2:]
        u_bot    = u[1:-1, :-2]

        v_center    = v[1:-1, 1:-1]
        v_right     = v[2:, 1:-1]
        v_left      = v[-2:, 1:-1]
        v_top       = v[1:-1, 2:]
        v_bot       = v[1:-1, :-2]
        v_bot_right = v[2:, :-2]

        # second derivative
        d2u_dx2 = (u_right - 2*u_center + u_left) / dx**2
        d2u_dy2 = (u_top - 2*u_center + u_bot) / dy**2

        d2v_dx2 = (v_right - 2*v_center + v_left) / dx**2
        d2v_dy2 = (v_top - 2*v_center + v_bot) / dy**2
        # nonlinear derivative u eq. 34
        u_avg_right = (u_center + u_right) / 2
        u_avg_left  = (u_left + u_center)  / 2
        u_diff_right = (u_center - u_right) / 2
        u_diff_left  = (u_left - u_center)  / 2

        du2_dx = ((u_avg_right**2 - u_avg_left**2) + gamma * (np.abs(u_avg_right)*u_diff_right - np.abs(u_avg_left)*u_diff_left)) / dx
        # nonlinear derivative v
        v_avg_top = (v_center + v_top) / 2
        v_avg_bot = (v_center + v_bot) / 2
        v_avg_bot_right = (v_bot + v_bot_right) / 2
        v_diff_top = (v_center - v_top) / 2
        v_diff_bot = (v_bot - v_center) / 2

        du2_dy = ((v_avg_top**2 - v_avg_bot**2) + gamma * (np.abs(v_avg_top)*v_diff_top - np.abs(v_avg_bot)*v_diff_bot)) / dy

        # nonlinear derivative uv
        
        pass 

    def solve_pressure_poisson(self, dt):
        # 1. Berechne RHS (Gl. 41)
        # 2. SOR Iteration (Gl. 42)
        it = 0
        residual = 1e6
        omega = 1.7 # Relaxationsfaktor
        
        while residual > 1e-4 and it < 1000:
            # Update p
            # Berechne residual
            it += 1
            pass
            
    def update_velocities(self, dt):
        # Berechne u_neu und v_neu basierend auf F, G und p (Gl. 21, 22)
        # u[i,j] = F[i,j] - dt/dx * (p[i+1,j] - p[i,j])
        pass

    def step(self, dt):
        # Hauptschleife pro Zeitschritt (Abschnitt 5 im PDF)
        self.apply_boundary_conditions()
        self.calculate_F_G(dt)
        self.solve_pressure_poisson(dt)
        self.update_velocities(dt)

# Beispiel Nutzung
solver = NavierStokesSolver(nx=50, ny=50, len_x=1.0, len_y=1.0, Re=100)
# Zeitschleife hier...

def solve_navier_stoke(step: float, timestep: float, endtime: float, x_velocity: np.array, y_velocity: np.array, pressure: np.array,  gx: float = 0, gy: float = -9.81, Re: float = 100)
    
    u: np.array = x_velocity
    v: np.array = y_velocity
    p: np.array = pressure

    u_sols: list[np.array] = []
        for _ in range(int(end_time/timestep)):
            u_change = (1 / Re) * ((u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / step**2
                                + (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / step**2)

            v_change = (1 / Re) * ((v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / step**2
                                + (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / step**2)


            du2_dx = (u[1:-1, 2:]**2 - u[1:-1, :-2]**2) / (2 * step)
            duv_dy = (u[2: , 1:-1] * v[2: , 1:-1]  - u[:-2 , 1:-1] * v[:-2 , 1:-1]) / (2 * step)
            duv_dx = (u[1:-1, 2:] * v[1:-1, 2:] - u[1:-1, :-2] * v[1:-1, :-2]) / (2 * step)
            dv2_dy = (v[2: , 1:-1]**2 - v[:-2 , 1:-1])**2 / (2 * step)
            F = u[1:-1, 1:-1] + (u_change - du2_dx - duv_dy + gx) * timestep
            G = v[1:-1, 1:-1] + (v_change - dv2_dx - duv_dx + gy) * timestep

            F = u.copy()
            G = v.copy()

            F[1:-1, 1:-1] = u[1:-1, 1:-1] + timestep * (u_change + u_ad)
            G[1:-1, 1:-1] = v[1:-1, 1:-1] + timestep * (v_change + v_ad)

            # boundary conditions
            F[0, :] = 0  
            F[-1, :] = 1 
            F[:, 0] = 0  
            F[:, -1] = 0 

            G[0, :] = 0  
            G[-1, :] = 0 
            G[:, 0] = 0  
            G[:, -1] = 0

            u = F
            v = G
        return F , G


