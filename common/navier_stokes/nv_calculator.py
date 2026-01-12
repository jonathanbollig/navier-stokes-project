import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import os
import pickle

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
        self.v = np.zeros_like(self.u)
        self.p = np.zeros_like(self.u)
        self.F = np.zeros_like(self.u)
        self.G = np.zeros_like(self.u)

        # lists with history for plotting
        self.u_history = []
        self.v_history = []
        self.p_history = []
        
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
        gx, gy = self.gx, self.gy
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
        v_left      = v[:-2, 1:-1]
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
        v_diff_top = (v_center - v_top) / 2
        v_diff_bot = (v_bot - v_center) / 2

        dv2_dy = ((v_avg_top**2 - v_avg_bot**2) + gamma * (np.abs(v_avg_top)*v_diff_top - np.abs(v_avg_bot)*v_diff_bot)) / dy

        # nonlinear derivative uv eq 37, 35
        v_avg_right = (v_right + v_center) / 2
        u_avg_top = (u_top + u_center) / 2
        v_avg_brb = (v_bot + v_bot_right) /2
        u_avg_bot = (u_bot + u_center)/2
        u_diff_top = (u_center - u_top) / 2
        u_avg_lbr = (u_left + u[:-2,2:]) / 2
        v_avg_left = (v_left + v_center) / 2
        u_diff_bot = (u_bot - u_center) / 2
        v_diff_right = (v_center - v_right) / 2
        v_diff_left = (v_left - v_center) / 2

        duv_dy = (v_avg_right*u_avg_top - v_avg_brb*u_avg_bot + gamma * (np.abs(v_avg_right) * u_diff_top - np.abs(v_avg_brb)*u_diff_bot)) / dy
        duv_dx = (u_avg_top*v_avg_right - u_avg_lbr*v_avg_left + gamma* (np.abs(u_avg_top)*v_diff_right - np.abs(u_avg_lbr)*v_diff_left)) / dx 

        F[1:-1,1:-1] = u_center + dt * ((d2u_dx2 + d2u_dy2) / Re - du2_dx - duv_dy + gx) 
        G[1:-1,1:-1] = v_center + dt * ((d2v_dx2 + d2v_dy2) / Re - duv_dx - dv2_dy + gy)

        G[:, 0] = 0.0
        G[:, -2] = 0.0 
        F[0, :] = 0.0
        F[-2, :] = 0.0

        self.F, self.G = F, G

    def SOR(self, dt, max_it = 10000):
        dx, dy = self.dx, self.dy
        F, G = self.F, self.G
        p = self.p.copy()

        # RHS eq. 41
        RHS = np.zeros_like(p)
        RHS[1:-1,1:-1] = ((F[1:-1,1:-1] - F[:-2,1:-1]) / dx + (G[1:-1,1:-1] - G[1:-1,:-2]) / dy) / dt
        mean_rhs = np.mean(RHS[1:-1,1:-1])
        # 2. SOR iteration (eq. 42)
        it = 0
        epsilon = 1e-3
        residual = np.zeros_like(p)
        norm_p = np.linalg.norm(p)
        if norm_p < 1e-10:
            norm_p = 1
        tolerance = max(epsilon * norm_p, 1e-4)
        omega = 1.7 # relaxation factor

        residual_norm = tolerance * 2
        
        while residual_norm > tolerance:
            if it >= max_it:
                print("-"*40)
                print("WARNING: SOR: max iterations reached")
                print("-"*40)
                break
            for i in range(1, self.nx + 1):
                for j in range(1, self.ny + 1):
                    # update p
                    p[i,j] = (1-omega) * p[i,j] + omega/(2*(1/dx**2+1/dy**2)) * ((p[i+1,j]+p[i-1,j])/dx**2 + (p[i,j+1]+p[i,j-1])/dy**2 - RHS[i,j]) 
            
            # fill boundary cells
            p[0,1:-1] = p[1,1:-1]
            p[1:-1,0] = p[1:-1,1]
            p[-1,1:-1] = p[-2,1:-1]
            p[1:-1,-1] = p[1:-1,-2]
            #p[1,1] = 0

            # calculate residual
            residual[1:-1,1:-1] = (p[2:,1:-1] - 2*p[1:-1,1:-1] + p[:-2,1:-1])/dx**2 + (p[1:-1,2:] - 2*p[1:-1,1:-1] + p[1:-1,:-2])/dy**2 - RHS[1:-1,1:-1]
            residual_norm = np.max(np.abs(residual))
            it += 1
        self.p = p
        self.p_history.append(p.copy())
        return it, mean_rhs
            
    def update_velocities(self, dt):
        # Berechne u_neu und v_neu basierend auf F, G und p (Gl. 21, 22)
        self.u[1:-1,1:-1] = self.F[1:-1,1:-1] - dt/self.dx * (self.p[2:,1:-1] - self.p[1:-1,1:-1])
        self.v[1:-1,1:-1] = self.G[1:-1,1:-1] - dt/self.dy * (self.p[1:-1,2:] - self.p[1:-1,1:-1])
        self.u_history.append(self.u.copy())
        self.v_history.append(self.v.copy())

    # main loop (section 5)
    def iterate(self, t_end, dt, print_info=True):
        t = 0
        while t < t_end:
            self.apply_boundary_conditions()
            self.calculate_F_G(dt)
            mean_rhs, it = self.SOR(dt)
            self.update_velocities(dt)
            t += dt
            
            if not print_info:
                continue
            max_u = np.max(np.abs(self.u[1:-1,1:-1]))
            print()
            print("p min:", np.min(self.p))
            print(f"sim time: {t:.3f}")
            print(f"Max U: {max_u:.4f}")
            print("mean RHS", mean_rhs)
            print("#iterations:", it)

    # a function that saves the complete object to a file, so it can be reloaded for plotting later
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


# lid driven cavity
if __name__ == "__main__":

    dt = 1e-2
    t_end = 0.1
    Re = 100
    nx, ny = 40, 40
    filename = f"lid_driven_n{nx}_re{Re}_t{int(t_end*1000)}_dt{int(dt*1000)}.pkl"
    
    # check if file exists
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            sim = pickle.load(f)
    else:
        sim = NavierStokesSolver(nx=nx, ny=ny, len_x=1.0, len_y=1.0, Re=Re)
        sim.iterate(t_end=t_end, dt=dt)
        sim.save(filename)

    # grid for plot
    x = np.linspace(0, 1.0, sim.nx)
    y = np.linspace(0, 1.0, sim.ny)
    X, Y = np.meshgrid(x, y)


    # set u and v back into the middle
    u_plot = (sim.u[1:-1, 1:-1] + sim.u[2:, 1:-1]) / 2
    v_plot = (sim.v[1:-1, 1:-1] + sim.v[1:-1, 2:]) / 2
    # velocity abs value
    velocity_mag = np.sqrt(u_plot**2 + v_plot**2)
    
    plt.contourf(X, Y, velocity_mag.T)
    plt.colorbar(label='velocity magnitude')

    plt.streamplot(X, Y, u_plot.T, v_plot.T, linewidth=0.5, density=2)
    
    plt.title(f"Lid Driven Cavity (Re={Re}, t={t_end:.2f}s)")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    #fig = plt.figure()

    # def animate(frame):
    #     steps_per_frame = 30
    #     for _ in range(steps_per_frame):
    #         if sim.t < t_end:
    #             sim.step(dt, sim.t)
    #             sim.t += dt
    #             max_u = np.max(np.abs(sim.u[1:-1,1:-1]))
    #             print(f"Zeit: {sim.t:.3f}, Max U: {max_u:.4f}")
    #         else:
    #             animation.event_source.stop()    

    #     # set u and v back into the middle
    #     u_plot = (sim.u[1:-1, 1:-1] + sim.u[2:, 1:-1]) / 2
    #     v_plot = (sim.v[1:-1, 1:-1] + sim.v[1:-1, 2:]) / 2
    #     # velocity abs value
    #     velocity_mag = np.sqrt(u_plot**2 + v_plot**2)
        
        
    #     plt.contourf(X, Y, velocity_mag.T)
    #     plt.colorbar(label='velocity magnitude')

    #     plt.streamplot(X, Y, u_plot.T, v_plot.T, linewidth=0.5, density=2)
        
    #     plt.title(f"Lid Driven Cavity (Re={sim.Re}, t={sim.t:.2f}s)")
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.clf()


    # animation = ani.FuncAnimation(fig, animate, interval = 50, cache_frame_data=False)
    #animation.save("lid_driven_cavity.gif", writer="pillow", fps=15)
    
