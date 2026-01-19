import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import SymLogNorm

#load data
data = np.load('sim_data_Re4000_t20_nx100_ny100.npz')
u_data = data['u']
v_data = data['v']
t_data = data['t']
p_data = data['p']
nx = int(data['nx'])
ny = int(data['ny'])
Re = data['Re']

#last time point
u = u_data[-1]
v = v_data[-1]
p = p_data[-1]
t = t_data[-1]

print(f"Plotte Ergebnis bei t = {t:.2f}s")

#create grid
x = np.linspace(0, 1.0, nx)
y = np.linspace(0, 1.0, ny)
X, Y = np.meshgrid(x, y)

#staggered grid correction
u_plot = (u[0:-2, 1:-1] + u[1:-1, 1:-1]) / 2
v_plot = (v[1:-1, 0:-2] + v[1:-1, 1:-1]) / 2
p_plot = p[1:-1, 1:-1]

#plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

#pressure
contour = ax1.contourf(X, Y, p_plot.T, levels=40)#, norm=SymLogNorm(linthresh=0.01, linscale=1.0, vmin=p_plot.min(), vmax=p_plot.max()))
cbar = plt.colorbar(contour, label='Pressure (p)')

#velocity
ax1.streamplot(X, Y, u_plot.T, v_plot.T, color='black', linewidth=0.6, density=2.0)
#velocity abs value
velocity_mag = np.sqrt(u_plot**2 + v_plot**2)


ax1.set_title(f"Lid Driven Cavity (Re={Re}, t={t:.2f}s)\nPressure & Velocity")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)


contour2 = ax2.contourf(X, Y, velocity_mag.T, levels=40, cmap='viridis')

fig.colorbar(contour2, ax=ax2, label='Velocity Magnitude')

ax2.streamplot(X, Y, u_plot.T, v_plot.T, linewidth=0.6, density=1.5)
ax2.set_title(f"Velocity Magnitude\n(Re={Re})")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

plt.show()