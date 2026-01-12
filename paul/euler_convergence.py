from euler import euler
import numpy as np
import matplotlib.pyplot as plt

# define function
def f(t,u):
    return -(t - a) * u

# analytical solution
def u_analytical(T, u_0, a):
    return u_0 * np.exp(-0.5 * T**2 + a * T)

# solve for all step sizes and calculate divergence
def conv_euler(f, u_0, step_size_ar):
    dts = []
    errors = []
    for dt in step_size_ar:
        u_num, T = euler(f, u_0, dt)
        u_ana = u_analytical(T, u_0, a)
        err = np.max(abs(u_num - u_ana))
        dts.append(dt)
        errors.append(err)
    return np.array(dts), np.array(errors)

def self_conv_euler (f, u_0, dt=0.1):
    u_num , T = euler(f, u_0, dt)
    u_num_2 , T_2 = euler(f, u_0, dt/2)
    u_num_4 , T_4 = euler(f, u_0, dt/4)
    conv = (abs(u_num[-1] - u_num_2[-1])/abs(u_num_2[-1] - u_num_4[-1]))
    p = np.log2(conv)
    return p

if __name__ == "__main__":
    a = 2
    t_0 = 0
    u_0 = 2
    t = 3

    # define step size array
    step_size_ar = np.logspace(-5 , -1, 10)

    dts, errors = conv_euler(f, u_0, step_size_ar)

    p, logC = np.polyfit(np.log10(dts), np.log10(errors), 1)

    p_self = self_conv_euler(f, u_0)
    print(f"order of self_convergence p = {p:.3f}")

    plt.figure()
    plt.title("convergence test")
    plt.loglog(dts, errors, "o-", label = f"order of convergence p = {p:.3f}")
    plt.xlabel("step size")
    plt.ylabel("max. error")
    plt.legend()
    plt.show()
