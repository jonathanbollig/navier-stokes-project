from finite_differences import deriv_central_2nd_order, deriv_central_4th_order
import numpy as np
import matplotlib.pyplot as plt


# define Grid
a = np.linspace(0, 2*np.pi, 100, endpoint=False)
# spacing
h = a[1] - a[0]
# define functions
function = {'sin': np.sin,
     'exp': np.exp,
     'pol': lambda x: 4*x**3 - 2*x**2 + x - 2}
deriv_ana = {'sin': np.cos,
             'exp': np.exp,
             'pol': lambda x: 12*x**2 - 4*x + 1}

def dif_test(f='sin', bc='periodic',s=0, e=2*np.pi, Ns=100):
    """
    f: function
    bc: boundary condition
    s: start
    e: end
    Ns: number of stencils
    plots the derivative (numerical 2nd and 4th order and analytic)
    plots the convergence
    """
    func = function[f]
    a = np.linspace(s, e, Ns, endpoint=False if bc == 'periodic' else True)
    h = a[1] - a[0]
    dif2 = deriv_central_2nd_order(a, h, func, bc=bc)
    dif4 = deriv_central_4th_order(a, h, func, bc=bc)
    ana = deriv_ana[f](a)

    plt.figure()
    plt.title(f'derivative {f}; bc = {bc}')
    plt.plot(a, dif2, label='dif2')
    plt.plot(a, dif4, label='dif4')
    plt.plot(a, ana, label='ana')
    plt.legend()

    # convergence test

    N = np.arange(10,1000,10)
    hs = []
    error2 = []
    error4 = []
    # create array for convergence test
    for n in N:
        a = np.linspace(0, 2*np.pi, n, endpoint=False if bc == 'periodic' else True)
        h = a[1] - a[0]
        hs.append(h)
        d2 = deriv_central_2nd_order(a, h, func, bc='periodic')
        d4 = deriv_central_4th_order(a, h, func, bc='periodic')
        ana = np.cos(a)
        error2.append(np.linalg.norm(d2 - ana, 2) * np.sqrt(h))
        error4.append(np.linalg.norm(d4 - ana, 2) * np.sqrt(h))

    plt.figure()
    plt.title(f'convergence derivative {f}; bc = {bc}')
    plt.plot(hs, error2, label='con2')
    plt.plot(hs, error4, label='con4')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('h')
    plt.ylabel('error')
    plt.legend()

    

dif_test()
dif_test(f='exp', bc='neumann')
dif_test(f='pol', bc='dirichlet')

plt.show()