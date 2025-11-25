import numpy as np

# adding ghost points depending on the bc

def add_ghost_points(f_val, ng, bc='periodic', bc_values=(0,0)):
    """
    f_val: array of function values
    ng: number of ghost points
    bc: boundary condition
    """
    bc_left = bc_values[0]
    bc_right = bc_values[1]
    if ng == 0:
        return f_val
    n = len(f_val)
    if bc == 'periodic':
        left = f_val[-ng:]
        right = f_val[:ng]
    elif bc == 'dirichlet':
        left = (2 * bc_left - f_val[1:ng+1])[::-1]
        right = (2 * bc_right - f_val[-ng-1:-1])[::-1]
    elif bc == 'neumann':
        left = f_val[1:ng+1][::-1]
        right = f_val[-ng-1:-1][::-1]
    return np.concatenate([left, f_val, right])
    
# central difference in 2nd and 4th order convergence

def deriv_central_2nd_order(x, h, f, ng=1, bc='dirichlet'):
    f_val = f(x)
    n = len(f_val)
    f_ext = add_ghost_points(f_val, ng, bc)
    return (f_ext[ng+1:ng+1+n] - f_ext[ng-1:ng-1+n]) / (2 * h)
    
def deriv_central_4th_order(x, h, f, ng=2, bc='dirichlet'):
    f_val = f(x)
    n = len(f_val)
    f_ext = add_ghost_points(f_val, ng, bc)
    return (f_ext[ng-2:ng-2+n] + 8 * (f_ext[ng+1:ng+1+n] - f_ext[ng-1:ng-1+n]) - f_ext[ng+2:ng+2+n]) / (12 * h)
