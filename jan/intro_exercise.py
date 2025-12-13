# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 19:35:17 2025

@author: reich
"""

from convergence import convergence_test, self_convergence_test
from euler_method import Euler_forward
from finite_differencing.finite_differencing import calculate_mixed_differencing
from finite_differencing.plotting import plot_num_derivative
from relaxation_method import relaxation_method

import matplotlib.pyplot as plt
import numpy as np



def task_1(function_id: str, h_param: float = 1) -> None:
    def f(x: float):
        return x**2 + 5 * x
    
    def f_1prime(x: float):
        return 2 * x + 5
    
    def f_2prime(x: float):
        return 2 * x / x
        
    
    def g(x: float):
        return np.exp(-1 * x**2)
    
    def g_1prime(x: float):
        return -2 * x * np.exp(-1 * x**2)
    
    def g_2prime(x: float):
        return (4 * x**2 - 2) * np.exp(-1 * x**2)
    
    
    def h(x: float, a: float = 1):
        return np.cos(a * np.pi * x)
    
    def h_1prime(x: float, a: float = 1):
        return -a * np.pi * np.sin(a * np.pi * x)
    
    def h_2prime(x: float, a: float = 1):
        return -a**2 * np.pi**2 * np.cos(a * np.pi * x)


    type_data: list[dict] = []
    type_ids: list[str] = ['function', 'analytic', 'analytic', 'mixed', 'mixed']
    orders: list[int] = [1, 1, 2, 1, 2]
    
    styles: list[dict] = [{}, {}, {}, {'linestyle': '--'}, {'linestyle': '--'}]
    
    if function_id == 'f':
        params: dict = {'interval': [-6.5, 1.5], 'step': 0.05}
        callables: list[callable] = [f, f_1prime, f_2prime, f, f]
        
        F_h: np.array = calculate_mixed_differencing(f, [-6.5, 1.5], 0.05)[2]
        F_h_half: np.array = calculate_mixed_differencing(f, [-6.5, 1.5], 0.025)[2]
        F_h_fourth: np.array = calculate_mixed_differencing(f, [-6.5, 1.5], 0.0125)[2]
        F_a: np.array = f_1prime(np.linspace(-6.5, 1.5, F_h.size))
        
        convergence_test(F_h, F_h_half, F_a)
        self_convergence_test(F_h, F_h_half, F_h_fourth)
    elif function_id == 'g':
        params: dict = {'interval': [-4, 4], 'step': 0.05}
        callables: list[callable] = [g, g_1prime, g_2prime, g, g]
    elif function_id == 'h':
        params: dict = {'interval': [0, 2 * np.pi], 'step': np.pi / 30}
        callables: list[callable] = [lambda x: h(x, h_param), lambda x: h_1prime(x, h_param),
                                     lambda x: h_2prime(x, h_param), lambda x: h(x, h_param), 
                                     lambda x: h(x, h_param)]
    else:
        raise ValueError('unknown function id provided')
        
    for i in range(len(type_ids)):
        type_data.append(dict({'id': type_ids[i], 'callable': callables[i], 'style': styles[i], 
                               'order': orders[i]}, **params))
    
    plot_num_derivative(type_data, ylabel = function_id + '(x)')
    
    
# task_1('h')



def task_2(interval: list[float], step: float, a: float = 1, C: float = 1) -> None:
    def f_change(t: float, u: float) -> float:
        return -1 * (t - a) * u
    
    
    T, U = Euler_forward(f_change, C, interval, step)
    
    T_analytic: np.array = np.linspace(interval[1], interval[0], 200)
    U_analytic: np.array = C * np.exp(-T_analytic**2 / 2 + a * T_analytic)
    
    plt.plot(T, U)
    plt.plot(T_analytic, U_analytic)
    
    plt.xlabel('t')
    plt.ylabel('u(t)')
    
    plt.legend(["Forward Euler", "Analytic solution"])
    plt.grid(True)
    
    plt.show()
    
    
# task_2([0, 10], 0.05, 2, 2)



def task_3() -> None:
    function: callable = np.cos
    interval: list[float] = [0, 3 * np.pi]
    step: float = np.pi / 50
    order: int = 2
    timestep: float = 0.001
    end_time: float = 5
    boundary_values: list[float] = [-2, 2]
    
    X, U = relaxation_method(function, interval, step, order, timestep, end_time, boundary_values)
    
    plt.plot(X, U)
    
    plt.xlabel('x')
    plt.ylabel('u')
    
    plt.grid(True)

    plt.show()
    

task_3()