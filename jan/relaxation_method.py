# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:49:43 2025

@author: reich
"""

from finite_differencing.finite_differencing import central_difference
import numpy as np



def relaxation_method(function: callable, interval: list[float], step: float, order: int, 
                      timestep: float, end_time: float, boundary_values: list[float]) -> None:
    def apply_boundary_values(Y: np.array, boundary_values: list[float]) -> np.array:
        Y[0] = boundary_values[0]
        Y[-1] = boundary_values[1]
        
        return Y
    
    
    N: int = int(np.round((interval[1] - interval[0]) / step) + 1)
    X: np.array = np.linspace(interval[0], interval[1], N)
    F: np.array = function(X)
    
    Y: np.array = np.zeros(N + 2)
    Y = apply_boundary_values(Y, boundary_values)
    Y_change: np.array = np.zeros(N)
    
    t: float = 0

    while t <= end_time:
        for i in range(1, Y.size - 1):
            Y_change[i - 1] = central_difference(Y, i, step, order) - F[i - 1]
        
        Y[1:-1] = Y[1:-1] + timestep * Y_change
        Y = apply_boundary_values(Y, boundary_values)
        
        t = t + timestep
        
    return X, Y[1:-1]