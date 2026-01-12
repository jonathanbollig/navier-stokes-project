# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 11:29:18 2025

@author: reich
"""

import timeit
import numpy as np



def time_function(function: callable) -> None:
    start_time = timeit.default_timer()
    function()
    end_time = timeit.default_timer()
    
    print("The function execution took: " + str((end_time - start_time) * 1000) + ' ms')


# Time the difference in execution time between calculating the derivative using function calls
#   versus from pre-existing array (much faster because less function calls are needed):
def time_function_vs_array(function: callable, interval: list[float], step: float, 
                           order: int = 1) -> None:
    from finite_differencing import forward_difference
    from finite_differencing import calculate_forward_differencing as forward_differencing_array
    
    
    def forward_differencing_function(function: callable, interval: list[float], step: float, 
                         order: int = 1) -> np.array:
        N: int = int(np.round((interval[1] - interval[0]) / step) + 1)
        X: np.array = np.linspace(interval[0], interval[1], N)
        Y_prime: np.array = np.zeros(N)
            
        for i in range(X.size):
            Y_prime[i] = forward_difference(function, X[i], step, order)
                
        return X, Y_prime
    
    
    time_function(lambda: forward_differencing_function(function, interval, step, order))
    time_function(lambda: forward_differencing_array(function, interval, step, order))
    

time_function_vs_array(np.cos, [0, 2 * np.pi], np.pi / 2000, 2)