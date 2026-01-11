# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 12:25:14 2025

@author: reich
"""

import numpy as np


def Euler_forward(f_change: callable, f_init: float, interval: list[float], step: float) -> tuple[np.array, np.array]:
    N: int = int((interval[1] - interval[0]) / step)
    F: np.array = np.zeros(N + 1)
    X: np.array = np.linspace(interval[0], interval[1], N + 1)
    F[0] = f_init
    
    for i in range(1, N + 1):
        F[i] = F[i - 1] + step * f_change(X[i - 1], F[i - 1])
        
    return X, F