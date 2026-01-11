# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 12:35:33 2025

@author: reich
"""

import numpy as np



def convergence_test(Y_h: [np.array, list[float]], Y_h_half: [np.array, list[float]], Y_a: [np.array, list[float]]) -> None:
    fractions: np.array = np.zeros(Y_h.size)
    
    for i in range(Y_h.size):
        fractions[i] = np.abs(Y_a[i] - Y_h[i]) / np.abs(Y_a[i] - Y_h_half[i])
        
    print(fractions)
    
    
def self_convergence_test(Y_h: [np.array, list[float]], Y_h_half: [np.array, list[float]], Y_h_fourth: [np.array, list[float]]) -> None:
    fractions: np.array = np.zeros(Y_h.size)
    
    for i in range(Y_h.size):
        fractions[i] = np.abs(Y_h[i] - Y_h_half[i]) / np.abs(Y_h_half[i] - Y_h_fourth[i])
        
    print(fractions)