# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 14:53:47 2026

@author: Jan
"""

import numpy as np

def U_like_to_grid(field: np.array) -> np.array:
    return (field[:-1, :] + field[1:, :]) / 2

def V_like_to_grid(field: np.array) -> np.array:    
    return (field[:, :-1] + field[:, 1:]) / 2

def P_like_to_grid(field: np.array) -> np.array:
    return V_like_to_grid(U_like_to_grid(field))
        
def U_like_from_grid(field: np.array) -> np.array:
    conv_field: np.array = np.zeros(shape = (field.shape[0] + 1, field.shape[1]))
    
    conv_field[1:-1, :] = (field[:-1, :] + field[1:, :]) / 2 # interpolate inner rows
    conv_field[0, :] = field[0, :] - (field[1, :] - field[0, :]) / 2 # extrapolate first row
    conv_field[-1, :] = field[-1, :] + (field[-1, :] - field[-2, :]) / 2 # extrapolate last row
        
    return conv_field

def V_like_from_grid(field: np.array) -> np.array:        
    conv_field = np.zeros(shape = (field.shape[0], field.shape[1] + 1))
    
    conv_field[:, 1:-1] = (field[:, :-1] + field[:, 1:]) / 2 # interpolate inner columns
    conv_field[:, 0] = field[:, 0] - (field[:, 1] - field[:, 0]) / 2 # extrapolate first column
    conv_field[:, -1] = field[:, -1] + (field[:, -1] - field[:, -2]) / 2 # extrapolate last column
    
    return conv_field

def P_like_from_grid(field: np.array) -> np.array:
    return V_like_from_grid(U_like_from_grid(field))