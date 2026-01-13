# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 19:09:31 2026

@author: common (based on Jan's code)
"""

from enum import Enum
import numpy as np
import conversions as conv

class NonlinType(Enum):
    MIXED = 1
    SQUARE = 2

def calc_deriv_factor(U: np.array, V: np.array, delta_x: float, delta_y: float, delta_t: float) -> float:
    x_cond: float = np.max(np.abs(U) * delta_t / delta_x)
    y_cond: float = np.max(np.abs(V) * delta_t / delta_y)
    return np.max([x_cond, y_cond]) # removed check if cond != 0.0 since I don't think it should occur.    

def lin_x(field: np.array, delta: float, order: int = 1) -> np.array:
    if order == 1:
        return (field[1:-1, 2:] - field[1:-1, :-2]) / (2 * delta)
    
    if order == 2:
        return (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / delta**2
    
    raise ValueError("requested order not implemented")
        
def lin_y(field: np.array, delta: float, order: int = 1) -> np.array:
    return lin_x(field.T, delta, order).T
    
def nonlin_x(U: np.array, V: np.array, delta_x: float, delta_y: float, delta_t: float, 
             type_: NonlinType) -> np.array:
    gamma: float = calc_deriv_factor(U, V, delta_x, delta_y, delta_t)
    
    if type_ == NonlinType.MIXED:
        # Convert u's and v's to actual grid points since they refer to different coordinate systems:
        U_grid: np.array = conv.U_like_to_grid(U)
        V_grid: np.array = conv.V_like_to_grid(V)
        
        U_grid_i: np.array = U_grid[:-1, 1:-1] + U_grid[1:, 1:-1]
        U_grid_i_off: np.array = U_grid[:-1, :-2] + U_grid[1:, :-2]
                            
        first_term: np.array = (U_grid_i * (V_grid[:-1, 1:-1] + V_grid[:-1, 2:])
                                - U_grid_i_off * (V_grid[:-1, :-2] + V_grid[:-1, 1:-1]))
        second_term: np.array = (np.abs(U_grid_i) * (V_grid[:-1, 1:-1] - V_grid[:-1, 2:])
                                - np.abs(U_grid_i_off) * (V_grid[:-1, :-2] - V_grid[:-1, 1:-1]))
        
        deriv: np.array = (first_term + gamma * second_term) / (4 * delta_x)
        
        return deriv[1:, :]
    
    if type_ == NonlinType.SQUARE:
        U_i: np.array = U[1:-1, 1:-1] + U[1:-1, 2:]
        U_i_off: np.array = U[1:-1, :-2] + U[1:-1, 1:-1]
        
        first_term: np.array = np.square(U_i) - np.square(U_i_off)
        second_term: np.array = (np.abs(U_i) * (U[1:-1, 1:-1] - U[1:-1, 2:])
                                 - np.abs(U_i_off) * (U[1:-1, :-2] - U[1:-1, 1:-1]))
        
        return (first_term + gamma * second_term) / (4 * delta_x)
    
    raise ValueError("unknown nonlinear derivative type provided")
    
def nonlin_y(U: np.array, V: np.array, delta_x: float, delta_y: float, delta_t: float, 
             type_: NonlinType) -> np.array:
    return nonlin_x(V.T, U.T, delta_y, delta_x, delta_t, type_).T