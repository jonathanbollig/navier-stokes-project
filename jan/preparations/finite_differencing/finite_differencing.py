# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:00:20 2025

@author: reich
"""

import numpy as np



def forward_difference(object_: [np.array, list[float], callable], pos: int | float, step: float, 
                       order: int = 1) -> float:
    if order == 1:
        if callable(object_) == True:
            return (object_(pos + step) - object_(pos)) / step
        
        return (object_[pos + 1] - object_[pos]) / step
    
    elif order == 2:
        if callable(object_) == True:
            return (object_(pos + 2 * step) - object_(pos + step) + object_(pos)) / step**2
        
        return (object_[pos + 2] - 2 * object_[pos + 1] + object_[pos]) / step**2
    
    else:
        raise ValueError("requested order is not implemented")


def backward_difference(object_: [np.array, list[float], callable], pos: int | float, step: float,
                        order: int = 1) -> float:
    if order == 1:
        if callable(object_) == True:
            return (object_(pos) - object_(pos - 1)) / step
        
        return (object_[pos] - object_[pos - 1]) / step
    
    elif order == 2:
        if callable(object_) == True:
            return (object_(pos) - 2 * object_(pos - step) + object_(pos - 2 * step)) / step**2
        
        return (object_[pos] - 2 * object_[pos - 1] + object_[pos - 2]) / step**2
    
    else:
        raise ValueError("requested order is not implemented")


def central_difference(object_: [np.array, list[float], callable], pos: int | float, step: float, 
                       order: int = 1) -> float:
    if order == 1:
        if callable(object_) == True:
           return (object_(pos + step) - object_(pos - step)) / (2 * step) 
        
        return (object_[pos + 1] - object_[pos - 1]) / (2 * step)
    
    elif order == 2:
        if callable(object_) == True:
            return (object_(pos + step) - 2 * object_(pos) + object_(pos - step)) / step**2

        return (object_[pos + 1] - 2 * object_[pos] + object_[pos - 1]) / step**2
    
    else:
        raise ValueError("requested order is not implemented")
        
        
def calculate_forward_differencing(function: callable, interval: list[float], step: float, 
                                   order: int = 1, boundary_values: list[float] = None) -> np.array:
    N: int = int(np.round((interval[1] - interval[0]) / step) + 1)
    X: np.array = np.linspace(interval[0], interval[1], N)
    Y: np.array = function(X)
    Y_prime: np.array = np.zeros(N)
    
    for i in range(order):
        if boundary_values == None:
            Y = np.insert(Y, Y.size, function(X[-1] + (i + 1) * step))
        else:
            Y = np.insert(Y, Y.size, boundary_values[i])
        
    for i in range(Y.size - order):
        Y_prime[i] = forward_difference(Y, i, step, order)
        
    # Y_prime = (Y[1:] - Y[:-1]) / step
            
    return X, Y, Y_prime
            
            
def calculate_backward_differencing(function: callable, interval: list[float], step: float, 
                                    order: int = 1, boundary_values: list[float] = None) -> np.array:
    N: int = int(np.round((interval[1] - interval[0]) / step) + 1)
    X: np.array = np.linspace(interval[0], interval[1], N)
    Y: np.array = function(X)
    Y_prime: np.array = np.zeros(N)
    
    for i in range(order):
        if boundary_values == None:
            Y = np.insert(Y, 0, function(X[0] - (i + 1) * step))
        else:
            Y = np.insert(Y, 0, boundary_values[i])
        
    for i in range(order, Y.size):
        Y_prime[i - order] = backward_difference(Y, i, step, order)
        
    # Y_prime = (Y[1:] - Y[:-1]) / step
            
    return X, Y, Y_prime
            
            
def calculate_central_differencing(function: callable, interval: list[float], step: float, 
                                   order: int = 1, boundary_values: list[float] = None) -> np.array:
    N: int = int(np.round((interval[1] - interval[0]) / step) + 1)
    X: np.array = np.linspace(interval[0], interval[1], N)
    Y: np.array = function(X)
    Y_prime: np.array = np.zeros(N)
    
    if boundary_values == None:
        Y = np.insert(Y, 0, function(X[0] - step))
        Y = np.insert(Y, Y.size, function(X[-1] + step))
    else:
        Y = np.insert(Y, 0, boundary_values[0])
        Y = np.insert(Y, Y.size, boundary_values[1])
    
    if order <= 2:
        for i in range(1, Y.size - 1):
            Y_prime[i - 1] = central_difference(Y, i, step, order)
            
    # Y_prime = (Y[2:] - Y[:-2]) / (2 * step)
            
    return X, Y, Y_prime


def calculate_mixed_differencing(function: callable, interval: list[float], step: float, 
                                 order: int = 1) -> np.array:
    N: int = int(np.round((interval[1] - interval[0]) / step) + 1)
    X: np.array = np.linspace(interval[0], interval[1], N)
    Y: np.array = function(X)
    Y_prime: np.array = np.zeros(N)
    
    Y_prime[0] = forward_difference(Y, 0, step, order)
    
    for i in range(1, Y.size - 1):
        Y_prime[i] = central_difference(Y, i, step, order)
        
    Y_prime[-1] = backward_difference(Y, -1, step, order)
        
    return X, Y, Y_prime
            

def calculate_finite_differencing(function: callable, interval: list[float], step: float, 
                                  diff_type: str, order: int = 1) -> np.array:
    if diff_type == 'forward':
        X, Y, Y_prime = calculate_forward_differencing(function, interval, step, order)      
    elif diff_type == 'backward':
        X, Y, Y_prime = calculate_backward_differencing(function, interval, step, order)
    elif diff_type == 'central':
        X, Y, Y_prime = calculate_central_differencing(function, interval, step, order)
    elif diff_type == 'mixed':
        X, Y, Y_prime = calculate_mixed_differencing(function, interval, step, order)
    else:
        raise ValueError("unknown difference quotient type provided")
        
    return X, Y, Y_prime