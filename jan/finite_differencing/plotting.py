# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 22:44:08 2025

@author: reich
"""

import numpy as np
import matplotlib.pyplot as plt

from .finite_differencing import calculate_finite_differencing



def find_ordinal(number: int) -> str:
    if number == 1:
        return '1st'
    
    if number == 2:
        return '2nd'
    
    if number == 3:
        return '3rd'
    
    return str(number) + 'th'
    

def plot_num_derivative(type_data: list[dict], **kwargs) -> None:
    labels: list[str] = []
    
    plt.figure()
    
    for type_ in type_data:
        type_id: str = type_.get('id')
        
        function: callable = type_.get('callable')
        interval: list[float] = type_.get('interval')
        order: int = type_.get('order', 1)
        
        style_kwargs = type_.get('style', dict())
        
        if type_id == 'function':
            label: str = type_.get('label', 'Function')
            
            X: np.array = np.linspace(interval[0], interval[1], 201)
            Y: np.array = function(X)
            plt.plot(X, Y, **style_kwargs)
            
        elif type_id == 'analytic':                
            label: str = type_.get('label', find_ordinal(int(order)) 
                                   + " order derivative (analytic)")
            
            X: np.array = np.linspace(interval[0], interval[1], 201)
            Y: np.array = function(X)
            plt.plot(X, Y, **style_kwargs)
        
        else:
            step: float = type_.get('step')
            label: str = type_.get('label', find_ordinal(int(order)) 
                                   + " order derivative (" + str(type_id) + ')')
            
            X, Y, Y_prime = calculate_finite_differencing(function, interval, step, str(type_id),
                                                          int(order))
            plt.plot(X, Y_prime, **style_kwargs)
            
        labels.append(label)
    
    plt.title(kwargs.get('title', ''))
    plt.xlabel(kwargs.get('xlabel', 'x'))
    plt.ylabel(kwargs.get('ylabel', 'f(x)'))
    
    plt.legend(labels)
    plt.grid(True)
    
    plt.show()