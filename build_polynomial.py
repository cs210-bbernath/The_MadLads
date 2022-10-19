# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,d+1)
        
    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    # polynomial basis function:
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    N = len(x)
    poly = np.empty([N, degree+1])
    for j in range(N):
        for d in range(degree+1):
            poly[j][d] = x[j]**d
    return poly