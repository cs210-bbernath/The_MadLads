# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


#def build_poly(x, degree):
 #   (M, N) = x.shape
  #  poly = np.ones((len(x), N))
   # for deg in range(1, degree+1):
    #    poly = np.c_[poly, np.power(x, deg)]
    #return poly
def build_poly(x, degree):
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly