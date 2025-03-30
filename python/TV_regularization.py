import numpy as np
from numpy import sqrt, sum, abs, max, maximum, logspace, exp, log, log10, zeros
from scipy.signal import convolve2d
from numpy.linalg import norm
# Gradient of the individual components of the objective
kernel_h = [[1,-1,0]]
kernel_v = [[1],[-1],[0]]

def gradh(x):
    """Discrete gradient/difference in horizontal direction"""
    return convolve2d(x,kernel_h, mode='same', boundary='wrap')
def gradv(x):
    """Discrete gradient/difference in vertical direction"""
    return convolve2d(x,kernel_v, mode='same', boundary='wrap')
def grad2d(x):
    """The full gradient operator: compute both x and y differences and return them all.  The x and y
    differences are stacked so that rval[0] is a 2D array of x differences, and rval[1] is the y differences."""
    return np.stack([gradh(x),gradv(x)])


def gradht(x):
    """Adjoint of gradh"""
    kernel_ht = [[0,-1,1]]
    return convolve2d(x,kernel_ht, mode='same', boundary='wrap')
def gradvt(x):
    """Adjoint of gradv"""
    kernel_vt = [[0],[-1],[1]]
    return convolve2d(x,kernel_vt, mode='same', boundary='wrap')
def divergence2d(x):
    "The method is the adjoint of grad2d."
    return gradht(x[0])+gradvt(x[1])


# Using the individual components to create the complete gradient of the objective

def h(z, eps=.01):
    """The hyperbolic approximation to L1"""
    return sum(sqrt(z*z+eps*eps).ravel())
def tv_denoise_objective(x,mu,b):
    return h(grad2d(x)) + 0.5*mu*norm(x-b)**2
def h_grad(z, eps=.01):
    """The gradient of h"""
    return z/sqrt(z*z+eps*eps)
def tv_denoise_grad(x,mu,b):
    """The gradient of the TV objective"""
    return divergence2d(h_grad(grad2d(x))) + mu*(x-b)

