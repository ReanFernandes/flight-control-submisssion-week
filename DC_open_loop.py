from casadi import *
import numpy as np
import time
from model import dynamics


def main(horizon):
     # define the starting state of the drone, and the desired state we wwant to reach
    x_start = [0.25, 0.15, 0.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_des = [0.3, 0.2, 0.7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    opts = {'ipopt': {'print_level': 5, 'max_iter': 1000, 
                     'acceptable_tol': 1e-8, 
                     'acceptable_obj_change_tol': 1e-6, 
                     'fast_step_computation': 'yes'},
                     "jit" :True, 
                     "jit_options" : {"flags" : "-O1"}}
    
    N = horizon # number of control intervals
    Q = diag([120,
                100,
                100,
                1*10,
                1*10,
                1*10,
                1*10,
                7e-1,
                1.0,
                1.0,
                1e-2,
                1e-2,
                1e-2])
    R = diag([1, 1, 1, 1])*0.01
    terminal_weight = 2
    
    # timing
    freq = 50 # hz
    h = 1 / freq
    T = N * h

    # define model dynamics
    mod = dynamics()
    xdot = mod.xdot
    x = mod.x
    u = mod.u
    nx = mod.nx
    nu = mod.nu
    f = Function('f', [x, u], [xdot]) # this function implements the dynamics of the quadcopter
    
    # define optimisation variables
    x0_hat = SX.sym('x0_hat', nx, 1)
    x_ref = SX.sym('x_ref', nx, 1)
    u_ref = SX.sym('u_ref', nu, 1) 

    # define collocation polynomial
    degree = 2
    tau_root = np.append(0, collocation_points(degree, 'legendre'))

    # Coefficients of the collocation equation
    C = np.zeros((degree + 1, degree + 1))

    # Coefficients of the continuity equation
    D = np.zeros(degree + 1)

    # Coefficients of the quadrature function
    B = np.zeros(degree + 1)

    # Construct polynomial basis
    for j in range(degree + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(degree + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(degree + 1):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    




    