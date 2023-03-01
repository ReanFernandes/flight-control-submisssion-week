from casadi import *
import numpy as np
from cost import cost

class solver:
    def __init__(self, Q, R, timing, solver_bounds):
        self.Q = Q 
        self.R = R
        self.cost = cost(self.Q, self.R)
        self.nx = self.cost.nx
        self.nu = self.cost.nu
        self.x = self.cost.x
        self.u = self.cost.u
        self.xdot = self.cost.xdot
        self.lagrange = self.cost.lagrange
        self.mayer = self.cost.mayer
        self.x_ref=  self.cost.x_ref
        self.u_ref = self.cost.u_ref
        self.x0_hat = SX.sym('x0_hat', self.nx, 1)
        self.L = Function('L',[self.x, self.u] , [self.xdot, self.lagrange])
        self.M = Function('M', [self.x], [self.mayer])
        # timings 
        self.N = timing['N']
        self.frequency = timing['frequency']
        self.h = 1 / self.frequency
        self.T = self.N * self.h

        # bounds
        self.x_min = solver_bounds['x_min']
        self.x_max = solver_bounds['x_max']

        self.u_min = solver_bounds['u_min']
        self.u_max = solver_bounds['u_max']


        # solver variables
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []
        self.J = 0

    def rk4(self):
        