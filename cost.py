import numpy as np
from casadi import * 
from model import dynamics

class cost: 
    def __init__(self, Q, R):
        self.Q = Q 
        self.R = R

        self.model = dynamics()
        self.nx = self.model.nx
        self.nu = self.model.nu
        self.x = self.model.x
        self.u = self.model.u
        self.xdot = self.model.xdot

        self.x_ref = SX.sym('x_ref', self.nx, 1)
        self.u_ref = SX.sym('u_ref', self.nu, 1)
        self.lagrange = None
        self.mayer = None
        self._cost()

    def _cost(self):
        self.lagrange = 0.5 * mtimes([(self.x - self.x_ref).T, self.Q, (self.x - self.x_ref)]) + 0.5 * mtimes([(self.u - self.u_ref).T, self.R, (self.u - self.u_ref)])
        self.mayer = 50 * mtimes([(self.x - self.x_ref).T, self.Q, (self.x - self.x_ref)])