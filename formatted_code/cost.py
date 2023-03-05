from casadi import *
import numpy as np
from model import dynamics


class cost:
    def __init__(self, Q, R, use_prev_control = False,  use_terminal_cost = False ,use_only_position = False,terminal_cost_weight = 50):

        self.use_prev_control = use_prev_control
        self.use_terminal_cost = use_terminal_cost
        self.use_only_position = use_only_position
        self.Q = Q
        self.R = R
        self.Q_n = terminal_cost_weight * self.Q
        # initialise model
        self.model = dynamics()
        self.x = self.model.x
        self.u = self.model.u
        self.nx = self.model.nx
        self.nu = self.model.nu
        self.hover_speed = self.model.hover_speed
        self.x_ref = SX.sym('x_ref', self.model.nx, 1)
        self.x0_hat = SX.sym('x0_hat', self.model.nx, 1)
        self.u_prev = SX.sym('u_prev', self.model.nu, 1)
        self.xdot = self.model.xdot
        # create the cost terms
        self.lagrange = None
        self.mayer = None
       
        
        self._stage_cost()
        if self.use_terminal_cost:
            self._terminal_cost()
        else:
            self.mayer = 0


    def _stage_cost(self):
        if self.use_only_position:
            if self.use_prev_control:
                self.lagrange = mtimes(mtimes((self.x[0:2] - self.x_ref[0:2]).T, self.Q[0:2,0:2]), (self.x[0:2] - self.x_ref[0:2])) + mtimes(mtimes((self.u - self.u_prev).T, self.R), (self.u - self.u_prev))
            else:
                self.lagrange = mtimes(mtimes((self.x[0:2] - self.x_ref[0:2]).T, self.Q[0:2,0:2]), (self.x[0:2] - self.x_ref[0:2])) + mtimes(mtimes(self.u.T, self.R), self.u)
        else:
            if self.use_prev_control:
                self.lagrange = mtimes(mtimes((self.x - self.x_ref).T, self.Q), (self.x - self.x_ref)) + mtimes(mtimes((self.u - self.u_prev).T, self.R), (self.u - self.u_prev))
            else:
                self.lagrange = mtimes(mtimes((self.x - self.x_ref).T, self.Q), (self.x - self.x_ref)) + mtimes(mtimes(self.u.T, self.R), self.u)        
    def _terminal_cost(self):
        if self.use_only_position:
            self.mayer = mtimes(mtimes((self.x[0:2] - self.x_ref[0:2]).T, self.Q_n[0:2,0:2]), (self.x[0:2] - self.x_ref[0:2]))
        else:
            self.mayer = mtimes(mtimes((self.x - self.x_ref).T, self.Q_n), (self.x - self.x_ref))