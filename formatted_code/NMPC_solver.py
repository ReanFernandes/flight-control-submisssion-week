from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import time

######################################################
#system model
class dynamics:
    def __init__(self):
        # define drone parameters
        self.g0  = 9.8066     # [m.s^2] accerelation of gravity
        self.mq  = 35e-3      # [kg] total mass (with one marker)
        self.Ixx = 1.395e-5   # [kg.m^2] Inertia moment around x-axis
        self.Iyy = 1.395e-5   # [kg.m^2] Inertia moment around y-axis
        self.Izz = 2.173e-5   # [kg.m^2] Inertia moment around z-axis
        self.Cd  = 7.9379e-06 # [N/krpm^2] Drag coef
        self.Ct  = 3.25e-4    # [N/krpm^2] Thrust coef
        self.dq  = 65e-3      # [m] distance between motors' center
        self.l   = self.dq/2       # [m] distance between motors' center and the axis of rotation
        # self.hover_speed = np.sqrt(self.mq * self.g0 / (4 * self.Ct)) # [krpm] hover speed
        self.hover_speed = np.sqrt((self.mq * self.g0)/(4 * self.Ct)) # [krpm] hover speed
        self.max_speed = 22 # [krpm] max speed
        # define state variables
    #    define the casadi variables for the system dynamics
        self.nx = 13
        self.nu = 4
        # state variables
        self.xq = SX.sym('xq')
        self.yq = SX.sym('yq')
        self.zq = SX.sym('zq')
        self.q1 = SX.sym('q1')
        self.q2 = SX.sym('q2')
        self.q3 = SX.sym('q3')
        self.q4 = SX.sym('q4')
        self.vbx = SX.sym('vbx')
        self.vby = SX.sym('vby')
        self.vbz = SX.sym('vbz')
        self.wx = SX.sym('wx')
        self.wy = SX.sym('wy')
        self.wz = SX.sym('wz')

        self.x = vertcat( self.xq, self.yq, self.zq, self.q1, self.q2, self.q3, self.q4, self.vbx, self.vby, self.vbz, self.wx, self.wy, self.wz)
        # control variables
        self.w1 = SX.sym('w1')
        self.w2 = SX.sym('w2')
        self.w3 = SX.sym('w3')
        self.w4 = SX.sym('w4')
        self.hover_speed = 18.175

        self.u =vertcat(self.w1, self.w2, self.w3, self.w4)
        # define the system dynamics 
        self.xdot = None   
        self._model()

    def _model(self):
        dxq = self.vbx*(2*self.q1**2 + 2*self.q2**2 - 1) - self.vby*(2*self.q1*self.q4 - 2*self.q2*self.q3) + self.vbz*(2*self.q1*self.q3 + 2*self.q2*self.q4)
        dyq = self.vby*(2*self.q1**2 + 2*self.q3**2 - 1) + self.vbx*(2*self.q1*self.q4 + 2*self.q2*self.q3) - self.vbz*(2*self.q1*self.q2 - 2*self.q3*self.q4)
        dzq = self.vbz*(2*self.q1**2 + 2*self.q4**2 - 1) - self.vbx*(2*self.q1*self.q3 - 2*self.q2*self.q4) + self.vby*(2*self.q1*self.q2 + 2*self.q3*self.q4)
        dq1 = - (self.q2*self.wx)/2 - (self.q3*self.wy)/2 - (self.q4*self.wz)/2
        dq2 = (self.q1*self.wx)/2 - (self.q4*self.wy)/2 + (self.q3*self.wz)/2
        dq3 = (self.q4*self.wx)/2 + (self.q1*self.wy)/2 - (self.q2*self.wz)/2
        dq4 = (self.q2*self.wy)/2 - (self.q3*self.wx)/2 + (self.q1*self.wz)/2
        dvbx = self.vby*self.wz - self.vbz*self.wy + self.g0*(2*self.q1*self.q3 - 2*self.q2*self.q4)
        dvby = self.vbz*self.wx - self.vbx*self.wz - self.g0*(2*self.q1*self.q2 + 2*self.q3*self.q4)
        dvbz = self.vbx*self.wy - self.vby*self.wx - self.g0*(2*self.q1**2 + 2*self.q4**2 - 1) + (self.Ct*(self.w1**2 + self.w2**2 + self.w3**2 + self.w4**2))/self.mq
        dwx = -(self.Ct*self.l*(self.w1**2 + self.w2**2 - self.w3**2 - self.w4**2) - self.Iyy*self.wy*self.wz + self.Izz*self.wy*self.wz)/self.Ixx
        dwy = -(self.Ct*self.l*(self.w1**2 - self.w2**2 - self.w3**2 + self.w4**2) + self.Ixx*self.wx*self.wz - self.Izz*self.wx*self.wz)/self.Iyy
        dwz = -(self.Cd*(self.w1**2 - self.w2**2 + self.w3**2 - self.w4**2) - self.Ixx*self.wx*self.wy + self.Iyy*self.wx*self.wy)/self.Izz
        self.xdot = vertcat(dxq, dyq, dzq, dq1, dq2, dq3, dq4, dvbx, dvby, dvbz, dwx, dwy, dwz)


#####################################################
# cost function
# 
class cost(): 
    def __init__(self, Q, R,terminal_cost_weight = 50):

        self.Q = Q
        self.R = R
        self.Q_n = terminal_cost_weight * self.Q
        self.Q_slack = np.array([self.Q_n[i,i] for i in range(self.Q_n.shape[0])])
        # print(self.Q_slack)
        # initialise model
        self.model = dynamics()
        self.x = self.model.x
        self.u = self.model.u
        self.nx = self.model.nx
        self.nu = self.model.nu

        self.max_speed = self.model.max_speed
        self.hover_speed = self.model.hover_speed
        self.x_ref = SX.sym('x_ref', self.nx, 1)
        self.x0_hat = SX.sym('x0_hat', self.nx, 1)
        self.u_prev = SX.sym('u_prev', self.nu, 1)
        self.xdot = self.model.xdot
        
        self.slack_low = SX.sym('slack_low', self.nx, 1)
        self.slack_high = SX.sym('slack_high', self.nx, 1)

        self.lagrange = None
        self.mayer = None

    def _stage_cost(self):
                self.lagrange = mtimes(mtimes((self.x - self.x_ref).T, self.Q), (self.x - self.x_ref)) + mtimes(mtimes((self.u - self.u_prev).T, self.R), (self.u - self.u_prev))
    def _terminal_cost(self):
        
            self.mayer = mtimes(mtimes((self.x - self.x_ref).T, self.Q_n), (self.x - self.x_ref))

    def _terminal_cost_slack(self):
        self.mayer = self.slack_low.T @ self.Q_slack + self.slack_high.T @ self.Q_slack

    def use_slack(self):
        self._stage_cost()
        self._terminal_cost_slack()

    def cost_without_slack(self):
        self._stage_cost()
        self._terminal_cost()


#####################################################
# solver

class solver():
    def __init__(self,  timing, solver_bounds, nlp_opts, Q, R, time_step = 0.01, cost_type ="slack", simulation_type= "Open_loop", use_shift = False) -> None:
        self.cost = cost(Q, R)
        self.simulation_type = simulation_type
        # model variables generated from the cost function ( no need to call model class again)
        self.xdot = self.cost.xdot
        self.x = self.cost.x
        self.u = self.cost.u
        self.nx = self.cost.nx
        self.nu = self.cost.nu
        self.slack_low = self.cost.slack_low
        self.slack_high = self.cost.slack_high
        self.u_prev  = self.cost.u_prev
        self.x_ref = self.cost.x_ref
        self.x0_hat = self.cost.x0_hat

        self.cost_type = cost_type
        self.max_speed = self.cost.max_speed
        self.max_thrust = self.max_thrust = 1.0942e-07 * (self.max_speed*1000)**2 - 2.1059e-04 * (self.max_speed*1000) + 1.5417e-01 # in grams
        self.max_trust_int_val = 0
        self.hover_speed = self.cost.hover_speed
        if cost_type == "slack":
            self.cost.use_slack()
            self.f_m = Function('f_m', [self.slack_low, self.slack_high], [self.cost.mayer])
        else:
            self.cost.cost_without_slack()
            self.f_m = Function('f_m', [self.x], [self.cost.mayer])
        self.lagrange = self.cost.lagrange
        self.f_l = Function('f_l', [self.x, self.u], [self.xdot, self.lagrange])
        
        self.use_shift = use_shift

        self.frequency = timing["frequency"]
        self.T = timing["N"] / self.frequency
        
        self.N = timing["N"]
        self.h = time_step

        self.method = None
        # solver variables
        self.nlp_opts = nlp_opts
        self.solver = None
        

        
        self.x_desired = None # desired position

        # solver bounds
        # define the bounds for the solver
        self.upper_pose_limit = solver_bounds["upper_pose_limit"]
        self.lower_pose_limit = solver_bounds["lower_pose_limit"]
        self.upper_vel_limit = solver_bounds["upper_vel_limit"]
        self.lower_vel_limit = solver_bounds["lower_vel_limit"]
        self.upper_att_limit = solver_bounds["upper_att_limit"]
        self.lower_att_limit = solver_bounds["lower_att_limit"]
        self.upper_ang_limit = solver_bounds["upper_ang_limit"]
        self.lower_ang_limit = solver_bounds["lower_ang_limit"]
        
        self.upper_state_bounds = [*self.upper_pose_limit, *self.upper_att_limit, *self.upper_vel_limit, *self.upper_ang_limit]
        self.lower_state_bounds = [*self.lower_pose_limit, *self.lower_att_limit, *self.lower_vel_limit, *self.lower_ang_limit]

        self.upper_control_bounds = [self.max_speed, self.max_speed, self.max_speed, self.max_speed]
        self.lower_control_bounds = [0, 0, 0, 0]

        self.initial_state_guess = None
        self.initial_control_guess = None

        # Optimisation variables
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []
        self.J = 0
        
        # solution vector
        self.w_opt = None
        self.X_opt = None 
        self.U_opt = None
        self.X_opt_current = None # at the first time step, it is equal to the given initial state
        self.U_opt_current = None
        self.slack_low_opt = None
        self.slack_high_opt = None

        # solver specific parameters
        # DMS parameters
        self.M = timing["DMS_RK4_step_size"]

        # DC parameters
        self.degree = timing["degree"]

        # Deviation 
        self.deviation = 100 # starting
        self.step_time = None
        self.solver_time = None
        self.total_solution_time = 0
        self.average_solution_time = 0
        self.deviation_list = []
        # control variable to send to drone
        self.control_list = []
        self.phi_list = []
        self.theta_list = []
        self.psi_list = []

        

    def set_initial_values(self, x_init, x_ref):
        # instantiate the parameters for the start and desired state, decided by user
        self.X_opt_current = np.concatenate((np.array(x_init), np.array([1]), np.zeros(self.nx -3 -1)) ) # change this later when taking actual drone state
        self.x_desired = np.concatenate((np.array(x_ref), np.array([1]), np.zeros(self.nx -3 -1)) ) 
        
        # instantiate the initial guess for state and control
        # self.initial_state_guess= self.X_opt_current
        # self.initial_control_guess = np.array([self.max_speed, self.max_speed, self.max_speed, self.max_speed])
        self.initial_state_guess = [0.001,0.001,0.001,1,0.1,0.1,0.1,0.01,0.01,0.01,0.010,.01,0.01]
        self.initial_control_guess = [self.hover_speed] * 4
        # the values to be passed to the solver 
        
        self.U_opt_current = self.initial_control_guess
        self.X_opt = self.X_opt_current
        self.U_opt = self.U_opt_current

    def hardware_set_initial_values(self, x_init, x_ref):
        self.X_opt_current = np.array(x_init)
        self.x_desired = np.concatenate((np.array(x_ref), np.array([1]), np.zeros(self.nx -3 -1)) ) 

        # self.initial_state_guess = [0.001,0.001,0.001,1,0.1,.1,.1,0.01,0.01,0.01,0.010,.01,0.01]

        self.initial_state_guess = self.X_opt_current.tolist()
        self.initial_control_guess = [self.hover_speed] * 4

        self.U_opt_current = self.initial_control_guess
        self.X_opt = self.X_opt_current
        self.U_opt = self.U_opt_current

    def create_dms_solver(self):
        # function call to create dms solver
        self.method = "DMS"
        # define integrator, ERK4
        DT = self.h / self.M
        X_int = SX.sym('X_int', self.nx, 1)
        U = SX.sym('U', self.nu, 1)
        X_l = X_int
        Q_l = 0
        for j in range(self.M):
            k1, k1_q = self.f_l(X_l, U)
            k2, k2_q = self.f_l(X_l + DT / 2 * k1, U)
            k3, k3_q = self.f_l(X_l + DT / 2 * k2, U)
            k4, k4_q = self.f_l(X_l + DT * k3, U)
            X_l = X_l + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Q_l = Q_l + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        F = Function('F', [X_int, U], [X_l, Q_l], ['x0', 'p'], ['xf', 'qf'])

        # create variable for the start of the shooting node
        X0 = SX.sym('X0', self.nx, 1)
        self.w += [X0]
        self.lbw += [*self.lower_state_bounds]
        self.ubw += [*self.upper_state_bounds]
        self.w0 += [*self.initial_state_guess]

        # add the equality constraints to make the starting state equal to the end state of the previous shooting node
        self.g += [X0 -self.x0_hat]
        self.lbg += [0] * self.nx
        self.ubg += [0] * self.nx

        Xk = X0
        
        #populate the shooting nodes
        for k in range(self.N):
            # nlp variable for control during the shooting interval
            Uk = SX.sym('U_' + str(k), self.nu, 1)
            self.w += [Uk]
            self.lbw += [*self.lower_control_bounds]
            self.ubw += [*self.upper_control_bounds]
            self.w0 += [*self.initial_control_guess]

            #integrate the the next shooting node
            Fk, Qk = F(Xk, Uk)
            Xk_end = Fk
            self.J += Qk

            # new NLP variable for the state at the end of the shooting interval
            Xk = SX.sym('X_' + str(k + 1), self.nx, 1)
            self.w += [Xk]
            self.lbw += [*self.lower_state_bounds]
            self.ubw += [*self.upper_state_bounds]
            self.w0 += [*self.initial_state_guess]

            # add equality constraint
            self.g += [Xk_end - Xk]
            self.lbg += [0] * self.nx
            self.ubg += [0] * self.nx

        if self.cost_type == "slack":
            # add the slack variables to the optimisation problem
            self.w += [self.slack_low]
            self.lbw += [0]* self.nx
            self.ubw += [inf] * self.nx
            self.w0 += [0] * self.nx

            self.w += [self.slack_high]
            self.lbw += [0] * self.nx
            self.ubw += [inf] * self.nx
            self.w0 += [0]* self.nx

            # add the slack constraint for the deviation from last state
            self.g += [Xk -self.x_ref + self.slack_low]
            self.lbg += [0] * self.nx
            self.ubg += [inf] * self.nx

            self.g += [self.x_ref - Xk + self.slack_high]
            self.lbg += [0] * self.nx
            self.ubg += [inf] * self.nx

            # add the end state cost
            cost = self.f_m(self.slack_low, self.slack_high)
            self.J += cost
        else:
            # add the end state cost
            cost = self.f_m(Xk)
            self.J += cost

        self.w_opt = np.zeros(len(self.w0))
        
        # create the solver
        prob = {'f': self.J, 'x': vertcat(*self.w), 'g': vertcat(*self.g), 'p': vertcat(self.x0_hat, self.u_prev, self.x_ref)}
        self.solver = nlpsol('solver', 'ipopt', prob, self.nlp_opts)

    def _collocation_polynomials(self):
        degree = self.degree
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

        return B, C, D 
    def create_dc_solver(self):
        self.method = "DC"
        B, C, D = self._collocation_polynomials()

        # create the state avriables for the beginning of the state
        X0 = SX.sym('X0', self.nx, 1)
        self.w += [X0]
        self.lbw += [*self.lower_state_bounds]
        self.ubw += [*self.upper_state_bounds]
        self.w0 += [*self.initial_state_guess]

        # add the equality constraints to make the starting state equal to the end state of the previous shooting node
        self.g += [X0 -self.x0_hat]
        self.lbg += [0] * self.nx
        self.ubg += [0] * self.nx

        Xk = X0
        for k in range(self.N):
            # new variable for control in the current interval
            Uk = SX.sym('U_' + str(k), self.nu, 1)
            self.w += [Uk]
            self.lbw += [*self.lower_control_bounds]
            self.ubw += [*self.upper_control_bounds]
            self.w0 += [*self.initial_control_guess]

            # state variable at the collocation points
            Xc = []
            for j in range(self.degree) :
                Xkj = SX.sym('X_' + str(k) + '_' + str(j), self.nx, 1)
                Xc += [Xkj]
                self.w += [Xkj]
                self.lbw += [*self.lower_state_bounds]
                self.ubw += [*self.upper_state_bounds]
                self.w0 += [*self.initial_state_guess]

            Xk_end = D[0] * Xk
            for j in range(1, self.degree + 1):
                #expression for the state derivative at the collocation point
                xp = C[0, j] * Xk
                for r in range(self.degree):
                    xp += C[r+1, j] * Xc[r]
                
                # append collocation equation
                fj, qj = self.f_l(Xc[j-1], Uk)
                self.g += [self.h * fj - xp]
                self.lbg += [0] * self.nx
                self.ubg += [0] * self.nx

                # add contribution to the end state
                Xk_end += D[j] * Xc[j-1]

                # add contribution to quadrature function
                self.J += self.h * B[j] * qj

            # new nlp variable for state at end of interval
            Xk = SX.sym('X_' + str(k+1), self.nx, 1)
            self.w += [Xk]
            self.lbw += [*self.lower_state_bounds]
            self.ubw += [*self.upper_state_bounds]
            self.w0 += [*self.initial_state_guess]

            # add equality constraint
            self.g += [Xk_end - Xk]
            self.lbg += [0] * self.nx
            self.ubg += [0] * self.nx

        if self.cost_type == "slack":
            # add the slack variables to the optimisation problem
            self.w += [self.slack_low]
            self.lbw += [-inf] * self.nx
            self.ubw += [0] * self.nx
            self.w0 += [0] * self.nx

            self.w += [self.slack_high]
            self.lbw += [0] * self.nx
            self.ubw += [inf] * self.nx
            self.w0 += [0] * self.nx
            
            # add the slack constraint for the deviation from last state
            self.g += [Xk -self.x_ref + self.slack_low]
            self.lbg += [0] * self.nx
            self.ubg += [inf] * self.nx

            self.g += [self.x_ref - Xk + self.slack_high]
            self.lbg += [0] * self.nx
            self.ubg += [inf] * self.nx

            # add the end state cost
            cost = self.f_m(self.slack_low, self.slack_high)
            self.J += cost
        else:
            # add the end state cost
            cost = self.f_m(Xk)
            self.J += cost
        self.w_opt = np.zeros(len(self.w0))
        # create the solver
        prob = {'f': self.J, 'x': vertcat(*self.w), 'g': vertcat(*self.g), 'p': vertcat(  self.x0_hat, self.u_prev, self.x_ref)}
        self.solver = nlpsol('solver', 'ipopt', prob, self.nlp_opts)

    def solve(self):
        # solve the ocp
        time_start = time.time()
        sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=vertcat(self.X_opt_current, self.initial_control_guess, self.x_desired))
        time_end = time.time()
        print(f"IPOPT status: %s" % self.solver.stats()["return_status"])

        self.solver_time = time_end - time_start
        self.w_opt = sol['x'].full().flatten()
        if self.simulation_type == "Open_loop":
            self._extract_solution()

    def extract_next_state(self):
        # based on the method, extract next state and the next control input for the solver
        if self.method == "DC":
            self.X_opt_current = self.w_opt[(self.nx + self.nu + self.nx * self.degree) : (self.nx + self.nu + self.nx * self.degree + self.nx)]
            self.U_opt_current = self.w_opt[(self.nx) : (self.nx + self.nu)]
        elif self.method == "DMS":
            self.X_opt_current = self.w_opt[(self.nx +self.nu) : (self.nx + self.nu + self.nx)]
            self.U_opt_current = self.w_opt[(self.nx) : (self.nx + self.nu)]
        self.deviation = np.linalg.norm((self.X_opt_current[:3] - self.x_desired[:3])) # calculate the deviation from the desired state
        self.deviation_list.append(self.deviation)
        self.X_opt = np.vstack([self.X_opt, self.X_opt_current])                        # save the state trajectory
        self.U_opt = np.vstack([self.U_opt, self.U_opt_current])   
        # print("Current pose ", self.X_opt_current[:3])                     # save the control input trajectory
        
        # set up the inital state for the next iterate
        if self.use_shift == True:
            self._shift_initialise()
        else:
            self._warm_start()
    
    def _extract_solution(self):
        if self.method == "DC":
            self.X_opt =np.vstack([self.w_opt[i * (self.nx +self.nu +self.degree * self.nx): i * (self.nx +self.nu +self.degree * self.nx)+self.nx] for i in range(self.N+1)])
            self.U_opt = np.vstack([self.w_opt[i * (self.nx +self.nu +self.degree * self.nx)+self.nx: i * (self.nx +self.nu +self.degree * self.nx)+self.nx+self.nu] for i in range(self.N)]) 
        elif self.method == "DMS":
            self.X_opt =np.vstack([self.w_opt[i * (self.nx +self.nu): i * (self.nx +self.nu)+self.nx] for i in range(self.N+1)])
            self.U_opt = np.vstack([self.w_opt[i * (self.nx +self.nu)+self.nx: i * (self.nx +self.nu)+self.nx+self.nu] for i in range(self.N)])
        self.deviation_list = [np.linalg.norm((self.X_opt[i,:3] - self.x_desired[:3])) for i in range(self.N+1)]
    def _shift_initialise(self):
        if self.method == "DC":
            self.w0 = np.hstack([self.w_opt[self.nx + self.nu + self.nx * self.degree:], self.w0[:self.nx + self.nu + self.nx * self.degree]])
        elif self.method == "DMS":
            self.w0 = np.hstack([self.w_opt[self.nx + self.nu:], self.w0[:self.nx + self.nu]])
    
    def _warm_start(self):
        self.w0 = self.w_opt

    def run_mpc(self, steps, min_deviation):
        # run the mpc for a certain number of steps
        stable_state_counter = 0
        for step in range(2, steps):
            step_start = time.time()
            self.solve()
            self.extract_next_state()
            self.generate_control_commands_for_drone()
            step_end = time.time()
            self.step_time = step_end - step_start
            self.total_solution_time += self.step_time
            print("time for solver in step ", step, " is ", self.solver_time)
            print("time for step ", step, " is ", self.step_time)
            if self.deviation < min_deviation:
                stable_state_counter += 1
                if stable_state_counter > 50:
                    print("MPC converged in ", step, " steps")
                    print("time taken",self.total_solution_time)
                    self.average_solution_time = self.total_solution_time / step
                    print("average solution time", self.average_solution_time)
                    
                    break
               
            if step == steps - 1:
                print("MPC did not converge in ", steps, " steps")

    def euler_2_quat(self):
        pass
    def quat_2_euler(self): # convert the X_opt_current quats to euler angles
        q = self.X_opt_current[3:7]
        w, x, y, z = q
        R11 = 2*(w*w + x*x) - 1
        R21 = 2*(x*y - w*z)
        R31 = 2*(x*z + w*y)
        R32 = 2*(y*z - w*x)
        R33 = 2*(w*w + z*z) - 1
        phi = np.arctan2(R32, R33) # roll
        theta = -np.arcsin(R31)    # pitch 
        psi = np.arctan2(R21, R11) # yaw
        self.phi_list = np.append(self.phi_list, self._rad_2_deg(phi))
        self.theta_list = np.append(self.theta_list, self._rad_2_deg(theta))
        self.psi_list = np.append(self.psi_list, self._rad_2_deg(psi))
        return phi, theta, psi
    
    def _rad_2_deg(self, angle):
        return angle * 180 / pi
    def _deg_2_rad(self, angle):
        return angle * pi / 180
    def _get_thrust(self):
        # use the mapping from rpm to thrust, for each motor, for the current applied control
            thrust = 1.0942e-07 * (self.U_opt_current*1000)**2 - 2.1059e-04 * (self.U_opt_current*1000) + 1.5417e-01 # in grams
            return thrust
    def _thrust_value_to_drone(self):
        # get the mapping from the thrust to the integer value to send drone controller ( min = 10001,  max = 60000)
            # mapping from thrust in grams to the value to send to drone 
            j = np.sum(self._get_thrust())
            k = j/(self.max_thrust * 4 )
            l = k * 50000
            m = l + 10001
            return int(m)

    def generate_control_commands_for_drone(self):
        self.control_list.append(self.control_to_drone())
        
    def control_to_drone(self):
        # the control to the drone is the next setpoint value we want for the following: [ roll, pitch, yaw_rate, thrust]
        phi, theta, _= self.quat_2_euler()
        roll = -1 * self._rad_2_deg(phi) # convert to degree,  since we use radians
        pitch = 1 * self._rad_2_deg(theta)
        yaw_rate = self._rad_2_deg(self.X_opt_current[12]) # psi_dot
        thrust = self._thrust_value_to_drone()
        return [roll, pitch, yaw_rate, thrust]



def main():
    Q = np.diag([120,
             100,
             100,
             1,
             1,
             1,
             1,
             7e-1,
             1.0,
             4.0,
             1e-2,
             1e-2,
             1e-2])
    R = np.diag([1, 1, 1, 1])*0.1

    solver_bounds = {"upper_pose_limit":[1, 1, 1.5],
                    "lower_pose_limit":[-1, -1, 0],
                    "upper_vel_limit":[2.5, 2.5, 2.5],
                    "lower_vel_limit":[-2.5, -2.5, -2.5],
                    "upper_att_limit": [inf]*4,#[1,1,1,1],
                    "lower_att_limit":[-inf]*4,#[0,-1,-1,-1],
                    "upper_ang_limit":[10, 10, 10],
                    "lower_ang_limit":[-10, -10, -10],
                    "u_min" : [ 0, 0, 0, 0],
                    "u_max" : [ 22, 22, 22, 22]}

    nlp_opts = {"ipopt": {"max_iter": 3000, "print_level" :5}, "jit": False, "print_time":0}
    # nlp_opts = {"max_iter":1}
    # nlp_opts = {"qpsol_options": {"max_iter":20}}
    cost_type = "slack"      # use slack variables for the terminal cost          
    # time for dms closed loop
    dms_timing = { "frequency" : 50,         # sampling frequency
                # "solution_time" : 0.05,      # real world time to navigate the drone
                "N" : 10    ,   
                "DMS_RK4_step_size" : 2, # step size of RK4 method for DMS
                "degree" : 2,
                "step_multiplier":2}           # degree of the collocation polynomial  
    # time for dc closed loop
    dc_timing = { "frequency" : 50,         # sampling frequency
                "solution_time" : 0.1, 
                "N" : 20   , # real world time to navigate the drone
                "DMS_RK4_step_size" : 2, # step size of RK4 method for DMS
                "degree" : 3,
                "step_multiplier":10}       

    min_deviation = 0.05
    # initialise class objects for the solvers
    dms_open_loop = solver(dms_timing, solver_bounds, nlp_opts, Q, R, cost_type )  # open loop by default
    dc_open_loop = solver(dc_timing, solver_bounds, nlp_opts, Q, R, cost_type = False)
    dms_closed_loop = solver(dms_timing, solver_bounds, nlp_opts, Q, R, cost_type, simulation_type = None) 
    dc_closed_loop = solver(dc_timing, solver_bounds, nlp_opts, Q, R, cost_type = False, simulation_type = None,use_shift=True)    
    
    x_start = [ 0, 0.0, 0.0]
    x_desired = [0.3, 0.4, 0.4]
    ################################################
    # Simulate open loop for DMS 

    # dms_open_loop.set_initial_values(x_start, x_desired)
    # dms_open_loop.create_dms_solver()
    # dms_open_loop.solve()
    # ("DMS solved")
    # print(dms_open_loop.X_opt)
    # print(dms_open_loop.U_opt)
    
    dc_open_loop.set_initial_values(x_start, x_desired)
    # dc_open_loop.initial_state_guess = np.zeros((13,))
    dc_open_loop.create_dc_solver()
    dc_open_loop.solve()
    dc_open_loop._extract_solution()
    x_opt = dc_open_loop.X_opt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_opt[:, 0], x_opt[:, 1], x_opt[:, 2], 'o')
    ax.plot(x_desired[0], x_desired[1], x_desired[2], 'ro', label = "desired")
    ax.plot(x_start[0], x_start[1], x_start[2], 'go', label = "start")
    ax.legend()
    plt.show()

    # dms_closed_loop.set_initial_values(x_start, x_desired)
    # dms_closed_loop.create_dms_solver()
    # dms_closed_loop.run_mpc(100, min_deviation)
    # # print(dms_closed_loop.X_opt[:, :3])
    # # print(dms_closed_loop.U_opt)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(dms_closed_loop.X_opt[:, 0], dms_closed_loop.X_opt[:, 1], dms_closed_loop.X_opt[:, 2])
    # plt.show()
    # time.sleep(2)
    # plt.close()

    # dc_closed_loop.set_initial_values(x_start, x_desired)
    # dc_closed_loop.create_dc_solver()
    # dc_closed_loop.run_mpc(100, min_deviation)
    # # print(dc_closed_loop.X_opt[:, :3])
    # print(dc_closed_loop.U_opt)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(dc_closed_loop.X_opt[:, 0], dc_closed_loop.X_opt[:, 1], dc_closed_loop.X_opt[:, 2])
    # plt.show()
    # time.sleep(2)
    # plt.close()
    # # print(dc_closed_loop.control_list)
    # # convert dc_closed_loop.control list to array 
    # control_array = dc_closed_loop.control_list
    # print(len(control_array))
    
    # # save control array to a text file
    # with open('formatted_code/control_array.txt', 'w') as f:
    #     for item in control_array:
    #         f.write("%s " % item[0])
    #         f.write("%s " % item[1])
    #         f.write("%s " % item[2])
    #         f.write("%s " % item[3])

    # with open('formatted_code/angle_list.txt', 'w') as f:
    #     for item in dc_closed_loop.phi_list, dc_closed_loop.theta_list, dc_closed_loop.psi_list:
    #         f.write("%s " % item)

    # with open('formatted_code/quaternion_list.txt', 'w') as f:
        # for item in dc_closed_loop.X_opt[:,3:7]:
            # f.write("%s " % item)
            


if __name__=="__main__":
    main() 

