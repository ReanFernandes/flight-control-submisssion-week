from model import *
from casadi import *

def main():
    N = 30 # number of control intervals
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
    terminal_weight = 50
    
    # timing
    f = 50 # hz
    h = 1 / f
    T = N * h

    # define model dynamics
    mod = dynamics()
    xdot = mod.xdot
    x = mod.x
    u = mod.u
    f = Function('f', [x, u], [xdot]) # this function implements the dynamics of the quadcopter
    # define optimisation variables
    U = SX.sym('U', mod.nu, N)
    X = SX.sym('X', mod.nx, N+1)
    
    x0_hat = SX.sym('x0_hat', mod.nx, 1) # parameter that takes the initial state
    x_ref = SX.sym('x_ref', mod.nx, 1) # parameter that takes the desired state
    u_ref = SX.sym('u_ref', mod.nu, 1) # parameter that takes the desired input for the cost fn ( hover or max speed)

    # define cost function
    cost = 0
    state_dev = X - repmat(x_ref, 1, N+1)
    state_dev[:, -1] *= terminal_weight # Weight the last state more
    cost += trace(state_dev.T @ Q @ state_dev)
    control_dev = U - repmat(u_ref, 1, N)
    cost += trace(control_dev.T @ R @ control_dev)
    cost_fn = Function('cost_fn', [X, U], [cost]) # this cost Implements the Q-weighted L-2 norm of the state devviation, and the R-weighted L-2 norm of the control deviation. The last state is multiplied by a terminal cost weight.

    # define integrator
    M = 2 # RK4 steps per interval
    DT = h / M
    x_integrated = SX(mod.nx, N+1)
    for i in range(N):
        x_integrated[:, i] = X[:, i]
        for j in range(M):
            k1 = f(x_integrated[:, i], U[:, i])
            k2 = f(x_integrated[:, i] + DT / 2 * k1, U[:, i])
            k3 = f(x_integrated[:, i] + DT / 2 * k2, U[:, i])
            k4 = f(x_integrated[:, i] + DT * k3, U[:, i])
            x_integrated[:, i] +=  DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    #int
    rk4 = Function('rk4', [X, U], [x_integrated]) # this function implements the RK4 integrator. The function integrated from each point Xk with control Uk to the next point Xk+1 starting with X0 and U0.
                                                #  the vector G contain Xk - x_integrated[k+1] 
    
    #initialise constraints
    g = DM(0, 1)
    lbg = DM(0, 1)
    ubg = DM(0, 1)
    w = DM(0,1)
    lbw = DM(0,1)
    ubw = DM(0,1)
    w0 = DM(0,1)
    # start state constraint
    start = x0_hat - X[:, 0]
    integrated_state = rk4(X, U)
    state_continuity_const = 



    # bounds
    initial_control_guess = [mod.hover_speed, mod.hover_speed, mod.hover_speed, mod.hover_speed]
    initial_state_guess = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_min = [ -1, -1, 0, -inf, -inf, -inf, -inf, -3, -3, -3, -2, -2, -2]
    x_max = [ 1, 1, 2, inf, inf, inf, inf, 3, 3, 3, 2, 2, 2]

    u_min = [0, 0, 0, 0]
    u_max = [22, 22, 22, 22]

if __name__ == '__main__':
    main()