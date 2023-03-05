from model import *
from casadi import *
import time
from pylab import spy
import matplotlib.pyplot as plt

def main( time_step, horizon = 20, ):
    # define the starting state of the drone, and the desired state we wwant to reach
    x_start = [0.25, 0.15, 0.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_des = [0.3, 0.2, 0.7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    opts = {'ipopt': {'print_level': 5, 'max_iter': 1000, 
                     'acceptable_tol': 1e-8, 
                     'acceptable_obj_change_tol': 1e-6, 
                     'fast_step_computation': 'yes'},
                     "jit" :False, 
                     "jit_options" : {"flags" : "-O1"}}
    
    N = horizon # number of control intervals
    Q = diag([120,
                100,
                100,
                1e-2,
                1e-2,
                1e-2,
                1e-2,
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
    U = SX.sym('U', mod.nu, N)
    X = SX.sym('X', mod.nx, N+1)
    
    x0_hat = SX.sym('x0_hat', mod.nx, 1) # parameter that takes the initial state
    x_ref = SX.sym('x_ref', mod.nx, 1) # parameter that takes the desired state
    u_ref = SX.sym('u_ref', mod.nu, 1) # parameter that takes the desired input for the cost fn ( hover or max speed)

    # define cost function
    cost = 0
    state_dev = X - repmat(x_ref, 1, N+1)
    state_dev[:, -1] *= terminal_weight # Weight the last state more
    # cost += trace(state_dev.T @ Q @ state_dev)
    cost += state_dev[:, -1].T @ Q @ state_dev[:, -1]
    control_dev = U - repmat(u_ref, 1, N)
    # control_dev = U
    control_dev[:, -1] *= 2 # the last control must be similar to the hover speed
    cost += trace(control_dev.T @ R @ control_dev)
    # cost += control_dev[:, -1].T @ R @ control_dev[:, -1]
    cost_fn = Function('cost_fn', [X, U], [cost]) # this cost Implements the Q-weighted L-2 norm of the state devviation, and the R-weighted L-2 norm of the control deviation. The last state is multiplied by a terminal cost weight.
    
    # define integrator
    M = 2 # RK4 steps per interval
    
    DT = time_step
    x_integrated = SX(mod.nx, N)
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
    
    # define solver variables
    g = []
    lbg = []
    ubg = []
    w = []
    lbw = []
    ubw = []
    w0 = []

    # bounds
    initial_control_guess = [mod.hover_speed, mod.hover_speed, mod.hover_speed, mod.hover_speed]
    initial_state_guess = x_start
    x_min = [ -1, -1, 0, -inf, -inf, -inf, -inf, -3, -3, -3, -2, -2, -2]
    x_max = [ 1, 1, 2, inf, inf, inf, inf, 3, 3, 3, 2, 2, 2]

    u_min = [0, 0, 0, 0]
    u_max = [22, 22, 22, 22]

    u_hover = [mod.hover_speed] * 4 # to define u_ref in the cost function for the solver
    

    # equality constraints 

    integrated_state = rk4(X, U)   # gives the next state from the current shooting nodde Xk with control Uk till k = N
    start_constraint = x0_hat - X[:, 0]
    continuity_constraint = X[:, 1:] - integrated_state
    g += horzsplit(horzcat(start_constraint, continuity_constraint))
    lbg += [0] * (nx * (N+1))
    ubg += [0] * (nx * (N+1))
    g = vertcat(*g)     # convert constraints to dense vector form for solver
    lbg = vertcat(*lbg)
    ubg = vertcat(*ubg)

    # Add the to variable vector
    w += horzsplit(X)
    w += horzsplit(U)
    lbw += [*[x_min] * (N+1)]
    lbw += [*[u_min] * (N)]
    ubw += [*[x_max] * (N+1)]
    ubw += [*[u_max] * (N)]

    w = vertcat(*w)
    lbw = vertcat(*lbw)
    ubw = vertcat(*ubw)

    # add initial guess for w i.e. w0
    w0 += [*[initial_state_guess] * (N+1)]
    w0 += [*[initial_control_guess] * (N)]

    w0 = vertcat(*w0)

    # define cost function
    J = cost_fn(X, U)
    # create the solver
    prob = {'f' : J, 'x' : w, 'g' : g, 'p' : vertcat(x0_hat, x_ref, u_ref)}
    solver = nlpsol('solver', 'ipopt', prob, opts)
    time_s = time.time()
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=vertcat(x_start, x_des, u_hover))
    time_e = time.time()
    print(f"IPOPT status: %s" % solver.stats()["return_status"])

    solution_time = time_e - time_s
    w_opt = sol['x'].full().flatten()
    x_opt = w_opt[ : nx * (N+1)].reshape((N+1, nx))
    u_opt = w_opt[nx * (N+1) : ].reshape((N, nu))
    # fig  = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x_opt[:, 0], x_opt[:, 1], x_opt[:, 2], 'o')
    # plt.show()
    # deviation =[np.linalg.norm( x_opt[i,:3] - x_des[:3]) for i in range(N+1)]
    final_dev = np.linalg.norm( x_opt[-1,:3] - x_des[:3])
    #save deviation to a file with the title 'dev' with the infromation as horizon N, and deviation  deviation
    with open('dev.txt', 'a') as f:
        f.write('sol_time' +  str(solution_time * 1000) + 't_ '+ str(time_step) + 'deviation = ' + str(final_dev) + '\n')
    # save the data to text files named after the the first three entries of x_start and x_des
    # np.savetxt('X/x_opt_' + str(x_start[0]) + '_' + str(x_start[1]) + '_' + str(x_start[2]) + '_' + str(x_des[0]) + '_' + str(x_des[1]) + '_' + str(x_des[2]) + '.txt', x_opt)
    # np.savetxt('U/u_opt_' + str(x_start[0]) + '_' + str(x_start[1]) + '_' + str(x_start[2]) + '_' + str(x_des[0]) + '_' + str(x_des[1]) + '_' + str(x_des[2]) + '.txt', u_opt)
    # # write the Horizon length N, time step time_step, frequency freq , solution time solution_time, number of iteration iteration_count, number of rk4 steps M to a file in folder statistics to the file stats.txt under column headings
    # with open('statistics/dms_stats.txt', 'a') as f:
    #     f.write(str(N) + ' ' + str(time_step) + ' ' + str(freq) + ' ' + str(solution_time * 1000) + ' ' + str(iteration_count) + ' ' + str(M) + '' + str(np.round(x_opt[-1,:3], 2)) + '\n')
           



if __name__ == '__main__':
    # horizon = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    time_step = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    for N in time_step:
        main(time_step=N)