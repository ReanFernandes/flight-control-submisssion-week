from casadi import *
import numpy as np
import time
from model import dynamics
import matplotlib.pyplot as plt


def main(time_step =0.01, horizon = 20):
     # define the starting state of the drone, and the desired state we wwant to reach
    x_start = [0.25, 0.15, 0.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_des = [0.3, 0.2, 0.7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    opts = {'ipopt': {'print_level': 0, 'max_iter': 3000, 
                     'acceptable_tol': 1e-8, 
                     'acceptable_obj_change_tol': 1e-6, 
                     'fast_step_computation': 'yes'},
                     "jit" :False, 
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
    h = time_step
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
    
    cost = 0
    state_dev = x - x_ref
     # Weight the last state more
    # cost += trace(state_dev.T @ Q @ state_dev)
    cost += state_dev.T @ Q @ state_dev
    control_dev = u - u_ref
    # control_dev = U
    cost += control_dev.T @ R @ control_dev
    # cost += control_dev[:, -1].T @ R @ control_dev[:, -1]
    
    

    # define collocation polynomial
    degree = 3
    d = degree
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

          # define continuous time dynamics
    f = Function('f', [x, u],[xdot, cost])

    # Time step
    h = T / N

    # define the system bounds
     # bounds
    initial_control_guess = [mod.hover_speed, mod.hover_speed, mod.hover_speed, mod.hover_speed]
    initial_state_guess = x_start
    x_min = [ -1, -1, 0, -inf, -inf, -inf, -inf, -3, -3, -3, -2, -2, -2]
    x_max = [ 1, 1, 2, inf, inf, inf, inf, 3, 3, 3, 2, 2, 2]

    u_min = [0, 0, 0, 0]
    u_max = [22, 22, 22, 22]

    u_hover = [mod.hover_speed] * 4 # to define u_ref in the cost function for the solver

    
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []
    J = 0

    # for plotting the trajectory
    x_plot = []
    u_plot = []

    # create the start state parameter 
    # that will take the value of the initial state
    x0_hat = SX.sym('x0_hat', nx, 1)

    # create variable for the beginning of the state
    X0 = SX.sym('X0', nx, 1)
    w += [X0]
    lbw += x_min
    ubw += x_max
    w0 += initial_state_guess

    # add constraint to make the start state equal to the initial state
    g += [X0 - x0_hat]
    lbg += [0] * nx
    ubg += [0] * nx

    Xk = x0_hat

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = SX.sym('U_' + str(k), nu, 1)
        w += [Uk]
        u_plot += [Uk]
        lbw += u_min
        ubw += u_max
        
        w0 += initial_control_guess

        # state at the collocation points
        Xc = []
        for j in range(d):
            Xkj = SX.sym('X_' + str(k)+'_'+str(j), nx, 1)
            Xc += [Xkj]
            w += [Xkj]
            lbw += x_min
            ubw +=  x_max
            w0 +=   initial_state_guess

        # Loop over collocation points
        Xk_end = D[0] * Xk
        for j in range(1,d+1):
            # Expression for the state derivative at the collocation point
            xp = C[0,j] * Xk
            for r in range(d):
                xp = xp + C[r+1,j] * Xc[r]

            # Append collocation equations
            fj, qj = f(Xc[j-1], Uk)
            g += [h * fj - xp]
            lbg += [0] * nx
            ubg += [0] * nx

            # Add contribution to the end state
            Xk_end = Xk_end + D[j] * Xc[j-1]

            # Add contribution to quadrature function
            J = J + B[j] * qj * h
        
        # New NLP variable for state at end of interval
        Xk = SX.sym('X_' + str(k+1), nx, 1)
        w += [Xk]
        x_plot += [Xk]
        lbw += x_min
        ubw += x_max
        w0 += initial_state_guess
        

        # Add equality constraint
        g += [Xk_end-Xk]
        lbg += [0] * nx
        ubg += [0] * nx

   
  
    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(x0_hat, x_ref, u_ref)}
    solver = nlpsol('solver', 'ipopt', prob, opts)

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=vertcat(x_start, x_des, u_hover))
    w_opt = sol['x'].full().flatten()
    # solution_time = sol['t_wall_mainloop']
    x_opt = np.array([w_opt[ (nx +nu +d * nx ) * i : (nx +nu +d * nx ) * i + nx] for i in range(N+1)])
    fig  = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_opt[:, 0], x_opt[:, 1], x_opt[:, 2], 'o')
    plt.show()
    # deviation =[np.linalg.norm( x_opt[i,:3] - x_des[:3]) for i in range(N+1)]
    
    # #save deviation to a file with the title 'dev' with the infromation as horizon N, and deviation  deviation
    # with open('dev.txt', 'a') as f:
    #     f.write('sol_time' +  str(solution_time * 1000) + 't_ '+ str(time_step) + 'deviation = ' + str(final_dev) + '\n')
    # # save the data to text files named after the the first three entries of x_start and x_des
    # np.savetxt('X/x_opt_' + str(x_start[0]) + '_' + str(x_start[1]) + '_' + str(x_start[2]) + '_' + str(x_des[0]) + '_' + str(x_des[1]) + '_' + str(x_des[2]) + '.txt', x_opt)
    # np.savetxt('U/u_opt_' + str(x_start[0]) + '_' + str(x_start[1]) + '_' + str(x_start[2]) + '_' + str(x_des[0]) + '_' + str(x_des[1]) + '_' + str(x_des[2]) + '.txt', u_opt)
    # # write the Horizon length N, time step time_step, frequency freq , solution time solution_time, number of iteration iteration_count, number of rk4 steps M to a file in folder statistics to the file stats.txt under column headings
    # with open('statistics/dms_stats.txt', 'a') as f:
    #     f.write(str(N) + ' ' + str(time_step) + ' ' + str(freq) + ' ' + str(solution_time * 1000) + ' ' + str(iteration_count) + ' ' + str(M) + '' + str(np.round(x_opt[-1,:3], 2)) + '\n')
           

    


if __name__ == '__main__':
    horizon = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # time_step = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    # time_step = [0.01]
    for N in horizon:
        print('Horizon = ', N)
        main(horizon=N)