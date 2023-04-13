# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2016 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Simple example that connects to one crazyflie (check the address at the top
and update it to your crazyflie address) and send a sequence of setpoints,
one every 5 seconds.
This example is intended to work with the Loco Positioning System in TWR TOA
mode. It aims at documenting how to set the Crazyflie in position control mode
and how to send setpoints.
"""
import time
import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper
from NMPC_solver import solver

##################################################
# define the solver params 
Q = np.diag([120,
            100,
            100,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            7e-1,
            1.0,
            4.0,
            1e-2,
            1e-2,
            1e-2])
R = np.diag([1, 1, 1, 1])*0.6

solver_bounds = {"upper_pose_limit":[1, 1, 1.5],
                "lower_pose_limit":[-1, -1, 0],
                "upper_vel_limit":[5, 5, 5],
                "lower_vel_limit":[-5, -5, -5],
                "upper_att_limit":[1,1,1,1],
                "lower_att_limit":[0,-1,-1,-1],
                "upper_ang_limit":[10, 10, 10],
                "lower_ang_limit":[-10, -10, -10],
                "u_min" : [ 0, 0, 0, 0],
                "u_max" : [ 22, 22, 22, 22]}

nlp_opts = {"ipopt": {"max_iter": 3000, "print_level" :5}, "print_time":0, "jit":True}
cost_type = "no_slack"      # use slack variables for the terminal cost          
# time for dms closed loop
dms_timing = { "frequency" : 50,         # sampling frequency
            # "solution_time" : 0.1,      # real world time to navigate the drone
            "N" : 10 ,   
            "DMS_RK4_step_size" : 2, # step size of RK4 method for DMS
            "degree" : 2,
            "step_multiplier":2}           # degree of the collocation polynomial  
# time for dc closed loop
dc_timing = { "frequency" : 50,         # sampling frequency
            "solution_time" : 0.1, 
            "N" : 15   , # real world time to navigate the drone
            "DMS_RK4_step_size" : 2, # step size of RK4 method for DMS
            "degree" : 3,
            "step_multiplier":10}       

min_deviation = 0.05
##################################################
# define constants used in quaternion decompression
M_SQRT1_2 = np.sqrt(0.5)
MASK = (1 << 9) - 1
##################################################
# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E789')

# Change the sequence according to your setup
#             x    y    z  YAW
# sequence = [
#     (0.0, 0.0, 0.4, 0),
#     (0.0, 0.0, 1.2, 0),
#     (0.5, -0.5, 1.2, 0),
#     (0.5, 0.5, 1.2, 0),
#     (-0.5, 0.5, 1.2, 0),
#     (-0.5, -0.5, 1.2, 0),
#     (0.0, 0.0, 1.2, 0),
#     (0.0, 0.0, 0.4, 0),
# ]
# def extract_control_input():
#     control_input  = np.loadtxt('formatted_code/control_array.txt')
#     control_input = control_input.reshape(100,4)
#     return control_input.tolist()

# sequence = extract_control_input()
# def extract_position():
#     position = np.loadtxt('position.txt')
#     position = position.reshape(130,3)
#     return position.tolist()

# sequence = extract_position()
state = np.zeros((13,))
control_sequence = []
f = open("position.txt", "a")

def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            print("data",  data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break


def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    wait_for_position_estimator(cf)

def quatdecompress(comp):
    i_largest = comp >> 30
    sum_squares = 0.0
    q = np.zeros(4)
    for i in range(3, -1, -1):
        if i != i_largest:
            mag = comp & MASK
            negbit = (comp >> 9) & 0x1
            comp = comp >> 10
            q[i] = M_SQRT1_2 * mag / MASK
            if negbit == 1:
                q[i] = -q[i]
            sum_squares += q[i] * q[i]

    q[i_largest] = np.sqrt(1.0 - sum_squares)
    return q

# def quatdecompress(comp):
#   q = np.zeros(4)
#   mask = (1 << 9) - 1
#   i_largest = comp >> 30
#   sum_squares = 0
#   for i in range(3, -1, -1):
#     if i != i_largest:
#       mag = comp & mask
#       negbit = (comp >> 9) & 0x1
#       comp = comp >> 10
#       q[i] = mag / mask / np.sqrt(2)
#       if negbit == 1:
#         q[i] = -q[i]
#       sum_squares += q[i] * q[i]
#   q[i_largest] = np.sqrt(1.0 - sum_squares)
#   return q


# def state_callback(timestamp, data, logconf):
#     roll = data['stateEstimate.x']
#     pitch = data['stateEstimate.y']
#     yaw_rate = data['stateEstimate.z']
#     # thrust = data['controller.actuatorThrust']
#     control_sequence.append([roll,pitch,yaw_rate])
#     # control_sequence.append([roll,pitch])
#     print('pos: ({}, {}, {})'.format(roll, pitch, yaw_rate))
#     # print('control: ({}, {})'.format(roll, pitch))

def state_callback(timestamp, data, logconf):
    quat = quatdecompress(np.uint32(data['stateEstimateZ.quat']))
    state[0] = data['stateEstimateZ.x'] / 1000
    state[1] = data['stateEstimateZ.y'] / 1000
    state[2] = data['stateEstimateZ.z'] / 1000
    state[3] = quat[3]
    state[4] = quat[0]
    state[5] = quat[1]
    state[6] = quat[2]
    state[7] = data['stateEstimateZ.vx'] / 1000
    state[8] = data['stateEstimateZ.vy'] / 1000
    state[9] = data['stateEstimateZ.vz'] / 1000
    state[10] = data['stateEstimateZ.rateRoll'] / 1000
    state[11] = data['stateEstimateZ.ratePitch'] / 1000
    state[12] = data['stateEstimateZ.rateYaw'] / 1000

    #write the current position to a file
    f = open("formatted_code/position.txt", "a")
    f.write(str(state[0]) + " " + str(state[1]) + " " + str(state[2]) + " \n")
    # print('current position: {}, {}, {}'.format(state[0], state[1], state[2]))
def control_callback(timestamp, data, logconf):
    roll = data['controller.roll']
    pitch = data['controller.pitch']
    yaw_rate = data['controller.yawRate']
    thrust = data['controller.actuatorThrust']
    control_sequence.append([roll,pitch,yaw_rate,thrust])
    # control_sequence.append([roll,pitch])
    print('Control value: ({}, {}, {}, {})'.format(roll, pitch, yaw_rate, thrust))
    # print('control: ({}, {})'.format(roll, pitch))

def start_control_logging(scf):
    # log the given r, p ,y ,t values 
    control_config = LogConfig(name='Control', period_in_ms=200)
    control_config.add_variable('controller.roll', 'float')
    control_config.add_variable('controller.pitch', 'float')
    control_config.add_variable('controller.yawRate', 'float')  
    control_config.add_variable('controller.actuatorThrust', 'float')

    scf.cf.log.add_config(control_config)
    control_config.data_received_cb.add_callback(control_callback)
    control_config.start()
def start_position_logging(scf):
    log_conf = LogConfig(name='State', period_in_ms=200)
    log_conf.add_variable('stateEstimateZ.x', 'int16_t')
    log_conf.add_variable('stateEstimateZ.y', 'int16_t')
    log_conf.add_variable('stateEstimateZ.z', 'int16_t')
    log_conf.add_variable('stateEstimateZ.quat', 'uint32_t')
    log_conf.add_variable('stateEstimateZ.vx', 'int16_t')
    log_conf.add_variable('stateEstimateZ.vy', 'int16_t')
    log_conf.add_variable('stateEstimateZ.vz', 'int16_t')
    log_conf.add_variable('stateEstimateZ.rateRoll', 'int16_t')
    log_conf.add_variable('stateEstimateZ.ratePitch', 'int16_t')
    log_conf.add_variable('stateEstimateZ.rateYaw', 'int16_t')


    # log_conf.add_variable('controller.actuatorThrust', 'float')
    scf.cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(state_callback)
    log_conf.start()
    


# def run_sequence(scf, sequence):
#     cf = scf.cf
#     cf.commander.send_setpoint(0,0,0,0) # unlock thrust control
#     for position in sequence:
#         # print('Setting position {}'.format(position))
#         for i in range(29):
#             cf.commander.send_setpoint(position[0],
#                                                 position[1],
#                                                 position[2],
#                                                 int(position[3]))
#             time.sleep(0.75)

#     cf.commander.send_stop_setpoint()
#     # Make sure that the last packet leaves before the link is closed
#     # since the message queue is not flushed before closing
#     time.sleep(0.1)

def run_mpc(scf, solver, state_reference):
    print("Starting MPC")
    cf = scf.cf
    step = 0
    # print("hovering to stabilize")
    # while step<20:
    #     cf.commander.send_hover_setpoint( 0.0,0.0,0.0, 0.2)
    #     time.sleep(0.2)
    #     step +=1
    # print("hovering done")
    cf.commander.send_setpoint(0,0,0,0) # unlock thrust control
    print("Unlocking thrust control")
    step = 0
    stable_state_counter = 0
    while stable_state_counter < 200 and step<500:
        if solver.deviation < min_deviation:
            stable_state_counter += 1
        # cf.commander.send_setpoint(0,0,0,0) # unlock thrust control
        print("solving ocp")
        solver.solve()
        solver.extract_next_state()
        print("Extracting new state and updating initial guess")
        control = solver.control_to_drone()
        print("Sending control to drone:",  control) 
        cf.commander.send_setpoint( control[0],     # roll
                                    control[1],     # pitch
                                    control[2],     # yaw rate  
                                    int(control[3]))# thrust
        time.sleep(0.2) # wait for drone to reach the next state
        solver.X_opt_current = state # update the current state
        step += 1
    print("MPC finished")
    cf.commander.send_stop_setpoint()
    print("Stopping thrust control and hovering at the last position")
    cf.commander.send_hover_setpoint(state[8], state[9], state[12], state[3])
    time.sleep(0.2)
    print("Hovering finished")
    # Make sure that the last packet leaves before the link is closed
    # since the message queue is not flushed before closing
    time.sleep(0.1)


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    # initialise solvers
    DMS = solver( timing = dms_timing,
                  solver_bounds = solver_bounds,
                  nlp_opts = nlp_opts,
                  Q = Q,
                  R = R,
                  cost_type = "slack",
                  simulation_type = None
    )
    
    state_reference = np.array([ 0.046, -0.7, 0.4])

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        print("Crazyflie connected")
        reset_estimator(scf)
        print("Estimator reset")
        start_position_logging(scf)
        start_control_logging(scf)
        print("Logging started")
        print("sleeping to get position reading")
        time.sleep(0.5) 
        print("position reading done")
        print("current state:", state)
        DMS.hardware_set_initial_values(state, state_reference) 
        print("Initial values set")
        DMS.create_dms_solver()
        print("DMS solver created")
        run_mpc(scf, DMS, state_reference)

    with open('position_sequence.txt', 'w') as f:
        for item in control_sequence:
            f.write("%s " % item[0])
            f.write("%s " % item[1])
            f.write("%s " % item[2])
            f.write("%s " % item[3])