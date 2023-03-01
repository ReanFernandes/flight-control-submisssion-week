from casadi import *
import numpy as np

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
        self.hover_speed = np.sqrt((self.mq * self.g0)/(4 * self.Ct)) # [krpm] 
        print("hover speed :" , self.hover_speed)
        self.max_speed = 23 # [krpm] max speed
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
        

        self.u = vertcat(self.w1, self.w2, self.w3, self.w4)
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