
# TODO. yaw_rate is weird
"""
2D Controller Class to be used for waypoint following demo.
"""

import numpy as np
from collections import deque
# from pytictoc import TicToc

from sympy.diffgeom import *
from sympy import sqrt,sin,cos
from sympy import *
import sympy
import time

# from sympy.diffgeom import *
# from sympy import *
from cmath import sqrt,pi, exp
# import sympy
from matplotlib import animation
import numpy as np
# from bicyclemodel import LinearBicycleModel

from traffic_model.utils.ccbf.obstacle import Obstacle
from traffic_model.utils.ccbf import VehicleSpec

Rw = 0.325       #rolling radius of drive wheel, m
m = 1412.0       #mass of vehicle, kg
Iz = 1536.7      #yaw inertia of vehicle, kg*m^2
a = 1.06         #distance between C.G. and front axle, m
b = 1.85         #distance between C.G. and rear axle, m
kf = -128915.5   #equivalent sideslip force stiffness of front axle, N/rad
kr = -85943.6    #equivalent sideslip force stiffness of rear axle, N/rad


# renew the parameters using config file
m = VehicleSpec.mass       #mass of vehicle, kg
Iz = VehicleSpec.I_zz      #yaw inertia of vehicle, kg*m^2
a = VehicleSpec.a         #distance between C.G. and front axle, m
b = VehicleSpec.b         #distance between C.G. and rear axle, m
kf = VehicleSpec.C_yf   #equivalent sideslip force stiffness of front axle, N/rad
kr = VehicleSpec.C_yr    #equivalent sideslip force stiffness of rear axle, N/rad

# Set maximum/minimum acceleration and steer angle
# min_acc = -1
# max_acc = 1
# min_steer = -1.22
# max_steer = 1.22

# Set maximum/minimum acceleration and steer angle
min_acc = -2
max_acc = 2
min_steer = -1.22
max_steer = 1.22



# Find the index of the closest point on the waypoints,
# and the distance betweent the current x,y location and the closest point
def find_closest_point(waypoints, current_x, current_y):
    # Find the closest distance and closest index
    closest_index=0
    closest_distance = np.linalg.norm(np.array([
        waypoints[0][closest_index] - current_x,
        waypoints[1][closest_index] - current_y]))
    new_distance = closest_distance
    new_index = closest_index
    while (new_distance <= closest_distance):
        closest_distance = new_distance
        closest_index = new_index
        new_index += 1
        if new_index >= len(waypoints[0]):  # End of path
            break
        new_distance = np.linalg.norm(np.array([
            waypoints[0][new_index] - current_x,
            waypoints[1][new_index] - current_y]))
        
    print("@@@@@@@@@closest distance : ", closest_distance)
    return closest_index, closest_distance


class CCBF_Controller(object):
    def __init__(self, vehicle_spec:VehicleSpec, dt=0.1, k=1.0):#k=1.0
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._desired_speed = 0
        self._current_vx = 0
        self._current_vy = 0
        self._current_frame = 0
        self._current_timestamp = 0
        self._ref_vx = 0
        self._ref_vy = 0
        self.throttle = 0
        self.brake = 0
        self.steer = 0
        self.steer_expect = 0
        self._waypoints = None
        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi
        self.e_buffer = deque(maxlen=20)
        self._e = 0
        self.dt = dt
        self._lyapunov = 0
        self._barrier = 0
        self.yaw_path = 0
        self._x_ref = 0
        self._y_ref = 0
        self.x_ref = 0
        self.y_ref = 0
        
        # for record
        self._ref_vx = 0
        self._ref_vy = 0
        self._lya_ref_vx = 0
        self._lya_ref_vy = 0

        # store the lyapunov and error values on the class for recording
        self._lyapunov = 0
        self.x_error = 0
        self.y_error = 0
        self.yaw_error = 0
        self.vx_error = 0
        self.vy_error = 0
        self.omega_error = 0

        # self.ka = 12
        # self.kb = 36
        self.min_brake_dist = 0
        self.d_safe = 10
        # self.d_safe_y = 5
        self.min_d_safe = 3 # 3m for default
        self.time_safe = 3.5
        # emergency braking distance = v^2/(2*a_max)
        
        self.d_safe_gp = 0

        # Obstacle location
        self.x_obs = 9999.0
        self.y_obs = 9999.0

        self.obstacle = None
        self.obstacle2 = None
        self.obstacle3 = None
        self.obstacle4 = None

        # parameters for pid speed controller
        self.K_P = 1
        self.K_D = 0.001
        self.K_I = 0.3

        # To check whether cbf is valid
        self.cbf_on = False
        self.b_positive_obs1 = None
        self.b_positive_obs2 = None
        self.b_positive_obs3 = None
        self.b_positive_obs4 = None

        # exponential parameter
        self.k = k
        self.slack_variable = 2 #slack_variable =2.0

        self.h = 0

        # self.steer_adjust_weight_cbf = 0.02
        # self.steer_adjust_weight_clf = 0.018
        # self.steer_adjust_weight_clf = 0.01
        self.steer_adjust_weight_clf = 1.0  #default:1.0
        self.steer_adjust_weight_cbf = 1.0

        self.acc_adjust_weight_clf = 200.0

        # set CLF and CBF control law functions
        self.cbf_col_avoid = None
        self.clf_control_law = None
        self.cbf_col_avoid_cont_law = None
        self.cbf_control_law_RD1 = None

        self.cbf_xmax_avoid = None
        self.cbf_control_law_RD2_xmax = None
        self.cbf_control_law_RD2_gauss = None

        self.cbf_col_avoid_btb = None
        self.cbf_col_avoid_LfV = None
        self.obstacle_cbf_value = None

        # Road Boundary CBF
        self.cbf_col_avoid_btb_rb = None
        self.cbf_col_avoid_LfV_rb = None
        self.obstacle_cbf_value_rb = None
        self.cbf_col_avoid_cont_law_rb = None



    # update state variables
    def update_values(self, x, y, yaw, u, v, omega):
        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw
        self._current_vx = u
        self._current_vy = v
        self._current_omega = omega

    # update lyanpunv parameters
    def update_lyapunov_parameter(self, P1, P2, P3, P4, P5, P6):
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3
        self.P4 = P4
        self.P5 = P5
        self.P6 = P6

    # update desired longitudinal speed for the vehicle
    def update_desired_speed(self):
        min_idx = 0
        min_dist = float("inf")
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                self._waypoints[i][0] - self._current_x,
                self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints) - 1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        if desired_speed <= 25:
            self._desired_speed = desired_speed
        else:
            self._desired_speed = min(desired_speed, 25.0) # clip speed smaller than the maximum vel

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints


    def set_obstacle(self, obstacle:Obstacle):
        if (self.obstacle == None):
            self.obstacle = obstacle
        else:
            if (self.obstacle2 == None):
                self.obstacle2 = obstacle
            else:
                if (self.obstacle3 == None):
                    self.obstacle3 = obstacle
                else:
                    if (self.obstacle4 == None):
                        self.obstacle4 = obstacle
                
        
    # Calculate the barrier function of obstacle. (real_num, symbolic)
    def generate_obstacle_barrier(self, x1_ego, x2_ego, obstacle:Obstacle):
        # var_x, var_y : How big the influence is in x and y direction.
        return (obstacle.C/np.sqrt(2*pi))*np.exp(-( ((x1_ego-obstacle.x1_obs)**2)/(2*obstacle.var_x) + ((x2_ego-obstacle.x2_obs)**2)/(2*obstacle.var_y) ))


    # Lyapunov Controller with stanley
    def construct_clf_heur(self):
        self.update_desired_speed()
        # how about connect d_safe with velocity?
        
        v_desired = self._desired_speed
        waypoints = self._waypoints

        # find the reference point index
        closest_index, closest_distance = find_closest_point(self._waypoints, self._current_x, self._current_y)
        self.closest_index = closest_index
        
        # set reference x,y
        x_ref_n = self._waypoints[self.closest_index+1][0]
        y_ref_n = self._waypoints[self.closest_index+1][1]
        
        # set reference yaw
        yaw_path = np.arctan2(waypoints[-1][1] - waypoints[0][1], waypoints[-1][0] - waypoints[0][0])
        self.yaw_path = yaw_path
        # TODO. need to change this as the yaw path in each point

        # calculate the stanley control expect_steer (for the lateral control)
        current_xy = np.array([self._current_x, self._current_y])
        crosstrack_error = np.min(np.sum((current_xy - np.array(waypoints)[:, :2]) ** 2, axis=1))
        yaw_cross_track = np.arctan2(self._current_y - waypoints[0][1], self._current_x - waypoints[0][0])
        yaw_path2ct = yaw_path - yaw_cross_track
        if yaw_path2ct > np.pi:
            yaw_path2ct -= 2 * np.pi
        if yaw_path2ct < - np.pi:
            yaw_path2ct += 2 * np.pi
        if yaw_path2ct > 0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = - abs(crosstrack_error)

        # set stanley steering controller parameter
        k_e = 0.3
        k_v = 20

        # stanley control -> cross track steering
        yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + self._current_speed))

        # correct reference yaw_path into -pi~pi
        if yaw_path > np.pi:
            yaw_path -= 2 * np.pi
        if yaw_path < - np.pi:
            yaw_path += 2 * np.pi

        # print(crosstrack_error, yaw_diff, yaw_diff_crosstrack)
        yaw_diff = yaw_path - self._current_yaw

        # stanley expected steer
        steer_expect = yaw_diff + yaw_diff_crosstrack # the persuing direction of

        # clip stanley expected steer in -1.22~1.22
        while (steer_expect>np.pi) or (steer_expect<-np.pi):
            if steer_expect > np.pi:
                steer_expect -= 2 * np.pi
            if steer_expect < - np.pi:
                steer_expect += 2 * np.pi
        
        steer_expect = min(max_steer, steer_expect)
        steer_expect = max(min_steer, steer_expect)
        self.steer_expect = steer_expect
        # TODO. How about if the steer_expect is small, clamp to 0?

        self.x_ref = self._current_x+self.dt*self._desired_speed
        self.y_ref = y_ref_n
        self.yaw_ref = yaw_path
        self.u_ref = self._desired_speed*np.cos(self.steer_expect)
        self.v_ref = self._desired_speed*np.sin(self.steer_expect)
        self.omega_ref = 0

        # print('ref points: x_ref: ', self.x_ref, ', y_ref:', self.y_ref, ', yaw_ref:', self.yaw_ref, ', u_ref:',self.u_ref, ', v_ref: ', self.v_ref, ', omega_ref: ', self.omega_ref)

        # From here, start clf construct
        # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0

        v_desired = self._desired_speed
        waypoints = self._waypoints

        if self.clf_control_law==None:

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            x_ref = sympy.Symbol('x_ref')
            y_ref = sympy.Symbol('y_ref')
            yaw_ref = sympy.Symbol('yaw_ref')
            u_ref = sympy.Symbol('u_ref')
            v_ref = sympy.Symbol('v_ref')
            omega_ref = sympy.Symbol('omega_ref')

            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()

            V = self.P1*(x-x_ref)**2 + self.P2*(y-y_ref)**2 + self.P3*(yaw-yaw_ref)**2 + self.P4*(u-u_ref)**2 + self.P5*(v-v_ref)**2 + self.P6*(omega-omega_ref)**2
            # print("P1: {}, P2: {}, P3: {}, P4: {}, P5: {}, P6: {}".format(self.P1,self.P2,self.P3,self.P4,self.P5,self.P6))
            self.clf_value = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], V, 'numpy')

            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/u)+(kr/m)*((v-b*omega)/u))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/u) - (b*kr/Iz)*((v-b*omega)/u) )*e_x6
                                
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

            Lg1V = LieDerivative(g_acc, V)
            Lg2V = LieDerivative(g_delta, V)
            LgV = sympy.Matrix([Lg1V, Lg2V])

            LfV = LieDerivative(f, V)
            self.clf_LfV = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], LfV, 'numpy')
            self.LfV = LfV


            LgVTLgV = sympy.transpose(LgV)*LgV
            bTb = LgVTLgV[0]
            self.clf_btb = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], bTb, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            gamma = 800.0
            # -((aaa+np.sqrt(aaa*aaa+gamma*np.matmul(bbb,bbb.T)*np.matmul(bbb,bbb.T)))/(np.matmul(bbb.T,bbb)))*bbb
            
            # Sontag's rule
            input_u = -((LfV + sympy.sqrt(LfV*LfV+gamma*sympy.transpose(bTb)*bTb))/(bTb))*LgV
            # input_u = -((LfV + self.k*V)/(bTb))*LgV # exponential control law - bad performance
            
            # direct derivation from Lyapunov
            # input_u = -(((LfV + 0.1*V))/(bTb))*LgV

            end_transform = time.time()

            start_input_eval = time.time()

            end_input_eval = time.time()

            f = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], input_u, 'numpy') # 'numpy : 137 ms, 

            self.clf_control_law = f
        
        start_lambdify_eval = time.time()
        # [x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref]
        grad_V = self.clf_btb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        grad1_V = self.clf_LfV(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        V_value = self.clf_value(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        # grad_h -> LfV이어야 되는거 아닌가?
        # if np.abs(grad1_V)<=0.00001:
        #     input_val = [0,0]
        self._lyapunov = V_value
        # else:
        input_val = self.clf_control_law(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        # print('*********time for ufuncify function: ', end_ufuncify_eval-start_ufuncify_eval)
        print("grad_V: ", grad_V)
        print('grad LfV', grad1_V)
        print("V value: ", V_value)
        print('(heur)self.throttle : ', input_val[0], ", self.steer: ", input_val[1])
        self.throttle = input_val[0]
        # self.steer = np.fmax(np.fmin(input_val[1], max_steer), min_steer)
        self.steer = input_val[1]*self.steer_adjust_weight_clf
        

    # Lyapunov Controller
    def construct_clf(self):
        self.update_desired_speed()
        # how about connect d_safe with velocity?
        
        v_desired = self._desired_speed
        waypoints = self._waypoints

        # find the reference point index
        closest_index, closest_distance = find_closest_point(self._waypoints, self._current_x, self._current_y)
        self.closest_index = closest_index
        
        # set reference x,y
        x_ref_n = self._waypoints[self.closest_index+1][0]
        y_ref_n = self._waypoints[self.closest_index+1][1]
        
        # set reference yaw
        yaw_path = np.arctan2(waypoints[-1][1] - waypoints[0][1], waypoints[-1][0] - waypoints[0][0])
        self.yaw_path = yaw_path
        # TODO. need to change this as the yaw path in each point

        # calculate the stanley control expect_steer (for the lateral control)
        current_xy = np.array([self._current_x, self._current_y])
        # correct reference yaw_path into -pi~pi
        if yaw_path > np.pi:
            yaw_path -= 2 * np.pi
        if yaw_path < - np.pi:
            yaw_path += 2 * np.pi

        # print(crosstrack_error, yaw_diff, yaw_diff_crosstrack)
        yaw_diff = yaw_path - self._current_yaw

        self.x_ref = self._current_x+self.dt*self._desired_speed
        self.y_ref = y_ref_n
        self.yaw_ref = yaw_path
        self.u_ref = self._desired_speed
        self.v_ref = 0
        self.omega_ref = 0

        # From here, start clf construct
        # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0

        self.update_desired_speed()

        v_desired = self._desired_speed
        waypoints = self._waypoints

        if self.clf_control_law==None:

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            x_ref = sympy.Symbol('x_ref')
            y_ref = sympy.Symbol('y_ref')
            yaw_ref = sympy.Symbol('yaw_ref')
            u_ref = sympy.Symbol('u_ref')
            v_ref = sympy.Symbol('v_ref')
            omega_ref = sympy.Symbol('omega_ref')

            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()

            V1 = self.P4*(u-u_ref)**2 + self.P5*(v-v_ref)**2 + self.P6*(omega-omega_ref)**2
            V2 = self.P1*(x-x_ref)**2 + self.P2*(y-y_ref)**2 + self.P3*(yaw-yaw_ref)**2

            
            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/u)+(kr/m)*((v-b*omega)/u))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/u) - (b*kr/Iz)*((v-b*omega)/u) )*e_x6
            
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

    
            LfV2 = LieDerivative(f, V2)
            V = V1 + self.k*V2 + LfV2

            self.clf_value = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], V, 'numpy')

            LfV = LieDerivative(f, V)

            Lg1V = LieDerivative(g_acc, V)
            Lg2V = LieDerivative(g_delta, V)
            LgV = sympy.Matrix([Lg1V, Lg2V])

            self.clf_LfV = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], LfV, 'numpy')
            self.LfV = LfV

            LgVTLgV = sympy.transpose(LgV)*LgV
            bTb = LgVTLgV[0]
            self.clf_btb = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], bTb, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            # gamma = 2.0
            # -((aaa+np.sqrt(aaa*aaa+gamma*np.matmul(bbb,bbb.T)*np.matmul(bbb,bbb.T)))/(np.matmul(bbb.T,bbb)))*bbb
            input_u = -((LfV + self.k*V)/(bTb))*LgV
            # input_u = -((LfV+sympy.sqrt(LfV*LfV+gamma*bTb*bTb))/(bTb))*LgV
            # print('input_u: ', input_u)

            end_transform = time.time()
            print('time for transforming to CLF input: ', end_transform - start_transform)

            start_input_eval = time.time()

            end_input_eval = time.time()
            print('time for evaluating CLF input: ', end_input_eval - start_input_eval)

            f = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], input_u, 'numpy') # 'numpy : 137 ms, 

            self.clf_control_law = f
        
        start_lambdify_eval = time.time()
        # [x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref]
        grad_V = self.clf_btb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        grad1_V = self.clf_LfV(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        V_value = self.clf_value(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        # grad_h -> LfV이어야 되는거 아닌가?
        # if np.abs(grad1_V)<=0.00001:
        #     input_val = [0,0]
        #     print("grad_V: ", grad_V)
        #     print('grad LfV:', grad1_V)
        #     print("V value: ", V_value)
        # else:
        
        # Prevent input_val become "Nan"
        if grad_V <= 1e-8:
            input_val = [0,0]
        else:
            input_val = self.clf_control_law(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        print('***********input_val:', input_val)
        end_lambdify_eval = time.time()
        print('*********time for lambdify function: ', end_lambdify_eval-start_lambdify_eval)
        # print('*********time for ufuncify function: ', end_ufuncify_eval-start_ufuncify_eval)
        print("grad_V: ", grad_V)
        print('grad LfV', grad1_V)
        print("V value: ", V_value)

        self.throttle = input_val[0]
        # self.steer = np.fmax(np.fmin(input_val[1], max_steer), min_steer)
        self.steer = input_val[1]


    # Lyapunov Controller / testing code (separately calculate input for rel1 and rel2, then add input)
    def construct_clf_sep(self):
        # how about connect d_safe with velocity?
        self.k = 3.5    #default:k = 1,0.5,3
        self.slack_variable = 1.5   #defalut:2,1.5,1


        waypoints = self._waypoints

        # find the reference point index
        closest_index, closest_distance = find_closest_point(self._waypoints, self._current_x, self._current_y)
        self.closest_index = min(closest_index,39)
        print('closest_index:',closest_index)

        lateral_error = (self._current_x-self._waypoints[0][self.closest_index+1])*(-sin(self._waypoints[2][self.closest_index+1])) + ((self._current_y-self._waypoints[1][self.closest_index+1]))*cos(self._waypoints[2][self.closest_index+1]) 
        
        # set reference x,y
        x_ref_n = self._waypoints[0][self.closest_index+1]
        y_ref_n = self._waypoints[1][self.closest_index+1]
        y_ref_n = self._current_y - lateral_error # transform reference y for this code.
        print('!!!lateral error verification!!! y_ref_n: ', y_ref_n, ', current_y: ', self._current_y, ', lateral_error: ', lateral_error)
        
        # yaw transformation
        # yaw_path = np.arctan2(waypoints[-1][1] - waypoints[0][1], waypoints[-1][0] - waypoints[0][0])
        # self.yaw_path = yaw_path

        yaw_path = self._waypoints[2][self.closest_index+1]
        # yaw_path = -1.570796
        # transform the coordinate
        # yaw_path = np.arctan2(waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0])
        # self.yaw_path = yaw_path
        
        v_desired = self._waypoints[3][self.closest_index+1]

        # Add scaling factor -> uniformize the size of errors
        x_y_scaling_factor = 5*((self._current_x-x_ref_n)**2+(self._current_y-y_ref_n)**2)**0.5
        # x_y_scaling_factor = 3
        self.x_y_scaling_factor = x_y_scaling_factor
        # y_scaling_factor = ((self._current_y-y_ref_n)**2)**0.5

        # regularize the size of errors
        x_differ = self._current_x-x_ref_n
        # x_ref_n = self._current_x + x_differ/x_y_scaling_factor
        y_differ = self._current_y-y_ref_n
        # y_ref_n = y_ref_n + y_differ/x_y_scaling_factor

        
        # set reference yaw
        
        self.yaw_path = yaw_path
        # TODO. need to change this as the yaw path in each point
        
        print('(CLF ego) x: ', self._current_x, ', y: ', self._current_y, ', phi: ', self._current_yaw, ', vx: ', self._current_vx, ', vy: ', self._current_vy)
        print('(CLF reference) x:', self._waypoints[0][self.closest_index+1], ', y:', self._waypoints[1][self.closest_index+1], ', phi:', self._waypoints[2][self.closest_index+1], ', vx:', self._waypoints[3][self.closest_index+1], ', vy:', 0)
        print('(CLF reference edited) x:', x_ref_n, ', y:', y_ref_n, ', phi:', yaw_path, ', vx:', self._waypoints[3][self.closest_index+1], ', vy:', 0)


        # calculate the stanley control expect_steer (for the lateral control)
        current_xy = np.array([self._current_x, self._current_y])
        # correct reference yaw_path into -pi~pi
        if yaw_path > np.pi:
            yaw_path -= 2 * np.pi
        if yaw_path < - np.pi:
            yaw_path += 2 * np.pi

        if (yaw_path - self._current_yaw) <= -np.pi:
            if (yaw_path > self._current_yaw):
                self._current_yaw += 2*np.pi
            else:
                yaw_path += 2*np.pi

        self.x_ref = x_ref_n
        self.y_ref = y_ref_n
        self.yaw_ref = yaw_path
        self.u_ref = v_desired
        self.v_ref = 0
        self.omega_ref = 0

        # From here, start clf construct
        # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0

        v_desired = self._desired_speed
        waypoints = self._waypoints

        if self.clf_control_law==None:

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            x_ref = sympy.Symbol('x_ref')
            y_ref = sympy.Symbol('y_ref')
            yaw_ref = sympy.Symbol('yaw_ref')
            u_ref = sympy.Symbol('u_ref')
            v_ref = sympy.Symbol('v_ref')
            omega_ref = sympy.Symbol('omega_ref')
            # xy_scaling_factor = sympy.Symbol('xy_scaling_factor')

            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()

            # self.P1 = sympy.Symbol('P1')
            # self.P2 = sympy.Symbol('P2')
            # self.P3 = sympy.Symbol('P3')
            # self.P4 = sympy.Symbol('P4')
            # self.P5 = sympy.Symbol('P5')
            # self.P6 = sympy.Symbol('P6')

            V1 = self.P4*(u-u_ref)**2 + self.P5*(v-v_ref)**2 + self.P6*(omega-omega_ref)**2
            V2 = self.P1*(x-x_ref)**2 + self.P2*(y-y_ref)**2 + self.P3*(yaw-yaw_ref)**2

            # V1 = self.P4*(u_ref-u)**2 + self.P5*(v_ref-v)**2 + self.P6*(omega_ref-omega)**2
            # V2 = self.P1*(x_ref-x)**2 + self.P2*(y_ref-y)**2 + self.P3*(yaw_ref-yaw)**2

            # transformed Lyapunov
            # V1 = self.P4*(u-u_ref)**2 + self.P5*(v-v_ref)**2 + self.P6*(omega-omega_ref)**2
            
            # TODO the 2nd lyapunov for curved roads have problems. I'm not sure why
            # V2 = self.P1*((x-x_ref)*sympy.cos(yaw_ref)+(y-y_ref)*sympy.sin(yaw_ref))**2 + self.P2*((x-x_ref)*(-sin(yaw_ref))+(y-y_ref)*cos(yaw_ref))**2 + self.P3*(yaw-yaw_ref)**2

            # transformed Lyapunov
            # V1 = self.P4*(u-u_ref)**2 + self.P5*(v-v_ref)**2 + self.P6*(omega-omega_ref)**2
            # V2 = self.P1*((x-x_ref)/xy_scaling_factor*sympy.cos(yaw_ref)+(y-y_ref)/xy_scaling_factor*sympy.sin(yaw_ref))**2 + self.P2*((x-x_ref)/xy_scaling_factor*(-sin(yaw_ref))+(y-y_ref)/xy_scaling_factor*cos(yaw_ref))**2 + self.P3*(yaw-yaw_ref)**2

            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/u)+(kr/m)*((v-b*omega)/u))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/u) - (b*kr/Iz)*((v-b*omega)/u) )*e_x6
            
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

    
            LfV2 = LieDerivative(f, V2)
            V = V1
            V2_new = self.k*V2 + LfV2

            self.clf_value = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], V, 'numpy')
            self.clf_value_2 = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], V2, 'numpy')

            LfV = LieDerivative(f, V)

            Lg1V = LieDerivative(g_acc, V)
            Lg2V = LieDerivative(g_delta, V)
            LgV = sympy.Matrix([Lg1V, Lg2V])

            self.clf_LfV = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], LfV, 'numpy')
            self.LfV = LfV
            

            LgVTLgV = sympy.transpose(LgV)*LgV
            bTb = LgVTLgV[0]
            self.clf_btb = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], bTb, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            # gamma = 2.0
            gamma = 800.0
            # -((aaa+np.sqrt(aaa*aaa+gamma*np.matmul(bbb,bbb.T)*np.matmul(bbb,bbb.T)))/(np.matmul(bbb.T,bbb)))*bbb
            # input_u1 = -((LfV + self.k*V)/(bTb))*LgV
            input_u1 = -((LfV + sympy.sqrt(LfV*LfV+gamma*sympy.transpose(bTb)*bTb))/(self.slack_variable + bTb))*LgV

            # to verify C++
            # input_u1[0].subs([(x, 0.1), (y, 0.1), (yaw, 0.1), (u, 0.1), (v, 0.1), (omega, 0.1), (x_ref, 1), (y_ref, 1), (yaw_ref, 1), (u_ref, 1), (v_ref, 1), (omega_ref, 1)])
            # input_u1[1].subs([(x, 0.1), (y, 0.1), (yaw, 0.1), (u, 0.1), (v, 0.1), (omega, 0.1), (x_ref, 1), (y_ref, 1), (yaw_ref, 1), (u_ref, 1), (v_ref, 1), (omega_ref, 1)])
            # input_u1[0].subs([(x, 0.5), (y, 0.5), (yaw, 0.5), (u, 0.5), (v, 0.5), (omega, 0.5), (x_ref, 1), (y_ref, 1), (yaw_ref, 1), (u_ref, 1), (v_ref, 1), (omega_ref, 1)])
            # input_u1[1].subs([(x, 0.5), (y, 0.5), (yaw, 0.5), (u, 0.5), (v, 0.5), (omega, 0.5), (x_ref, 1), (y_ref, 1), (yaw_ref, 1), (u_ref, 1), (v_ref, 1), (omega_ref, 1)])

            # input_u = -((LfV+sympy.sqrt(LfV*LfV+gamma*bTb*bTb))/(bTb))*LgV

            # calculate input for V2_new
            self.clf_value_V2_new = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], V2_new, 'numpy')

            LfV2_new = LieDerivative(f, V2_new)

            Lg1V2_new = LieDerivative(g_acc, V2_new)
            Lg2V2_new = LieDerivative(g_delta, V2_new)
            LgV2_new = sympy.Matrix([Lg1V2_new, Lg2V2_new])

            self.clf_LfV2_new = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], LfV2_new, 'numpy')
            self.LfV2_new = LfV2_new

            LgVTLgV2_new = sympy.transpose(LgV2_new)*LgV2_new
            bTbV2_new = LgVTLgV2_new[0]
            self.clf_btbV2_new = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], bTbV2_new, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            input_u2 = -((LfV2_new + self.k*V2_new)/(self.slack_variable + bTbV2_new))*LgV2_new
            # input_u2 = -0.003*LgV2_new

            # to verify C++
            # input_u2[0].subs([(x, 0.1), (y, 0.1), (yaw, 0.1), (u, 0.1), (v, 0.1), (omega, 0.1), (x_ref, 1), (y_ref, 1), (yaw_ref, 1), (u_ref, 1), (v_ref, 1), (omega_ref, 1)])
            # input_u2[1].subs([(x, 0.1), (y, 0.1), (yaw, 0.1), (u, 0.1), (v, 0.1), (omega, 0.1), (x_ref, 1), (y_ref, 1), (yaw_ref, 1), (u_ref, 1), (v_ref, 1), (omega_ref, 1)])
            # input_u2[0].subs([(x, 0.5), (y, 0.5), (yaw, 0.5), (u, 0.5), (v, 0.5), (omega, 0.5), (x_ref, 1), (y_ref, 1), (yaw_ref, 1), (u_ref, 1), (v_ref, 1), (omega_ref, 1)])
            # input_u2[1].subs([(x, 0.5), (y, 0.5), (yaw, 0.5), (u, 0.5), (v, 0.5), (omega, 0.5), (x_ref, 1), (y_ref, 1), (yaw_ref, 1), (u_ref, 1), (v_ref, 1), (omega_ref, 1)])


            if (input_u1 == nan):
                input_u = input_u2
            elif (input_u2 == nan):
                input_u = input_u1
            else:
                input_u = input_u1+input_u2
            
            self.input_u1 = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], input_u1, 'numpy')
            self.input_u2 = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], input_u2, 'numpy')

            # print('input_u1: ', input_u1, '\n, input_u2: ', input_u2)
            # print('input_u: ', input_u)

            end_transform = time.time()
            print('time for transforming to CLF input: ', end_transform - start_transform)

            start_input_eval = time.time()

            end_input_eval = time.time()
            print('time for evaluating CLF input: ', end_input_eval - start_input_eval)

            f = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], input_u, 'numpy') # 'numpy : 137 ms, 

            self.clf_control_law = f
        
        start_lambdify_eval = time.time()
        # [x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref]
        # grad_V = self.clf_btb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        # grad1_V = self.clf_LfV(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        V_value = self.clf_value(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref) \
            + self.clf_value_2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        # grad_h -> LfV이어야 되는거 아닌가?
        # if np.abs(grad1_V)<=0.00001:
        #     input_val = [0,0]
        #     print("grad_V: ", grad_V)
        #     print('grad LfV:', grad1_V)
        #     print("V value: ", V_value)
        # else:
        
        self._lyapunov = V_value
        
        # Prevent input_val become "Nan"
        # if grad_V <= 1e-8:
        #     input_val = [0,0]
        # else:

        # print('input_u: ', input_u)

        if (self._current_vx <= 0.00001):
            self._current_vx = 0.00001
        # input_val = self.clf_control_law(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        input_val1 = self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        input_val2 = self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        
        # print('input_val: ', input_val)
        # print('input_u1: ', self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref), ', input_u2: ', self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref))
        # print('input_u2: ', self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref), ', input_u2: ', self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref))
        

        print('longitudinal error: ', self._current_x - self.x_ref)
        print('lateral error: ', self._current_y - self.y_ref)
        print("yaw error: ", self._current_yaw - self.yaw_ref)
        # print('***********input_val:', input_val)
        end_lambdify_eval = time.time()
        print('*********time for lambdify function: ', end_lambdify_eval-start_lambdify_eval)
        # print('*********time for ufuncify function: ', end_ufuncify_eval-start_ufuncify_eval)
        # print("grad_V: ", grad_V)
        # print('grad LfV', grad1_V)
        print("V value: ", V_value)
        
        self.throttle = input_val1[0]
        # self.steer = np.fmax(np.fmin(input_val1[1], max_steer), min_steer)
        self.steer = input_val2[1]

# Lyapunov Controller / testing code (separately calculate input for rel1 and rel2, then add input)
    def construct_clf_sep_mix(self):
        self.update_desired_speed()
        # how about connect d_safe with velocity?
        self.k = 1
        
        v_desired = self._desired_speed
        waypoints = self._waypoints

        # find the reference point index
        closest_index, closest_distance = find_closest_point(self._waypoints, self._current_x, self._current_y)
        self.closest_index = closest_index
        
        # set reference x,y
        x_ref_n = self._waypoints[self.closest_index+1][0]
        y_ref_n = self._waypoints[self.closest_index+1][1]
        
        # set reference yaw
        yaw_path = np.arctan2(waypoints[-1][1] - waypoints[0][1], waypoints[-1][0] - waypoints[0][0])
        self.yaw_path = yaw_path
        # TODO. need to change this as the yaw path in each point

        # calculate the stanley control expect_steer (for the lateral control)
        current_xy = np.array([self._current_x, self._current_y])
        # correct reference yaw_path into -pi~pi
        if yaw_path > np.pi:
            yaw_path -= 2 * np.pi
        if yaw_path < - np.pi:
            yaw_path += 2 * np.pi

        # print(crosstrack_error, yaw_diff, yaw_diff_crosstrack)
        yaw_diff = yaw_path - self._current_yaw

        self.x_ref = self._current_x+self.dt*self._desired_speed
        self.y_ref = y_ref_n
        self.yaw_ref = yaw_path
        self.u_ref = self._desired_speed
        self.v_ref = 0
        self.omega_ref = 0

        # From here, start clf construct
        # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0

        self.update_desired_speed()

        v_desired = self._desired_speed
        waypoints = self._waypoints

        if self.clf_control_law==None:

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            x_ref = sympy.Symbol('x_ref')
            y_ref = sympy.Symbol('y_ref')
            yaw_ref = sympy.Symbol('yaw_ref')
            u_ref = sympy.Symbol('u_ref')
            v_ref = sympy.Symbol('v_ref')
            omega_ref = sympy.Symbol('omega_ref')

            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()

            h = self.d_safe**2 - (x-self.x_obs)**2 - (y-self.y_obs)**2

            V1 = self.P4*(u-u_ref)**2 + self.P5*(v-v_ref)**2 + self.P6*(omega-omega_ref)**2
            V2 = 10*(self.P1*(x-x_ref)**2 + self.P2*(y-y_ref)**2 + self.P3*(yaw-yaw_ref)**2) + 6.0*(sympy.log(1+sympy.exp(0.1*(h))))

            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/(u+0.001))+(kr/m)*((v-b*omega)/(u+0.001)))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/(u+0.001)) - (b*kr/Iz)*((v-b*omega)/(u+0.001)) )*e_x6
            
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

    
            LfV2 = LieDerivative(f, V2)
            V = V1
            V2_new = self.k*V2 + LfV2

            self.clf_value = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], V, 'numpy')

            LfV = LieDerivative(f, V)

            Lg1V = LieDerivative(g_acc, V)
            Lg2V = LieDerivative(g_delta, V)
            LgV = sympy.Matrix([Lg1V, Lg2V])

            self.clf_LfV = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], LfV, 'numpy')
            self.LfV = LfV

            LgVTLgV = sympy.transpose(LgV)*LgV
            bTb = LgVTLgV[0]
            self.clf_btb = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], bTb, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            # gamma = 2.0
            gamma = 800.0
            # -((aaa+np.sqrt(aaa*aaa+gamma*np.matmul(bbb,bbb.T)*np.matmul(bbb,bbb.T)))/(np.matmul(bbb.T,bbb)))*bbb
            # input_u1 = -((LfV + self.k*V)/(bTb))*LgV
            input_u1 = -((LfV + sympy.sqrt(LfV*LfV+gamma*sympy.transpose(bTb)*bTb))/(self.slack_variable + bTb))*LgV
            # input_u = -((LfV+sympy.sqrt(LfV*LfV+gamma*bTb*bTb))/(bTb))*LgV

            # calculate input for V2_new
            self.clf_value_V2_new = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], V2_new, 'numpy')

            LfV2_new = LieDerivative(f, V2_new)

            Lg1V2_new = LieDerivative(g_acc, V2_new)
            Lg2V2_new = LieDerivative(g_delta, V2_new)
            LgV2_new = sympy.Matrix([Lg1V2_new, Lg2V2_new])

            self.clf_LfV2_new = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], LfV2_new, 'numpy')
            self.LfV2_new = LfV2_new

            LgVTLgV2_new = sympy.transpose(LgV2_new)*LgV2_new
            bTbV2_new = LgVTLgV2_new[0]
            self.clf_btbV2_new = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], bTbV2_new, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            input_u2 = -((LfV2_new + self.k*V2_new)/(self.slack_variable + bTbV2_new))*LgV2_new

            if (input_u1 == nan):
                input_u = input_u2
            elif (input_u2 == nan):
                input_u = input_u1
            else:
                input_u = input_u1+input_u2
            
            self.input_u1 = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], input_u1, 'numpy')
            self.input_u2 = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], input_u2, 'numpy')

            print('input_u1: ', input_u1, '\n input_u2: ', input_u2)
            print('input_u: ', input_u)

            end_transform = time.time()
            print('time for transforming to CLF input: ', end_transform - start_transform)

            start_input_eval = time.time()

            end_input_eval = time.time()
            print('time for evaluating CLF input: ', end_input_eval - start_input_eval)

            f = lambdify([x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref], input_u, 'numpy') # 'numpy : 137 ms, 

            self.clf_control_law = f
        
        start_lambdify_eval = time.time()
        # [x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref]
        grad_V = self.clf_btb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        grad1_V = self.clf_LfV(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        V_value = self.clf_value(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        # grad_h -> LfV이어야 되는거 아닌가?
        # if np.abs(grad1_V)<=0.00001:
        #     input_val = [0,0]
        #     print("grad_V: ", grad_V)
        #     print('grad LfV:', grad1_V)
        #     print("V value: ", V_value)
        # else:
        
        self._lyapunov = V_value
        
        # Prevent input_val become "Nan"
        # if grad_V <= 1e-8:
        #     input_val = [0,0]
        # else:

        print('input_u1: ', self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref), '\n, input_u2: ', self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref))
        # print('input_u: ', input_u)

        input_val = self.clf_control_law(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)
        print('***********input_val:', input_val)
        end_lambdify_eval = time.time()
        print('*********time for lambdify function: ', end_lambdify_eval-start_lambdify_eval)
        # print('*********time for ufuncify function: ', end_ufuncify_eval-start_ufuncify_eval)
        print("grad_V: ", grad_V)
        print('grad LfV', grad1_V)
        print("V value: ", V_value)
        
        self.throttle = input_val[0]
        # self.steer = np.fmax(np.fmin(input_val[1], max_steer), min_steer)
        self.steer = input_val[1]

    # Velocity CBF control law - original log version
    def construct_cbf_RD1(self, vmax, strength = 0.01, type='max'):
        self.update_desired_speed()

        # From here, start clf construct
        # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0
        self.default_acc = 1.0

        if self.cbf_control_law_RD1==None:
            self.v_max = vmax

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            u_max = sympy.Symbol('u_max')

            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()

            V = -strength*(sympy.log(-(u-u_max))+sympy.log(u_max))
            # print("P1: {}, P2: {}, P3: {}, P4: {}, P5: {}, P6: {}".format(self.P1,self.P2,self.P3,self.P4,self.P5,self.P6))
            self.cbfRD1_value = lambdify([x, y, yaw, u, v, omega, u_max], V, 'numpy')

            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/u)+(kr/m)*((v-b*omega)/u))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/u) - (b*kr/Iz)*((v-b*omega)/u) )*e_x6
                                
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

            Lg1V = LieDerivative(g_acc, V)
            Lg2V = LieDerivative(g_delta, V)
            LgV = sympy.Matrix([Lg1V, Lg2V])

            LfV = LieDerivative(f, V)
            self.cbfRD1_LfV = lambdify([x, y, yaw, u, v, omega, u_max], LfV, 'numpy')
            self.LfBRD1 = LfV

            LgVTLgV = sympy.transpose(LgV)*LgV
            bTb = LgVTLgV[0]
            self.cbfRD1_btb = lambdify([x, y, yaw, u, v, omega, u_max], bTb, 'numpy')

            gamma = 100.0
            # -((aaa+np.sqrt(aaa*aaa+gamma*np.matmul(bbb,bbb.T)*np.matmul(bbb,bbb.T)))/(np.matmul(bbb.T,bbb)))*bbb
            input_u = -((LfV + sympy.sqrt(LfV*LfV+gamma*sympy.transpose(bTb)*bTb))/(bTb))*LgV
            # input_u = -((LfV + self.k*V)/(bTb))*LgV # exponential control law - bad performance
            f = lambdify([x, y, yaw, u, v, omega, u_max], input_u, 'numpy') # 'numpy : 137 ms, 

            self.cbf_control_law_RD1 = f
        
        start_lambdify_eval = time.time()
        # [x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref]
        grad_V = self.cbfRD1_btb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.v_max)
        grad1_V = self.cbfRD1_LfV(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.v_max)
        V_value = self.cbfRD1_value(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.v_max)
        # grad_h -> LfV이어야 되는거 아닌가?
        # if np.abs(grad1_V)<=0.00001:
        #     input_val = [0,0]
        self._lyapunov = V_value
        # else:
        input_val = self.cbf_control_law_RD1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.v_max)
        print("(cbf rd1) states: ", self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.v_max)
        print('***********input_val:', input_val)
        end_lambdify_eval = time.time()
        print('(cbf rd1) time for calculation: ', end_lambdify_eval-start_lambdify_eval)
        # print('*********time for ufuncify function: ', end_ufuncify_eval-start_ufuncify_eval)
        print("(cbf rd1) grad_V: ", grad_V)
        print('(cbf rd1) grad LfV', grad1_V)
        print("(cbf rd1) V value: ", V_value)
        print('(cbf rd1) self.throttle : ', input_val[0], ", self.steer: ", input_val[1])
        
        self.throttle = input_val[0]
        # self.steer = np.fmax(np.fmin(input_val[1], max_steer), min_steer)
        self.steer = input_val[1]
        
    # log and exponential version
    def construct_cbf_RD1_logexp(self, vmax, strength = 1, type='max'):
        self.update_desired_speed()

        # From here, start clf construct
        # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0
        self.default_acc = 1.0
        self.cbf_on = False

        self.exponen_stiff = 30

        if self.cbf_control_law_RD1==None:
            self.v_max = vmax

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            u_max = sympy.Symbol('u_max')

            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()

            V = strength*(sympy.log(1+sympy.exp(self.exponen_stiff*(u-u_max))))

            # V = -strength*(sympy.log(-(u-u_max))+sympy.log(u_max))
            # print("P1: {}, P2: {}, P3: {}, P4: {}, P5: {}, P6: {}".format(self.P1,self.P2,self.P3,self.P4,self.P5,self.P6))
            self.cbfRD1_value = lambdify([x, y, yaw, u, v, omega, u_max], V, 'numpy')

            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/u)+(kr/m)*((v-b*omega)/u))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/u) - (b*kr/Iz)*((v-b*omega)/u) )*e_x6
                                
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

            Lg1V = LieDerivative(g_acc, V)
            Lg2V = LieDerivative(g_delta, V)
            LgV = sympy.Matrix([Lg1V, Lg2V])

            LfV = LieDerivative(f, V)
            self.cbfRD1_LfV = lambdify([x, y, yaw, u, v, omega, u_max], LfV, 'numpy')
            self.LfBRD1 = LfV

            LgVTLgV = sympy.transpose(LgV)*LgV
            bTb = LgVTLgV[0]
            self.cbfRD1_btb = lambdify([x, y, yaw, u, v, omega, u_max], bTb, 'numpy')

            gamma = 100.0
            # -((aaa+np.sqrt(aaa*aaa+gamma*np.matmul(bbb,bbb.T)*np.matmul(bbb,bbb.T)))/(np.matmul(bbb.T,bbb)))*bbb
            input_u = -((LfV + sympy.sqrt(LfV*LfV+gamma*sympy.transpose(bTb)*bTb))/(self.slack_variable + bTb))*LgV
            # input_u = -((LfV + self.k*V)/(bTb))*LgV # exponential control law - bad performance
            f = lambdify([x, y, yaw, u, v, omega, u_max], input_u, 'numpy') # 'numpy : 137 ms, 

            self.cbf_control_law_RD1 = f

            print('input_u: ', input_u)
            print('input_u_acc: ', input_u[0])
            print('input_u_steer: ', input_u[1])
        
        start_lambdify_eval = time.time()
        # [x, y, yaw, u, v, omega, x_ref, y_ref, yaw_ref, u_ref, v_ref, omega_ref]
        grad_V = self.cbfRD1_btb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.v_max)
        grad1_V = self.cbfRD1_LfV(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.v_max)
        V_value = self.cbfRD1_value(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.v_max)
        # grad_h -> LfV이어야 되는거 아닌가?
        # if np.abs(grad1_V)<=0.00001:
        #     input_val = [0,0]
        self._lyapunov = V_value
        # else:
        input_val = self.cbf_control_law_RD1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.v_max)
        print("(cbf rd1) states: ", self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.v_max)
        print('***********input_val:', input_val)
        end_lambdify_eval = time.time()
        print('(cbf rd1) time for calculation: ', end_lambdify_eval-start_lambdify_eval)
        # print('*********time for ufuncify function: ', end_ufuncify_eval-start_ufuncify_eval)
        print("(cbf rd1) grad_V: ", grad_V)
        print('(cbf rd1) grad LfV', grad1_V)
        print("(cbf rd1) V value: ", V_value)
        print('(cbf rd1) self.throttle : ', input_val[0], ", self.steer: ", input_val[1])
        
        self.throttle = input_val[0]
        # self.steer = np.fmax(np.fmin(input_val[1], max_steer), min_steer)
        self.steer = input_val[1]
        

# log and exponential version
    def construct_cbf_RD2_gauss(self, x_obs_n, y_obs_n):
                # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0
        self.update_desired_speed()
        width = 4.5
        height = 2.0
        self.cbf_on = False

        if self.cbf_col_avoid_cont_law==None:

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            x_obs = sympy.Symbol('x_obs')
            y_obs = sympy.Symbol('y_obs')
            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()
            # sympy.exp()

            # Set Obstacle CBF
            # V = sympy.exp(-(1/(width**2 - (x-x_obs)**2) + 1/(height**2 - (y-y_obs)**2)))
            # V = -(x-x_obs)**2 -(y-y_obs)**2 +self.d_safe**2 # original constraint

            alpha = 1.0
            u_obs = 0
            v_obs = 0
            # V_t = alpha*((u_obs-u)*(x-x_obs)/(sympy.Abs(x-x_obs)) + (v_obs-v)*(y-y_obs)/(sympy.Abs(y-y_obs)))
            # V_t = 1.0

            # # add front gaussian and rear gaussian
            V_front = (self.obstacle.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs-width/3)**2)/(2*self.obstacle.var_x) + ((y-y_obs)**2)/(2*self.obstacle.var_y) ))
            V_rear = (self.obstacle.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs+width/3)**2)/(2*self.obstacle.var_x) + ((y-y_obs)**2)/(2*self.obstacle.var_y) ))
            V = V_front+V_rear

            # add front gaussian and rear gaussian
            # V_front = (self.obstacle.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs-width/3)**2)/(2*self.obstacle.var_x) + ((y-y_obs+height/2)**2)/(2*self.obstacle.var_y) ))
            # V_rear = (self.obstacle.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs+width/3)**2)/(2*self.obstacle.var_x) + ((y-y_obs-height/2)**2)/(2*self.obstacle.var_y) ))
            # V_front2 = (self.obstacle.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs-width/3)**2)/(2*self.obstacle.var_x) + ((y-y_obs-height/2)**2)/(2*self.obstacle.var_y) ))
            # V_rear2 = (self.obstacle.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs+width/3)**2)/(2*self.obstacle.var_x) + ((y-y_obs+height/2)**2)/(2*self.obstacle.var_y) ))
            # V = V_front+V_rear+V_front2+V_rear2
            self.obstacle_cbf_value = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs], V, 'numpy')

            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/u)+(kr/m)*((v-b*omega)/u))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/u) - (b*kr/Iz)*((v-b*omega)/u) )*e_x6
                                
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

            LfV = LieDerivative(f, V)
            self.cbf_col_avoid_LfV = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs], LfV, 'numpy')
            self.LfV = LfV

            # second derivative
            LfLfV = LieDerivative(f, LfV)
            LfLg1V = LieDerivative(g_acc, LfV)
            LfLg2V = LieDerivative(g_delta, LfV)
            LfLgV = sympy.Matrix([LfLg1V, LfLg2V])

            LgVTLgV = sympy.transpose(LfLgV)*LfLgV
            bTb = LgVTLgV[0]
            self.cbf_col_avoid_btb = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs], bTb, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            input_u = -(LfLfV/(self.slack_variable+bTb))*LfLgV - (1+V)*((2*self.k*LfV + (self.k**2)*V)/(self.slack_variable+bTb))*LfLgV
            # input_u = -(LfLfV/(self.slack_variable+bTb))*LfLgV - (V)*((2*self.k*LfV + (self.k**2)*V)/(self.slack_variable+bTb))*LfLgV
            # input_u = -(LfLfV/(self.slack_variable+bTb))*LfLgV - (1)*((2*self.k*LfV + (self.k**2)*V)/(self.slack_variable+bTb))*LfLgV
            # - nonlinear term - exponential term
            # input_u =- (1+V)*((2*self.k*LfV + (self.k**2)*V)/(self.slack_variable+bTb))*LfLgV

            f = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs], input_u, 'numpy') # 'numpy : 137 ms, 

            self.cbf_col_avoid_cont_law = f
        
        start_lambdify_eval = time.time()
        grad_h = self.cbf_col_avoid_btb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, x_obs_n, y_obs_n)
        grad1_h = self.cbf_col_avoid_LfV(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, x_obs_n, y_obs_n)
        h_value = self.obstacle_cbf_value(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, x_obs_n, y_obs_n)
        self._barrier = h_value
        # grad_h -> LfV이어야 되는거 아닌가?
        # self.cbf_on = True
        input_val = self.cbf_col_avoid_cont_law(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, x_obs_n, y_obs_n)
        print("grad_h: ", grad_h)
        print('grad LfV', grad1_h)
        print("h value: ", h_value)
        self.throttle = input_val[0]
        # self.steer = np.fmax(np.fmin(input_val[1], max_steer), min_steer)
        self.steer = input_val[1]
        print('(gauss cbf) throtte: ', self.throttle, ', steer: ', self.steer)

# log and exponential version
    def construct_cbf_RD2_gauss_multiple(self, x_obs_n, y_obs_n,x_obs_n2, y_obs_n2,x_obs_n3, y_obs_n3,x_obs_n4, y_obs_n4):
                # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0
        self.update_desired_speed()
        width = 4.5
        height = 2.0
        self.cbf_on = False

        if self.cbf_col_avoid_cont_law==None:

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            x_obs = sympy.Symbol('x_obs')
            y_obs = sympy.Symbol('y_obs')
            x_obs2 = sympy.Symbol('x_obs2')
            y_obs2 = sympy.Symbol('y_obs2')
            x_obs3 = sympy.Symbol('x_obs3')
            y_obs3 = sympy.Symbol('y_obs3')
            x_obs4 = sympy.Symbol('x_obs4')
            y_obs4 = sympy.Symbol('y_obs4')
            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()
            # sympy.exp()

            # Set Obstacle CBF
            # V = sympy.exp(-(1/(width**2 - (x-x_obs)**2) + 1/(height**2 - (y-y_obs)**2)))
            # V = -(x-x_obs)**2 -(y-y_obs)**2 +self.d_safe**2 # original constraint

            alpha = 1.0
            u_obs = 0
            v_obs = 0
            # V_t = alpha*((u_obs-u)*(x-x_obs)/(sympy.Abs(x-x_obs)) + (v_obs-v)*(y-y_obs)/(sympy.Abs(y-y_obs)))
            # V_t = 1.0

            V1_front = (self.obstacle.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs+width/2)**2)/(2*self.obstacle.var_x) + ((y-y_obs)**2)/(2*self.obstacle.var_y) ))
            V1_rear = (self.obstacle.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs-width/2)**2)/(2*self.obstacle.var_x) + ((y-y_obs)**2)/(2*self.obstacle.var_y) ))
            V2_front = (self.obstacle2.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs2+width/2)**2)/(2*self.obstacle2.var_x) + ((y-y_obs2)**2)/(2*self.obstacle2.var_y)))
            V2_rear = (self.obstacle2.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs2-width/2)**2)/(2*self.obstacle2.var_x) + ((y-y_obs2)**2)/(2*self.obstacle2.var_y)))
            V3_front = (self.obstacle3.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs3+width/2)**2)/(2*self.obstacle3.var_x) + ((y-y_obs3)**2)/(2*self.obstacle3.var_y)))
            V3_rear = (self.obstacle3.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs3-width/2)**2)/(2*self.obstacle3.var_x) + ((y-y_obs3)**2)/(2*self.obstacle3.var_y)))
            V4_front = (self.obstacle.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs4+width/2)**2)/(2*self.obstacle4.var_x) + ((y-y_obs4)**2)/(2*self.obstacle4.var_y)))
            V4_rear = (self.obstacle.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs4-width/2)**2)/(2*self.obstacle4.var_x) + ((y-y_obs4)**2)/(2*self.obstacle4.var_y)))
            V = V1_front+V1_rear+V2_front+V2_rear+V3_front+V3_rear+V4_front+V4_rear
            self.obstacle_cbf_value = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs, x_obs2, y_obs2, x_obs3, y_obs3, x_obs4, y_obs4], V, 'numpy')

            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/u)+(kr/m)*((v-b*omega)/u))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/u) - (b*kr/Iz)*((v-b*omega)/u) )*e_x6
                                
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

            LfV = LieDerivative(f, V)
            self.cbf_col_avoid_LfV = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs, x_obs2, y_obs2, x_obs3, y_obs3, x_obs4, y_obs4], LfV, 'numpy')
            self.LfV = LfV

            # second derivative
            LfLfV = LieDerivative(f, LfV)
            LfLg1V = LieDerivative(g_acc, LfV)
            LfLg2V = LieDerivative(g_delta, LfV)
            LfLgV = sympy.Matrix([LfLg1V, LfLg2V])

            LgVTLgV = sympy.transpose(LfLgV)*LfLgV
            bTb = LgVTLgV[0]
            self.cbf_col_avoid_btb = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs, x_obs2, y_obs2, x_obs3, y_obs3, x_obs4, y_obs4], bTb, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            input_u = -(LfLfV/(self.slack_variable+bTb))*LfLgV - (1+V)*((2*self.k*LfV + (self.k**2)*V)/(self.slack_variable+bTb))*LfLgV
            # - nonlinear term - exponential term

            f = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs, x_obs2, y_obs2, x_obs3, y_obs3, x_obs4, y_obs4], input_u, 'numpy') # 'numpy : 137 ms, 

            self.cbf_col_avoid_cont_law = f
        
        start_lambdify_eval = time.time()
        grad_h = self.cbf_col_avoid_btb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, x_obs_n, y_obs_n, x_obs_n2, y_obs_n2,x_obs_n3, y_obs_n3,x_obs_n4, y_obs_n4)
        grad1_h = self.cbf_col_avoid_LfV(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, x_obs_n, y_obs_n,x_obs_n2, y_obs_n2,x_obs_n3, y_obs_n3,x_obs_n4, y_obs_n4)
        h_value = self.obstacle_cbf_value(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, x_obs_n, y_obs_n,x_obs_n2, y_obs_n2,x_obs_n3, y_obs_n3,x_obs_n4, y_obs_n4)
        self._barrier = h_value
        # grad_h -> LfV이어야 되는거 아닌가?
        # self.cbf_on = True
        input_val = self.cbf_col_avoid_cont_law(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, x_obs_n, y_obs_n,x_obs_n2, y_obs_n2,x_obs_n3, y_obs_n3,x_obs_n4, y_obs_n4)
        print("grad_h: ", grad_h)
        print('grad LfV', grad1_h)
        print("h value: ", h_value)
        self.throttle = input_val[0]
        # self.steer = np.fmax(np.fmin(input_val[1], max_steer), min_steer)
        self.steer = input_val[1]
        print('(gauss cbf) throtte: ', self.throttle, ', steer: ', self.steer)


# log and exponential version
    def construct_cbf_RD2_logexp_collision(self, strength=10, exponen_stiff = 1.5):
                # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0
        self.update_desired_speed()
        width = 4.5
        height = 2.0
        self.cbf_on = False

        if self.cbf_col_avoid_cont_law==None:

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            x_obs = sympy.Symbol('x_obs')
            y_obs = sympy.Symbol('y_obs')
            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()
            # sympy.exp()

            # Set Obstacle CBF
            # V = sympy.exp(-(1/(width**2 - (x-x_obs)**2) + 1/(height**2 - (y-y_obs)**2)))
            # V = -(x-x_obs)**2 -(y-y_obs)**2 +self.d_safe**2 # original constraint

            # V = (self.obstacle.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs)**2)/(2*self.obstacle.var_x) + ((y-y_obs)**2)/(2*self.obstacle.var_y) ))
            # Set Obstacle CBF
            h = self.d_safe**2 - (x-x_obs)**2 - (y-y_obs)**2
            V = strength*(sympy.log(1+sympy.exp(exponen_stiff*(h))))
            self.obstacle_cbf_value = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs], V, 'numpy')

            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/u)+(kr/m)*((v-b*omega)/u))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/u) - (b*kr/Iz)*((v-b*omega)/u) )*e_x6
                                
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

            LfV = LieDerivative(f, V)
            self.cbf_col_avoid_LfV = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs], LfV, 'numpy')
            self.LfV = LfV

            # second derivative
            LfLfV = LieDerivative(f, LfV)
            LfLg1V = LieDerivative(g_acc, LfV)
            LfLg2V = LieDerivative(g_delta, LfV)
            LfLgV = sympy.Matrix([LfLg1V, LfLg2V])

            LgVTLgV = sympy.transpose(LfLgV)*LfLgV
            bTb = LgVTLgV[0]
            self.cbf_col_avoid_btb = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs], bTb, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            input_u = -((LfLfV + 2*self.k*LfV + (self.k**2)*V)/(self.slack_variable+bTb))*LfLgV

            f = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs], input_u, 'numpy') # 'numpy : 137 ms, 

            self.cbf_col_avoid_cont_law = f
        
        start_lambdify_eval = time.time()
        grad_h = self.cbf_col_avoid_btb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_obs, self.y_obs)
        grad1_h = self.cbf_col_avoid_LfV(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_obs, self.y_obs)
        h_value = self.obstacle_cbf_value(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_obs, self.y_obs)
        self._barrier = h_value
        # grad_h -> LfV이어야 되는거 아닌가?
        if np.abs(h_value)<=0.000000000001:
            input_val = [0,0]
            print("(explog cbf) grad_h: ", grad_h)
            print('(explog cbf) grad LfV:', grad1_h)
            print("(explog cbf) h value: ", h_value)
            self.throttle = 0
            self.steer = 0
            print('(explog cbf) throtte: ', self.throttle, ', steer: ', self.steer)
            self.cbf_on = False
        else:
            self.cbf_on = True
            input_val = self.cbf_col_avoid_cont_law(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_obs, self.y_obs)
            print("(explog cbf) grad_h: ", grad_h)
            print('(explog cbf) grad LfV', grad1_h)
            print("(explog cbf) h value: ", h_value)
            self.throttle = input_val[0]
            # self.steer = np.fmax(np.fmin(input_val[1], max_steer), min_steer)
            self.steer = input_val[1]
            print('(explog cbf) throtte: ', self.throttle, ', steer: ', self.steer)


# log and exponential version for RD1 (testing)
    def construct_cbf_RD1_logexp_collision(self, strength=10, exponen_stiff = 0.1):
                # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0
        self.update_desired_speed()
        width = 4.5
        height = 2.0
        self.cbf_on = False

        if self.cbf_col_avoid_cont_law==None:

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            x_obs = sympy.Symbol('x_obs')
            y_obs = sympy.Symbol('y_obs')
            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()
            # sympy.exp()

            # Set Obstacle CBF
            # V = sympy.exp(-(1/(width**2 - (x-x_obs)**2) + 1/(height**2 - (y-y_obs)**2)))
            # V = -(x-x_obs)**2 -(y-y_obs)**2 +self.d_safe**2 # original constraint

            # V = (self.obstacle.C/sympy.sqrt(2*pi))*sympy.exp(-( ((x-x_obs)**2)/(2*self.obstacle.var_x) + ((y-y_obs)**2)/(2*self.obstacle.var_y) ))
            # Set Obstacle CBF
            
            # h = self.d_safe**2 - (x-x_obs)**2 - (y-y_obs)**2 # new version
            h = (self.d_safe**2 - (x-x_obs)**2 - (y-y_obs)**2) # new version
            V = strength*(sympy.log(1+sympy.exp(exponen_stiff*(h))))
            self.obstacle_cbf_value = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs], V, 'numpy')

            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/u)+(kr/m)*((v-b*omega)/u))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/u) - (b*kr/Iz)*((v-b*omega)/u) )*e_x6
                                
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

            LfV = LieDerivative(f, V)
            Lg1V = LieDerivative(g_acc, V)
            Lg2V = LieDerivative(g_delta, V)
            LgV = sympy.Matrix([Lg1V, Lg2V])
            self.cbf_col_avoid_LfV = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs], LfV, 'numpy')
            self.LfV = LfV

            # second derivative
            LfLfV = LieDerivative(f, LfV)
            LfLg1V = LieDerivative(g_acc, LfV)
            LfLg2V = LieDerivative(g_delta, LfV)
            LfLgV = sympy.Matrix([LfLg1V, LfLg2V])

            LgVTLgV = sympy.transpose(LfLgV)*LfLgV
            bTb = LgVTLgV[0]
            self.cbf_col_avoid_btb = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs], bTb, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            gamma = 2.0
            # -((aaa+np.sqrt(aaa*aaa+gamma*np.matmul(bbb,bbb.T)*np.matmul(bbb,bbb.T)))/(np.matmul(bbb.T,bbb)))*bbb
            input_u = -((LfV + sympy.sqrt(LfV*LfV+gamma*sympy.transpose(bTb)*bTb))/(self.slack_variable + bTb))*LgV

            f = lambdify([x, y, yaw, u, v, omega, x_obs, y_obs], input_u, 'numpy') # 'numpy : 137 ms, 

            self.cbf_col_avoid_cont_law = f
        
        start_lambdify_eval = time.time()
        grad_h = self.cbf_col_avoid_btb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_obs, self.y_obs)
        grad1_h = self.cbf_col_avoid_LfV(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_obs, self.y_obs)
        h_value = self.obstacle_cbf_value(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_obs, self.y_obs)
        self._barrier = h_value
        # grad_h -> LfV이어야 되는거 아닌가?
        if np.abs(h_value)<=0.000000000001:
            input_val = [0,0]
            print("(explog cbf) grad_h: ", grad_h)
            print('(explog cbf) grad LfV:', grad1_h)
            print("(explog cbf) h value: ", h_value)
            self.throttle = 0
            self.steer = 0
            print('(explog cbf) throtte: ', self.throttle, ', steer: ', self.steer)
            self.cbf_on = False
        else:
            self.cbf_on = True
            input_val = self.cbf_col_avoid_cont_law(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_obs, self.y_obs)
            print("(explog cbf) grad_h: ", grad_h)
            print('(explog cbf) grad LfV', grad1_h)
            print("(explog cbf) h value: ", h_value)
            self.throttle = input_val[0]
            # self.steer = np.fmax(np.fmin(input_val[1], max_steer), min_steer)
            self.steer = input_val[1]
            print('(explog cbf) throtte: ', self.throttle, ', steer: ', self.steer)



    # log and exponential version
    def construct_cbf_front_veh(self, x_max_n, strength = 100, exponen_stiff = 1.5, type='max'):
                # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0
        self.update_desired_speed()
        width = 4.5
        height = 2.0
        
        self.exponen_stiff = exponen_stiff
        self.cbf_on = False


        if self.cbf_col_avoid_cont_law==None:

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            x_max = sympy.Symbol('x_max')
            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()
            # sympy.exp()

            # Set Obstacle CBF
            V = strength*(sympy.log(1+sympy.exp(self.exponen_stiff*(x-x_max))))


            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/u)+(kr/m)*((v-b*omega)/u))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/u) - (b*kr/Iz)*((v-b*omega)/u) )*e_x6
                                
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

            # Lg1V = LieDerivative(g_acc, V)
            # Lg2V = LieDerivative(g_delta, V)
            # LgV = sympy.Matrix([Lg1V, Lg2V])

            self.obstacle_cbf_value = lambdify([x, y, yaw, u, v, omega, x_max], V, 'numpy')

            LfV = LieDerivative(f, V)
            self.cbf_col_avoid_LfV = lambdify([x, y, yaw, u, v, omega, x_max], LfV, 'numpy')
            self.LfV = LfV

            # second derivative
            LfLfV = LieDerivative(f, LfV)
            LfLg1V = LieDerivative(g_acc, LfV)
            LfLg2V = LieDerivative(g_delta, LfV)
            LfLgV = sympy.Matrix([LfLg1V, LfLg2V])

            LgVTLgV = sympy.transpose(LfLgV)*LfLgV
            bTb = LgVTLgV[0]
            self.cbf_col_avoid_btb = lambdify([x, y, yaw, u, v, omega, x_max], bTb, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            input_u = -((LfLfV + 2*self.k*LfV + (self.k**2)*V)/(bTb))*LfLgV

            # print('input_u: ', input_u)

            end_transform = time.time()
            print('time for transforming to CLF input: ', end_transform - start_transform)

            start_input_eval = time.time()
            
            end_input_eval = time.time()
            print('time for evaluating CLF input: ', end_input_eval - start_input_eval)

            f = lambdify([x, y, yaw, u, v, omega, x_max], input_u, 'numpy') # 'numpy : 137 ms, 

            self.cbf_col_avoid_cont_law = f
        
        start_lambdify_eval = time.time()
        grad_h = self.cbf_col_avoid_btb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, x_max_n)
        grad1_h = self.cbf_col_avoid_LfV(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, x_max_n)
        h_value = self.obstacle_cbf_value(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, x_max_n)
        self._barrier = h_value
        # grad_h -> LfV이어야 되는거 아닌가?
        if np.abs(grad1_h)<=0.00000000001:
            input_val = [0,0]
            print("grad_h: ", grad_h)
            print('grad LfV:', grad1_h)
            print("h value: ", h_value)
            self.cbf_on = False
        else:
            input_val = self.cbf_col_avoid_cont_law(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, x_max_n)
            print('***********input_val:', input_val)
            end_lambdify_eval = time.time()
            print('*********time for lambdify function: ', end_lambdify_eval-start_lambdify_eval)
            # print('*********time for ufuncify function: ', end_ufuncify_eval-start_ufuncify_eval)
            print("grad_h: ", grad_h)
            print('grad LfV', grad1_h)
            print("h value: ", h_value)
            self.throttle = input_val[0]
            # self.steer = np.fmax(np.fmin(input_val[1], max_steer), min_steer)
            self.steer = input_val[1] # 차선책
            self.cbf_on = True

    # log and exponential version
    def construct_cbf_RD2_logexp_road_boundary(self, y_max=100,y_min=-100, strength = 10, exponen_stiff = 3, type='max'):
                # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0
        self.update_desired_speed()
        width = 4.5
        height = 2.0
        threshold = 0
        
        self.exponen_stiff = exponen_stiff
        self.cbf_on = False


        if self.cbf_col_avoid_cont_law_rb==None:

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()
            # sympy.exp()

            # Set Obstacle CBF
            V = strength*(sympy.log(1+sympy.exp(self.exponen_stiff*(y-y_max)))) + strength*(sympy.log(1+sympy.exp(self.exponen_stiff*(y_min-y))))

            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/u)+(kr/m)*((v-b*omega)/u))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/u) - (b*kr/Iz)*((v-b*omega)/u) )*e_x6
                                
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

            # Lg1V = LieDerivative(g_acc, V)
            # Lg2V = LieDerivative(g_delta, V)
            # LgV = sympy.Matrix([Lg1V, Lg2V])

            self.obstacle_cbf_value_rb = lambdify([x, y, yaw, u, v, omega], V, 'numpy')

            LfV = LieDerivative(f, V)
            self.cbf_col_avoid_LfV_rb = lambdify([x, y, yaw, u, v, omega], LfV, 'numpy')
            self.LfV = LfV

            # second derivative
            LfLfV = LieDerivative(f, LfV)
            LfLg1V = LieDerivative(g_acc, LfV)
            LfLg2V = LieDerivative(g_delta, LfV)
            LfLgV = sympy.Matrix([LfLg1V, LfLg2V])

            LgVTLgV = sympy.transpose(LfLgV)*LfLgV
            bTb = LgVTLgV[0]
            self.cbf_col_avoid_btb_rb = lambdify([x, y, yaw, u, v, omega], bTb, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            input_u = -((LfLfV + 2*self.k*LfV + (self.k**2)*V)/(self.slack_variable+bTb))*LfLgV

            print('input_u: ', input_u)

            end_transform = time.time()
            print('time for transforming to CLF input: ', end_transform - start_transform)

            start_input_eval = time.time()
            
            end_input_eval = time.time()
            print('time for evaluating CLF input: ', end_input_eval - start_input_eval)

            f = lambdify([x, y, yaw, u, v, omega], input_u, 'numpy') # 'numpy : 137 ms, 

            self.cbf_col_avoid_cont_law_rb = f
        
        start_lambdify_eval = time.time()
        grad_h = self.cbf_col_avoid_btb_rb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega)
        grad1_h = self.cbf_col_avoid_LfV_rb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega)
        h_value = self.obstacle_cbf_value_rb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega)
        self._barrier = h_value
        # grad_h -> LfV이어야 되는거 아닌가?
        if np.abs(grad1_h)<=0.00000000001:
            input_val = [0,0]
            print("grad_h: ", grad_h)
            print('grad LfV:', grad1_h)
            print("h value: ", h_value)
            self.cbf_on = False
        else:
            input_val = self.cbf_col_avoid_cont_law_rb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega)
            print('***********input_val:', input_val)
            end_lambdify_eval = time.time()
            print('*********time for lambdify function: ', end_lambdify_eval-start_lambdify_eval)
            # print('*********time for ufuncify function: ', end_ufuncify_eval-start_ufuncify_eval)
            print("grad_h: ", grad_h)
            print('grad LfV', grad1_h)
            print("h value: ", h_value)
            self.throttle = input_val[0]
            # self.steer = np.fmax(np.fmin(input_val[1], max_steer), min_steer)
            if (np.abs(input_val[1])>=threshold):
                self.steer = input_val[1] # 차선책
            else:
                self.steer=0
            self.cbf_on = True

    # log and exponential version
    def construct_cbf_RD2_logexp_ymin(self, y_min=100, strength = 100, exponen_stiff = 0.9, type='min'):
                # vehicle width and height
        self.throttle = 0.0
        self.steer = 0.0
        self.update_desired_speed()
        width = 4.5
        height = 2.0
        
        self.exponen_stiff = exponen_stiff
        self.cbf_on = False


        if self.cbf_col_avoid_cont_law==None:

            start_transform = time.time()
            M = Manifold("M",6) # 6 dimensions (x, y, yaw, u, v, omega)
            P = Patch("P",M) #

            coord = CoordSystem("coord",P,sympy.symbols("x y yaw u v omega", real=True))
            x,y,yaw,u,v,omega   = coord.coord_functions()

            x1 = sympy.Matrix([x,y,yaw,u,v,omega])
            e_x1,e_x2,e_x3,e_x4,e_x5,e_x6 = coord.base_vectors()
            # sympy.exp()

            # Set Obstacle CBF
            V = strength*(sympy.log(1+sympy.exp(self.exponen_stiff*(y_min-y))))


            # Construct the state space equation
            f  = (u*sympy.cos(yaw)-v*sympy.sin(yaw))*e_x1 \
                + (u*sympy.sin(yaw)+v*sympy.cos(yaw))*e_x2 \
                    + omega*e_x3 \
                        + (v*omega)*e_x4 \
                            + (-u*omega + kf/m *((v+a*omega)/u)+(kr/m)*((v-b*omega)/u))*e_x5 \
                                + ( (a*kf/Iz)*((v+a*omega)/u) - (b*kr/Iz)*((v-b*omega)/u) )*e_x6
                                
            g_acc = 0*e_x1 + 0*e_x2 + 0*e_x3 + 1*e_x4 + 0*e_x5 + 0*e_x6
            # Edit the error
            g_delta = 0*e_x1 + 0*e_x2 + 0*e_x3 - 1/m*(kf*((v+a*omega)/u))*e_x4 -kf/m*e_x5 -a*kf/Iz*e_x6

            # Lg1V = LieDerivative(g_acc, V)
            # Lg2V = LieDerivative(g_delta, V)
            # LgV = sympy.Matrix([Lg1V, Lg2V])

            self.obstacle_cbf_value = lambdify([x, y, yaw, u, v, omega], V, 'numpy')

            LfV = LieDerivative(f, V)
            self.cbf_col_avoid_LfV = lambdify([x, y, yaw, u, v, omega], LfV, 'numpy')
            self.LfV = LfV

            # second derivative
            LfLfV = LieDerivative(f, LfV)
            LfLg1V = LieDerivative(g_acc, LfV)
            LfLg2V = LieDerivative(g_delta, LfV)
            LfLgV = sympy.Matrix([LfLg1V, LfLg2V])

            LgVTLgV = sympy.transpose(LfLgV)*LfLgV
            bTb = LgVTLgV[0]
            self.cbf_col_avoid_btb = lambdify([x, y, yaw, u, v, omega], bTb, 'numpy')
            # if bTb.subs([(x, self._current_x), (y, self._current_y),(yaw, self._current_yaw),(u, self._current_vx), (v, self._current_vy), (omega, self._current_omega), (x_obs, self.x_obs),(y_obs, self.y_obs)]).evalf(10) <= 0.01:
            #     return [0,0]

            input_u = -((LfLfV + 2*self.k*LfV + (self.k**2)*V)/(bTb))*LfLgV

            print('input_u: ', input_u)

            end_transform = time.time()
            print('time for transforming to CLF input: ', end_transform - start_transform)

            start_input_eval = time.time()
            
            end_input_eval = time.time()
            print('time for evaluating CLF input: ', end_input_eval - start_input_eval)

            f = lambdify([x, y, yaw, u, v, omega], input_u, 'numpy') # 'numpy : 137 ms, 

            self.cbf_col_avoid_cont_law = f
        
        start_lambdify_eval = time.time()
        grad_h = self.cbf_col_avoid_btb(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega)
        grad1_h = self.cbf_col_avoid_LfV(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega)
        h_value = self.obstacle_cbf_value(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega)
        self._barrier = h_value
        # grad_h -> LfV이어야 되는거 아닌가?
        if np.abs(grad1_h)<=0.00000000001:
            input_val = [0,0]
            print("grad_h: ", grad_h)
            print('grad LfV:', grad1_h)
            print("h value: ", h_value)
            self.cbf_on = False
        else:
            input_val = self.cbf_col_avoid_cont_law(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega)
            print('***********input_val:', input_val)
            end_lambdify_eval = time.time()
            print('*********time for lambdify function: ', end_lambdify_eval-start_lambdify_eval)
            # print('*********time for ufuncify function: ', end_ufuncify_eval-start_ufuncify_eval)
            print("grad_h: ", grad_h)
            print('grad LfV', grad1_h)
            print("h value: ", h_value)
            self.throttle = input_val[0]
            # self.steer = np.fmax(np.fmin(input_val[1], max_steer), min_steer)
            self.steer = input_val[1] # 차선책
            self.cbf_on = True

    def get_control_law(self, mode):
        input_val = [0,0]
        cbf_value = 0
        if mode == 'cbf_and_clf':
            self.construct_cbf_RD2_gauss(self.obstacle.x1_obs, self.obstacle.x2_obs)
            input_val[0] += self.throttle
            input_val[1] += np.abs(self.steer)*self.steer_adjust_weight_clf
            self.construct_clf_heur()
            input_val[0] += self.throttle
            input_val[1] += self.steer*self.steer_adjust_weight_clf
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif mode == 'clf_and_xmax':
            # self.construct_cbf_RD2_logexp(100, strength = 100, exponen_stiff=1.2)
            # input_val[0] += self.throttle
            # input_val[1] += np.abs(self.steer)
            self.construct_clf_heur()
            input_val[0] += self.throttle
            input_val[1] += self.steer*self.steer_adjust_weight_clf
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif mode == 'clf_heur':
            self.construct_clf_heur()
            input_val[0] += self.throttle
            input_val[1] += self.steer
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif mode == 'sep_clf':
            self.construct_clf_sep()
            input_val[0] += self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[0]
            input_val[1] += self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[1]
            # print('scaling factor: ', self.x_y_scaling_factor)
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif mode == 'front_veh':
            self.construct_cbf_front_veh(x_max_n = 100)
            input_val[0] += self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[0]
            input_val[1] += self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[1]
            # print('scaling factor: ', self.x_y_scaling_factor)
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif mode == 'sep_clf_and_cbf':
            self.construct_cbf_RD2_gauss(self.obstacle.x1_obs, self.obstacle.x2_obs)
            input_val[0] += self.throttle*0.02
            input_val[1] += np.abs(self.steer)*0.02
            # input_val[1] += self.steer*0.02
            self.construct_clf_sep()
            input_val[0] += self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[0]
            input_val[1] += self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[1]
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif mode == 'sep_clf_and_cbf_obsexp':
            self.construct_cbf_RD2_logexp_collision(strength = 1, exponen_stiff=0.01)
            input_val[0] += self.throttle*0.01
            input_val[1] += np.abs(self.steer)*0.01 # trick
            # input_val[1] += self.steer*0.02 # trick
            # input_val[1] += np.abs(self.steer)*0.015 # original
            self.construct_clf_sep()
            input_val[0] += self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[0]
            input_val[1] += self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[1]
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif (mode == 'sep_clf_mix'): # unstable
            self.construct_clf_sep_mix()
            input_val[0] += self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[0]
            input_val[1] += self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[1]
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif (mode == 'rd1_umax_const'):
            self.construct_cbf_RD1_logexp(vmax=10)
            input_val[0] += 1.0 # default acceleration
            input_val[0] += self.throttle
            input_val[1] += np.abs(self.steer)
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif mode == 'sep_clf_RD2':
            self.construct_clf_sep()
            input_val[0] += self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[0]
            input_val[1] += self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[1]
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif mode == 'sep_clf_RD1':
            self.construct_clf_sep()
            input_val[0] += self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[0]
            input_val[1] += self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[1]
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif mode == 'sep_clf_and_cbf_RD1_collision':
            self.construct_cbf_RD1_logexp_collision()
            input_val[0] += self.throttle*0.02
            input_val[1] += np.abs(self.steer)*0.02
            # input_val[1] += self.steer*0.02
            self.construct_clf_sep()
            input_val[0] += self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[0]
            input_val[1] += self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[1]
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif mode == 'sep_clf_and_multiple_cbf':
            weight_cbf = 0.035
            weight_clf = 1.0
            max_risk = 4000.0
            threshold = 0.01
            if (self.obstacle != None):
                self.construct_cbf_RD2_gauss(self.obstacle.state[0], self.obstacle.state[1])
                input_val[0] += self.throttle*weight_cbf
                if (self.b_positive_obs1 == None):
                    if (self.steer >= threshold):
                        self.b_positive_obs1 = True
                    elif (self.steer <= -threshold):
                        self.b_positive_obs1 = False

                if (self.b_positive_obs1 == None):
                    input_val[1] += self.steer*weight_cbf
                elif (self.b_positive_obs1 == True):
                    input_val[1] += np.abs(self.steer)*weight_cbf
                elif (self.b_positive_obs1 == False):
                    input_val[1] += -np.abs(self.steer)*weight_cbf
                input_val[0] += self.throttle*weight_cbf
                cbf_value += self._barrier
            if (self.obstacle2 != None):
                self.construct_cbf_RD2_gauss(self.obstacle2.state[0], self.obstacle2.state[1])
                if (self.b_positive_obs2 == None):
                    if (self.steer >= threshold):
                        self.b_positive_obs2 = True
                    elif (self.steer <= -threshold):
                        self.b_positive_obs2 = False
                if (self.b_positive_obs2 == None):
                    input_val[1] += self.steer*weight_cbf
                elif (self.b_positive_obs2 == True):
                    input_val[1] += np.abs(self.steer)*weight_cbf
                elif (self.b_positive_obs2 == False):
                    input_val[1] += -np.abs(self.steer)*weight_cbf
                # input_val[0] += self.throttle*weight_cbf
                # input_val[1] += self.steer*weight_cbf
                input_val[0] += self.throttle*weight_cbf
                cbf_value += self._barrier
            if (self.obstacle3 != None):
                self.construct_cbf_RD2_gauss(self.obstacle3.state[0], self.obstacle3.state[1])
                if (self.b_positive_obs3 == None):
                    if (self.steer >= threshold):
                        self.b_positive_obs3 = True
                    elif (self.steer <= -threshold):
                        self.b_positive_obs3 = False
                if (self.b_positive_obs3 == None):
                    input_val[1] += self.steer*weight_cbf
                elif (self.b_positive_obs3 == True):
                    input_val[1] += np.abs(self.steer)*weight_cbf
                elif (self.b_positive_obs3 == False):
                    input_val[1] += -np.abs(self.steer)*weight_cbf
                # input_val[0] += self.throttle*weight_cbf
                # input_val[1] += self.steer*weight_cbf
                input_val[0] += self.throttle*weight_cbf
                cbf_value += self._barrier
            if (self.obstacle4 != None):
                self.construct_cbf_RD2_gauss(self.obstacle4.state[0], self.obstacle4.state[1])
                if (self.b_positive_obs4 == None):
                    if (self.steer >= threshold):
                        self.b_positive_obs4 = True
                    elif (self.steer <= -threshold):
                        self.b_positive_obs4 = False
                if (self.b_positive_obs4 == None):
                    input_val[1] += self.steer*weight_cbf
                elif (self.b_positive_obs4 == True):
                    input_val[1] += np.abs(self.steer)*weight_cbf
                elif (self.b_positive_obs4 == False):
                    input_val[1] += -np.abs(self.steer)*weight_cbf
                input_val[0] += self.throttle*weight_cbf
                cbf_value += self._barrier
            
            if (cbf_value <= max_risk):
                self.construct_clf_sep()
                input_val[0] += weight_clf*self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[0]
                input_val[1] += weight_clf*self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[1]
            else:
                self.construct_clf_sep()
            self._barrier = cbf_value
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif (mode == 'road_boundary_test'):
            self.construct_cbf_RD2_logexp_road_boundary(y_max=3, y_min=-3)
            # input_val[1] += 0.1 # default acceleration
            input_val[0] += self.throttle
            input_val[1] += self.steer
            # self.throttle = np.clip(input_val[0], min_acc, max_acc)
            # self.steer = np.clip(input_val[1], min_steer, max_steer)
            # self.construct_cbf_RD2_logexp_ymin(-3)
            # input_val[0] += self.throttle
            # input_val[1] += self.steer
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif (mode == 'multiple_gauss'): # test
            # max_risk = 3.0
            weight_clf = 1.0
            weight_cbf = 0.03
            max_risk = 5.0
            threshold = 0.01
            input_val = [0,0]
            self.construct_cbf_RD2_gauss_multiple(self.obstacle.state[0], self.obstacle.state[1],self.obstacle2.state[0], self.obstacle2.state[1],self.obstacle3.state[0], self.obstacle3.state[1],self.obstacle4.state[0], self.obstacle4.state[1])
            input_val[0] += weight_cbf*self.throttle
            input_val[1] += weight_cbf*self.steer
            if (self._barrier <= max_risk):
                self.construct_clf_sep()
                input_val[0] += self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[0]
                input_val[1] += self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[1]
            else:
                self.construct_clf_sep()
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
        elif mode == 'sep_clf_and_multiple_cbf_road_boundary':
            weight_cbf = 0.015
            weight_clf = 1.0
            max_risk = 5.0
            threshold = 100
            if (self.obstacle != None):
                self.construct_cbf_RD2_gauss(self.obstacle.state[0], self.obstacle.state[1])
                input_val[0] += self.throttle*weight_cbf
                if (self.b_positive_obs1 == None):
                    if (self.steer >= threshold):
                        self.b_positive_obs1 = True
                    elif (self.steer <= -threshold):
                        self.b_positive_obs1 = False

                if (self.b_positive_obs1 == None):
                    input_val[1] += self.steer*weight_cbf
                elif (self.b_positive_obs1 == True):
                    input_val[1] += np.abs(self.steer)*weight_cbf
                elif (self.b_positive_obs1 == False):
                    input_val[1] += -np.abs(self.steer)*weight_cbf
                input_val[0] += self.throttle*weight_cbf
                cbf_value += self._barrier
            if (self.obstacle2 != None):
                self.construct_cbf_RD2_gauss(self.obstacle2.state[0], self.obstacle2.state[1])
                if (self.b_positive_obs2 == None):
                    if (self.steer >= threshold):
                        self.b_positive_obs2 = True
                    elif (self.steer <= -threshold):
                        self.b_positive_obs2 = False
                if (self.b_positive_obs2 == None):
                    input_val[1] += self.steer*weight_cbf
                elif (self.b_positive_obs2 == True):
                    input_val[1] += np.abs(self.steer)*weight_cbf
                elif (self.b_positive_obs2 == False):
                    input_val[1] += -np.abs(self.steer)*weight_cbf
                # input_val[0] += self.throttle*weight_cbf
                # input_val[1] += self.steer*weight_cbf
                input_val[0] += self.throttle*weight_cbf
                cbf_value += self._barrier
            if (self.obstacle3 != None):
                self.construct_cbf_RD2_gauss(self.obstacle3.state[0], self.obstacle3.state[1])
                if (self.b_positive_obs3 == None):
                    if (self.steer >= threshold):
                        self.b_positive_obs3 = True
                    elif (self.steer <= -threshold):
                        self.b_positive_obs3 = False
                if (self.b_positive_obs3 == None):
                    input_val[1] += self.steer*weight_cbf
                elif (self.b_positive_obs3 == True):
                    input_val[1] += np.abs(self.steer)*weight_cbf
                elif (self.b_positive_obs3 == False):
                    input_val[1] += -np.abs(self.steer)*weight_cbf
                # input_val[0] += self.throttle*weight_cbf
                # input_val[1] += self.steer*weight_cbf
                input_val[0] += self.throttle*weight_cbf
                cbf_value += self._barrier
            if (self.obstacle4 != None):
                self.construct_cbf_RD2_gauss(self.obstacle4.state[0], self.obstacle4.state[1])
                if (self.b_positive_obs4 == None):
                    if (self.steer >= threshold):
                        self.b_positive_obs4 = True
                    elif (self.steer <= -threshold):
                        self.b_positive_obs4 = False
                if (self.b_positive_obs4 == None):
                    input_val[1] += self.steer*weight_cbf
                elif (self.b_positive_obs4 == True):
                    input_val[1] += np.abs(self.steer)*weight_cbf
                elif (self.b_positive_obs4 == False):
                    input_val[1] += -np.abs(self.steer)*weight_cbf
                input_val[0] += self.throttle*weight_cbf
                cbf_value += self._barrier
            
            if (cbf_value <= max_risk):
                self.construct_clf_sep()
                input_val[0] += weight_clf*self.input_u1(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[0]
                input_val[1] += weight_clf*self.input_u2(self._current_x, self._current_y, self._current_yaw, self._current_vx, self._current_vy, self._current_omega, self.x_ref, self.y_ref, self.yaw_ref, self.u_ref, self.v_ref, self.omega_ref)[1]
            else:
                self.construct_clf_sep()
            self.construct_cbf_RD2_logexp_road_boundary(y_max=5, y_min=-5, exponen_stiff=8)
            # input_val[1] += 0.1 # default acceleration
            input_val[0] += self.throttle
            input_val[1] += self.steer
            # cbf_value += self._barrier
            # self.throttle = np.clip(input_val[0], min_acc, max_acc)
            # self.steer = np.clip(input_val[1], min_steer, max_steer)
            # self.construct_cbf_RD2_logexp_ymin(-3)
            # input_val[0] += self.throttle
            # input_val[1] += self.steer
            self._barrier = cbf_value
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)
            self.throttle = np.clip(input_val[0], min_acc, max_acc)
            self.steer = np.clip(input_val[1], min_steer, max_steer)


    # PID controller (compare)
    def get_pid_control_law(self):
        # update status
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        v = self._current_speed
        self.update_desired_speed()
        v_desired = self._desired_speed
        waypoints = self._waypoints

        # ==================================
        # LONGITUDINAL CONTROLLER, using PID speed controller
        # ==================================
        self._e = v_desired - v  # v_desired
        self.e_buffer.append(self._e)

        if len(self.e_buffer) >= 2:
            _de = (self.e_buffer[-1] - self.e_buffer[-2]) / self.dt
            _ie = sum(self.e_buffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        self.throttle = np.clip((self.K_P * self._e) + (self.K_D * _de / self.dt) + (self.K_I * _ie * self.dt), -1.0, 1.0)

        # ==================================
        # LATERAL CONTROLLER, using stanley steering controller for lateral control.
        # ==================================
        k_e = 0.3
        k_v = 20

        # 1. calculate heading error
        yaw_path = np.arctan2(waypoints[-1][1] - waypoints[0][1], waypoints[-1][0] - waypoints[0][0])
        # self.yaw_path = yaw_path
        yaw_diff = yaw_path - yaw
        if yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        if yaw_diff < - np.pi:
            yaw_diff += 2 * np.pi

        # 2. calculate crosstrack error
        current_xy = np.array([x, y])
        crosstrack_error = np.min(np.sum((current_xy - np.array(waypoints)[:, :2]) ** 2, axis=1))

        yaw_cross_track = np.arctan2(y - waypoints[0][1], x - waypoints[0][0])
        yaw_path2ct = yaw_path - yaw_cross_track
        if yaw_path2ct > np.pi:
            yaw_path2ct -= 2 * np.pi
        if yaw_path2ct < - np.pi:
            yaw_path2ct += 2 * np.pi
        if yaw_path2ct > 0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = - abs(crosstrack_error)

        yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + v))

        # print(crosstrack_error, yaw_diff, yaw_diff_crosstrack)

        # TODO ****** For Lyapunov controller,
        # Firstly use yaw_diff, second use steer_expect. 
        # 3. control low
        steer_expect = yaw_diff + yaw_diff_crosstrack
        if steer_expect > np.pi:
            steer_expect -= 2 * np.pi
        if steer_expect < - np.pi:
            steer_expect += 2 * np.pi
        steer_expect = min(1.22, steer_expect)
        steer_expect = max(-1.22, steer_expect)

        # 4. update
        steer_output = steer_expect

        # Convert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * steer_output
        # Clamp the steering command to valid bounds
        self.steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        # Calculate the control input for lyapunov function