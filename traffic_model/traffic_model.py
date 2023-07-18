

import numpy as np
import scipy.linalg as la
import random
import pretty_errors
import copy
from matplotlib import pyplot as plt
import os,sys
import math
import pdb
import time
from typing import Dict, List
from dataclasses import dataclass, field
from traffic_model.podar import PODAR
from traffic_model.utils.global_value import global_value
from traffic_model.utils.ccbf import ccbf_controller, CCBFOption, VehicleSpec

@dataclass
class info_update:
    veh_info:Dict[str, str] = field(default_factory=dict)
    traffic_info:Dict[str, str] = field(default_factory=dict)
    static_paths:List[str] = field(default_factory=list)

    def update(self,Surr):
        self.veh_info=Surr.veh_info
        self.traffic_info=Surr.traffic_info
        self.static_paths=Surr.static_paths
@dataclass
class driver_data():
    nou:float=1
    y_bias:float=0
    v_bias:float=0

    def sampler_bias(self):
        """

        :return:lateral bias and longitudinal velocity bias for driver
        """
        y_list=np.array([-1,-0.5,0,0.5,1])
        v_list=np.array([-1,-0.5,0,0.5,1])
        d_ybias=random.choice(y_list)
        d_vbias=random.choice(v_list)

        # print(d_ybias,d_vbias)
        return d_ybias,d_vbias

@dataclass()
class model_params():
    coef_v:float=1.5
    coef_phi:float=1
    target_v:float=10
    g_attention:float=30
    turn_dis_base:float=5

class Surr(object):
    def __init__(self):


        self.nou=driver_data.nou                    #驾驶员激进系数
        # self.driver_style=driver_info[1]            #驾驶员驾驶风格
        self.y_bias=driver_data.y_bias                  #驾驶员横向偏置
        self.v_bias=driver_data.v_bias               #驾驶员速度偏置
        self.ccbf_controller = ccbf_controller.CCBF_Controller(VehicleSpec())
        (P1, P2, P3, P4, P5, P6) = CCBFOption.weight
        self.ccbf_controller.update_lyapunov_parameter(P1, P2, P3, P4, P5, P6)
        global_value._init()
        self.vehs_relation= {}
        self.veh_info= {}
        self.traffic_info= {}
        # self.x = round(self.veh_info.moving_info.position.point.x, 2)
        # self.y = round(self.veh_info.moving_info.position.point.y, 2)
        self.x=0
        self.y=0
        self.reset()




        print(f"nou:{self.nou},y_bias:{self.y_bias},v_bias:{self.v_bias}")



    def update(self, input_params,veh_id_around) :
        start_time=time.time()

        veh_info=input_params['veh_info']
        self.veh_info=veh_info
        traffic_info=input_params['traffic_info']
        self.traffic_info=traffic_info
        self.ctrl_veh=veh_info.id

        x=round(veh_info.moving_info.position.point.x,2)
        self.x=x
        y=round(veh_info.moving_info.position.point.y,2)
        self.y=y
        phi=round(veh_info.moving_info.position.phi,3)
        u = round(veh_info.moving_info.u, 2)

        tag=global_value.tag_getValue()
        print(f"tag:{tag}")
        self.ov_data = copy.deepcopy(traffic_info)


        #2控制其中一辆车，得到其与周围车辆关系

        IDC = IDC_decision(input_params)
        if len(self.ov_data)!=0:

            vehs_relation = IDC.get_vehs_relation( veh_id_around, input_params)
            print(vehs_relation)
        else:
            vehs_relation={}
            print('no necerity to get vehicles relation')
        self.vehs_relation=vehs_relation
        #3采样行为，标准驾驶行为、异常驾驶行为
        d_behavior=IDC.sampler_behavior()



        #IDC决策与静态路径优选

        static_path_des=IDC.IDC_decide(vehs_relation,d_behavior)

        ref_path=static_path_des[0]
        target_idx=static_path_des[1]

        print(f"target_idx:{target_idx}")


        # 6静态路径偏置，输出


        v_A = IDC.get_track_speed(vehs_relation)
        is_safe_LC=IDC.judge_LC_conditon(vehs_relation)

        new_ref_path=IDC.output_satic_path(ref_path,v_A,target_idx,is_safe_LC,vehs_relation)
        print(f"new_ref_path_x:{new_ref_path[0][0]},new_ref_path_y:{new_ref_path[1][0]}")
        lane_rad = IDC.lane_rad

        new_phi = phi - lane_rad

        print(f"new_phi:{new_phi}")
        # if new_phi<0:
        #     new_phi=new_phi+2*math.pi

        transfered_path=self.transfer_path(new_ref_path, IDC.lane_rad)
        print(f"new_x:{transfered_path[0][0]},new_y:{transfered_path[1][0]}")

        print("跟驰行为车速：",v_A)

        # 7跟踪静态路径
        v=0
        w=0

        transfered_x, transfered_y = self.world_to_self_car(new_ref_path[0][0], new_ref_path[1][0], self.x, self.y,  IDC.lane_rad)
        # ccbf controller
        self.ccbf_controller.update_values(0, 0, new_phi, u, v, w)
        self.ccbf_controller.update_waypoints(transfered_path)
        # self.ccbf_controller.get_control_law('sep_clf')

        self.ccbf_controller.construct_clf_sep()
        # self.ccbf_controller.g
        accel = self.ccbf_controller.throttle
        steer = self.ccbf_controller.steer
        CCBFOption.control_bound
        accel = np.clip(accel, -3, 1) #-2,2
        steer = np.clip(steer, -0.4, 0.4)#-1.22, 1.22

        curr_veh_front_list=vehs_relation['veh_cl']

        print(f"x:{x},y:{y}")
        # print(f"transfered_x:{transfered_x},transfered_y:{transfered_y}")

        if len(curr_veh_front_list)!=0:
            # # Car-following rule
            leader_id = list(curr_veh_front_list[0].keys())[0]
            leader_x = traffic_info[leader_id][0].position.point.x
            leader_y = traffic_info[leader_id][0].position.point.y
            front_veh_x, front_veh_y = (leader_x, leader_y)
        else:
            if IDC.rad_tag ==1:
                front_veh_x, front_veh_y = (self.x+100, self.y)
            if IDC.rad_tag ==2:
                front_veh_x, front_veh_y = (self.x , self.y+100)
            if IDC.rad_tag == 3:
                front_veh_x, front_veh_y = (self.x-100, self.y )
            if IDC.rad_tag ==4:
                front_veh_x, front_veh_y = (self.x , self.y-100)

        front_veh_x, front_veh_y=self.world_to_self_car(front_veh_x, front_veh_y, self.x, self.y, IDC.lane_rad)

        #todo 是否用转换坐标
        self.ccbf_controller.construct_cbf_front_veh(x_max_n=front_veh_x, strength=5, exponen_stiff=3)    #default:strength=5,exponen_stiff=2.5,3

        # if (self.ccbf_controller.throttle <= -0.001):
        #     accel += self.ccbf_controller.throttle
        #     # steer = 0
        #     print('(CCBF) car following input, acc: ', self.ccbf_controller.throttle[0], ', steer: ',
        #           self.ccbf_controller.steer[0])

        accel = np.clip(accel, CCBFOption.control_bound[0][1], CCBFOption.control_bound[1][1])
        steer = np.clip(steer, CCBFOption.control_bound[0][0], CCBFOption.control_bound[1][0])


        print('(CCBF) car following input, acc: ', accel, ', steer: ', steer)

        result=[accel,steer]

        end_time_update=time.time()
        print(f'单个step中update花费时间:{round((end_time_update - start_time), 2)}s')
        return result




    def transfer_path(self,new_ref_path,lane_rad):
        tansfered_path=[]
        x_list=new_ref_path[0]
        y_list=new_ref_path[1]
        x_list_new=[]
        y_list_new = []

        # for x,y in zip(x_list,y_list):
        #     x_transered,y_transfered=self.world_to_self_car(x,y,0,0,lane_rad)
        #     x_list_new.append(x_transered)
        #     y_list_new.append(y_transfered)
        x_list_new,y_list_new=[self.world_to_self_car(x,y,self.x,self.y,lane_rad)[0] for x,y in zip(x_list,y_list)],\
            [self.world_to_self_car(x,y,self.x,self.y,lane_rad)[1] for x,y in zip(x_list,y_list)]
        tansfered_path.append(x_list_new)
        tansfered_path.append(y_list_new)
        tansfered_path.append(new_ref_path[2])
        tansfered_path.append(new_ref_path[3])

        return tansfered_path

    def driver_info(self):
        '''

        :return: driver info,including driving styles,bias for speed and lateral position
        '''
        self.nou=self.sampler_nou()
        if 0.1<=self.nou<=0.4:
            self.driver_style=0
        if 0.4<self.nou<=0.7:
            self.driver_style=1
        if 0.7<self.nou<=1:
            self.driver_style=2
        # print(self.driver_style)
        self.y_bias,self.v_bias=self.sampler_bias()
        return self.nou,self.driver_style,self.y_bias,self.v_bias

    def sampler_nou(self):
        """

        :return:
        """
        nou_list=np.arange(10)+np.ones(10)
        nou_list=nou_list*0.1
        # print(nou_list)
        d_nou=random.choices(nou_list,weights=[5,5,5,5,20,20,20,8,6,6])[0]
        d_nou=round(d_nou,1)
        # print(d_nou)
        return d_nou

    def sampler_bias(self):
        """

        :return:lateral bias and longitudinal velocity bias for driver
        """
        y_list=np.array([-1,-0.5,0,0.5,1])
        v_list=np.array([-1,-0.5,0,0.5,1])
        d_ybias=random.choice(y_list)
        d_vbias=random.choice(v_list)

        # print(d_ybias,d_vbias)
        return d_ybias,d_vbias

    def sampler_behavior(self):
        """

        :return:
        """
        behavior_list=['std_driving','overtake_right','brake_suddenly','cut_in','lane_changes','flane_slow','turtle_speed','flane_parking','slane_high','reversing','serpentine_driving','off_center']
        behavior_numlist=np.arange(12)+np.ones(12)
        # print(nou_list)
        d_behavior=random.choices(behavior_numlist,weights=[89,1,1,1,1,1,1,1,1,1,1,1])[0]
        d_behavior=int(d_behavior)
        d_behavior=1
        print(d_behavior,behavior_list[d_behavior-1])
        return d_behavior

    def world_to_self_car(self,world_x, world_y, car_x, car_y, car_heading):
        # Calculate the relative position of the world coordinates with respect to the car
        relative_x = world_x - car_x
        relative_y = world_y - car_y

        # Rotate the relative position based on the car's heading
        rotated_x = relative_x * math.cos(car_heading) + relative_y * math.sin(car_heading)
        rotated_y = -relative_x * math.sin(car_heading) + relative_y * math.cos(car_heading)    #-relative_x修改为relative_x

        return rotated_x, rotated_y


    #todo
    def select_veh(self,x,y):
        """

        :return: select vehicles around ego cars in range of 100 meters
        """
        tmp_list = []  # 当前自车的周车列表
        sur_vehs = {}  # 每个自车在计算范围内的周车name列表，包含其他自车

        # numbers = [1, 2, 3, 4, 5]
        # dis = la.norm(np.array([(x - value[0].position.point.x), (y - value[0].position.point.y)]))
        # even_numbers = [x for x in numbers if dis <= 100]
        #


        for key,value in self.ov_data.items():  # (Procedure 2)
            if key == self.ctrl_veh: continue  # 如果是自车，跳过

            dis=la.norm(np.array([(x - value[0].position.point.x),(y - value[0].position.point.y)]))
            # print('dis2',dis2)

            # dis3=abs(x - ov_info.x) + abs(y - ov_info.y)          #曼哈顿距离
            # print('dis3',dis3)
            # print("="*100)

            if dis<100:
                tmp_list.append(key)
            sur_vehs[self.ctrl_veh] = tmp_list
        return sur_vehs

    def get_leader_list(self):
        leader_list=[]

        for i in range(3):
            str_name = 'veh_' + str(i + 1) + 'l'
            if bool(self.vehs_relation[str_name]):
                veh = list(self.vehs_relation[str_name][0].keys())[0]
                leader_list.append(veh)
        return leader_list

    def get_follower_list(self):
        follower_list=[]
        for i in range(3):
            str_name='veh_'+str(i+1)+'f'
            if bool(self.vehs_relation[str_name]):
                veh=list(self.vehs_relation[str_name][0].keys())[0]
                follower_list.append(veh)
        return follower_list

    def get_veh_acc(self,veh_list,target_v):
        veh_info_list=[]
        traffic_info=self.traffic_info
        # target_v=6
        leader_u_list=[traffic_info[x][0].u for x in veh_list]

        acc_list=[math.copysign(1,target_v-x)*min(abs(target_v-x),2) for x in leader_u_list]
        veh_info_list.append(veh_list)
        veh_info_list.append(acc_list)
        veh_info_list = list(zip(*veh_info_list))
        return veh_info_list

    def reset(self):
        """clear the information in the last simulation
        """
        # self.ego_name_list = self.init.ego_veh_name

        # print(self.ego_name_list)





class IDC_decision():

    def __init__(self,input_params):
        self.veh_info=input_params['veh_info']
        self.traffic_info=input_params['traffic_info']
        self.static_paths=input_params['static_paths']
        self.u = round(self.veh_info.moving_info.u, 2)
        self.x = round(self.veh_info.moving_info.position.point.x, 2)
        self.y = round(self.veh_info.moving_info.position.point.y, 2)
        self.length=self.veh_info.base_info.Length

        self.vehs_relation=[]
        self.acc_lon_list=[]
        self.ctrl_veh=self.veh_info.id
        self.left_1st=1
        self.right_1st=3
        self.middle_lane=2
        self.target_v=15

        self.lane_rad=self.cul_lane_rad()
        self.rad_tag=self.output_lane_rad_tag()
        self.static_paths_sorted=self.static_paths_sort()
        self.y_lane_list=[path[1][0] for path in self.static_paths_sorted]
        self.x_lane_list = [path[0][0] for path in self.static_paths_sorted]

        if self.rad_tag in [1,3] :
            # self.y_lane_list = self.get_y_lane_list()
            print('y_lane_list:{},y:{}'.format(self.y_lane_list,self.y))
            apply_list=[self.y_lane_list,self.y]
        else:
            # x_lane_list=[path.point.x for path in self.static_paths_sorted]
            print('x_lane_list:{},x:{}'.format(self.x_lane_list, self.x))
            apply_list = [self.x_lane_list, self.x]

        current_laneID = self.veh_info.moving_info.position.lane_id
        if current_laneID=='':
            self.current_idx=1
        else:
            self.current_idx = int(current_laneID.split('_')[2][2])
            #todo
            get_current_idx=min(range(len(self.static_paths)), key=lambda i: abs(apply_list[0][i] - apply_list[1]))+1
            if get_current_idx!=self.current_idx: #避免给了错误的current_idx
                self.current_idx=get_current_idx



        print('current_idx:',self.current_idx)




    #todo
    def get_vehs_relation(self,veh_id_around,input_params):
        """
        input vehicle id,location,get vehicles' ids around the vehicle

        step1
        判断周围车辆所在车道
        自车所在车道
        计算和自车距离
        step2
        根据车道号和x坐标判断位置关系
        :return: {str:list}
        """
        vehs_relation= {}
        traffic_info=input_params['traffic_info']

        veh_ll = []  # 左侧前车,车辆id,与车辆距离
        veh_lf = []  # 左侧后车
        veh_rl = []  # 右侧前车
        veh_rf = []  # 右侧后车
        veh_cl = []  # 本车道前车
        veh_cf = []  #本车道后车

        veh_1l = []  #1#车道前车
        veh_1f = []  #1#车道前车
        veh_2l = []  # 2#车道前车
        veh_2f = []  # 2#车道前车
        veh_3l = []  # 0#车道前车
        veh_3f = []  # 0#车道后车

        veh_info={}
        dis_to_link_end_ctrlveh=round(self.veh_info.moving_info.position.dis_to_link_end,2)
        veh_info[self.ctrl_veh]=[dis_to_link_end_ctrlveh,self.current_idx]
        print(veh_info)
        # veh_simp_info={}
        veh_simp={}
        print("ctrl_veh:",self.ctrl_veh)

        for value in veh_id_around:

            veh_id=value
            position=traffic_info[veh_id][0].position.dis_to_link_end
            # print('junction_id:',traffic_info[value][0].position.junction_id)
            lane_id=traffic_info[veh_id][0].position.lane_id
            # print('lane_id:',lane_id)
            if traffic_info[veh_id][0].position.junction_id == '' and lane_id!='':
                land_id=int(traffic_info[veh_id][0].position.lane_id.split('_')[2][2])
            else:
                land_id=''

            d_v2o=dis_to_link_end_ctrlveh-position
            veh_simp={veh_id:[position,land_id,d_v2o]}

            if traffic_info[value][0].position.junction_id =='' and lane_id!='':
                if position<dis_to_link_end_ctrlveh and land_id==self.current_idx:       #同车道前车
                    # print(f"{veh_id} is leader on this lane")
                    if veh_cl is not None:
                        veh_cl.append(veh_simp)
                if position < dis_to_link_end_ctrlveh and land_id == self.current_idx-1:    #在自车左侧前方
                    # print(f"{veh_id} is leader on left lane")
                    if veh_ll is not None:
                        veh_ll.append(veh_simp)
                if position < dis_to_link_end_ctrlveh and land_id == self.current_idx+1:    #在自车右侧前方
                    # print(f"{veh_id} is leader on right lane")
                    veh_rl.append(veh_simp)
                if position>dis_to_link_end_ctrlveh and land_id==self.current_idx:       #同车道后方
                    # print(f"{veh_id} is follower on this lane")
                    veh_cf.append(veh_simp)
                if position>dis_to_link_end_ctrlveh and land_id==self.current_idx-1:        #在自车左侧后方
                    # print(f"{veh_id} is follower on left lane")
                    veh_lf.append(veh_simp)
                if position > dis_to_link_end_ctrlveh and land_id == self.current_idx+1:    #在自车右侧后方
                    # print(f"{veh_id} is follower on right lane")
                    veh_rf.append(veh_simp)

                if land_id == 1:                                                    # 在1#车道
                    if position < dis_to_link_end_ctrlveh:
                        # print(f"{veh_id} is leader on 1# lane")
                        veh_1l.append(veh_simp)

                    else:
                        # print(f"{veh_id} is follower on 1# lane")
                        veh_1f.append(veh_simp)

                if land_id == 2:                                                    # 在2#车道
                    if position < dis_to_link_end_ctrlveh:
                        # print(f"{veh_id} is leader on 1# lane")
                        veh_2l.append(veh_simp)

                    else:
                        # print(f"{veh_id} is follower on 1# lane")
                        veh_2f.append(veh_simp)

                if land_id ==3:                                                     # 在3#车道
                    if position < dis_to_link_end_ctrlveh:
                        # print(f"{veh_id} is leader on 0# lane")
                        veh_3l.append(veh_simp)

                    else:
                        # print(f"{veh_id} is follower on 0# lane")
                        veh_3f.append(veh_simp)


        if len(veh_cl)>=1:
            veh_cl=self.order_vehicles(veh_cl)
        if len(veh_ll)>=1:
            veh_ll=self.order_vehicles(veh_ll)
        if len(veh_rl) >= 1:
            veh_rl = self.order_vehicles(veh_rl)
        if len(veh_cf)>=1:
            veh_cf=self.order_vehicles(veh_cf)
        if len(veh_lf)>=1:
            veh_lf=self.order_vehicles(veh_lf)
        if len(veh_rf)>=1:
            veh_rf=self.order_vehicles(veh_rf)

        if len(veh_1l)>=1:
            veh_1l=self.order_vehicles(veh_1l)
        if len(veh_1f)>=1:
            veh_1f=self.order_vehicles(veh_1f)
        if len(veh_2l) >= 1:
            veh_2l = self.order_vehicles(veh_2l)
        if len(veh_2f) >= 1:
            veh_2f = self.order_vehicles(veh_2f)
        if len(veh_3l)>=1:
            veh_3l=self.order_vehicles(veh_3l)
        if len(veh_3f)>=1:
            veh_3f=self.order_vehicles(veh_3f)

        vehs_relation['veh_cl'] = veh_cl
        vehs_relation['veh_ll'] = veh_ll
        vehs_relation['veh_rl'] = veh_rl
        vehs_relation['veh_cf'] = veh_cf
        vehs_relation['veh_lf'] = veh_lf
        vehs_relation['veh_rf'] = veh_rf

        vehs_relation['veh_1l'] = veh_1l
        vehs_relation['veh_1f'] = veh_1f
        vehs_relation['veh_2l'] = veh_2l
        vehs_relation['veh_2f'] = veh_2f
        vehs_relation['veh_3l'] = veh_3l
        vehs_relation['veh_3f'] = veh_3f


        # print(vehs_relation)
        print(vehs_relation['veh_cl'])
        self.vehs_relation=vehs_relation
        return vehs_relation

    def order_vehicles(self,vehicles):
        """

        :return:
        """

        dis_list=[]
        if isinstance(vehicles,list):
            dis_list=[list(x.values())[0][2] for x in vehicles]
            index = min(range(len(dis_list)), key=lambda i: abs(dis_list[i]))
        vehicle=[vehicles[index]]
        # print(f'vehicle{vehicle}')

        return vehicle

    def eval_J_common(self,vehs_relation):
        '''

        :return: evaluation for common case
        '''

        c1=1
        c2=1
        c3=0
        c4=0
        c5=0
        c6=0
        lane_num=len(self.static_paths)


        J_traffic=self.eval_traffic(vehs_relation,lane_num)
        J_safety=self.eval_safety(vehs_relation,lane_num)
        J_energy=self.eval_energy(vehs_relation,lane_num)
        J_comfort=self.eval_comfort(vehs_relation,lane_num)
        J_regularity=self.eval_regularity(vehs_relation,lane_num)
        # J_strategy=self.eval_strategy(vehs_relation,lane_num,target_lane_idx=3)




        J_common = [c1 *J_traffic[i] + c2 *J_safety[i] +c3*J_energy[i]+c4*J_comfort[i]+c5 * J_regularity[i]
                    for i in range(lane_num)]
                    # +c5 * J_regularity[i] + c6 * J_strategy[i]
                    # for i in range(min(len(J_traffic), len(J_safety)))]

        print(f"J_traffic:{J_traffic},J_safety:{J_safety},"
              f"J_energy:{J_energy},J_comfort:{J_comfort},"
              f"J_regularity:{J_regularity}")
        return J_common

    def eval_J_special(self,vehs_relation,d_behavior):
        '''

        :return: evaluation for common case
        '''

        coefficient=np.ones(6)
        lane_num = len(self.static_paths)
        if d_behavior==2:
            coefficient=np.mat([1,0,0,0,0,1])



        elif d_behavior==3:
            pass
        elif d_behavior==4:
            pass
        elif d_behavior==5:
            pass
        elif d_behavior==6:
            pass
        elif d_behavior==7:
            pass
        elif d_behavior==8:
            pass
        elif d_behavior==9:
            pass
        elif d_behavior==10:
            pass
        elif d_behavior==11:
            pass
        elif d_behavior==12:
            pass

        J_traffic=self.eval_traffic(vehs_relation,lane_num)
        J_safety=self.eval_safety(vehs_relation,lane_num)
        J_energy=self.eval_energy(vehs_relation,lane_num)
        J_comfort=self.eval_comfort(vehs_relation,lane_num)
        J_regularity=self.eval_regularity(vehs_relation,lane_num)
        J_strategy=self.eval_strategy(vehs_relation,lane_num,d_behavior)

        J_array=np.mat([J_traffic,J_safety,J_energy,J_comfort,J_regularity,J_strategy])

        print("shape:",[coefficient.shape,J_array.shape])
        if J_array.shape==(1,6):
            print("J_array:", J_array)

        J_special =np.matmul(coefficient,J_array)



        return J_special


    def judge_action_complete(self):
        pass

    def judge_action_onestep(self):
        pass
    def cul_lane_rad(self):
        waypoints = self.static_paths[0]
        lane_rad=math.atan((waypoints.point[2].y-waypoints.point[0].y)/(waypoints.point[2].x-waypoints.point[0].x))
        if waypoints.point[2].y - waypoints.point[0].y>0:
            up=True
        else:
            up=False
        if waypoints.point[2].x-waypoints.point[0].x>0:
            right=True
        else:
            right = False
        if up and not right:
            lane_rad = lane_rad + math.pi
        if not up and not right:
            lane_rad = lane_rad + math.pi
        if not up and right:
            lane_rad = lane_rad + math.pi*2

        #unit is rad
        return lane_rad

    def output_lane_rad_tag(self):
        waypoints = self.static_paths[0].point
        lane_rad = self.lane_rad

        tag=1
        if 0 <= lane_rad < math.pi / 4 or math.pi * 7 / 4 < lane_rad <= math.pi * 2:
            tag = 1
        if math.pi / 4 < lane_rad <= math.pi * 3 / 4:
            tag = 2
        if math.pi * 3 / 4 < lane_rad <= math.pi * 5 / 4:
            tag = 3
        if math.pi * 5 / 4 < lane_rad <= math.pi * 7 / 4:
            tag = 4
        return tag
    def find_closest_idx(self):
        waypoints=self.static_paths[0]
        # lane_rad=self.cul_lane_rad()
        lane_rad=self.lane_rad
        rad_tag=self.rad_tag
        # lane_rad2angle=lane_rad*180/math.pi
        current_x = self.veh_info.moving_info.position.point.x
        current_y = self.veh_info.moving_info.position.point.y
        # if 0<=lane_rad <math.pi/4 or math.pi*7/4<lane_rad <=math.pi*2:
        if rad_tag==1:
            return min(range(len(waypoints.point)),
                       key=lambda closest_idx: abs(waypoints.point[closest_idx].x - current_x))
        # if math.pi/4<lane_rad <=math.pi*3/4:
        if rad_tag==2:
            return min(range(len(waypoints.point)),
                       key=lambda closest_idx: abs(waypoints.point[closest_idx].y - current_y))
        # if math.pi*3/4<lane_rad <=math.pi*5/4:
        if rad_tag==3:
            return min(range(len(waypoints.point)),
                       key=lambda closest_idx: abs(waypoints.point[closest_idx].x - current_x))
        # if math.pi*5/4<lane_rad <=math.pi*7/4:
        if rad_tag==4:
            return min(range(len(waypoints.point)),
                       key=lambda closest_idx: abs(waypoints.point[closest_idx].y - current_y))

    def out_static_path(self):

        new_static_paths=[]
        path= []
        x_list=[]
        y_list=[]
        points_num=41
        closest_idx=self.find_closest_idx()

        for i in range(len(self.static_paths)):
            for j in range(points_num):
                x_list.append(self.static_paths[i].point[j+closest_idx].x)
                y_list.append(self.static_paths[i].point[j+closest_idx].y)
            x_list=np.array(x_list)
            y_list=np.array(y_list)
            path.append(x_list)
            path.append(y_list)
            new_static_paths.append(path)

            x_list=[]
            y_list=[]
            path=[]

        return new_static_paths

    #todo
    def static_paths_sort(self):
        """
        get the right static paths rank
        :return:
        """
        # lane_rad=self.cul_lane_rad()
        rad_tag=self.rad_tag

        static_paths_n_points=self.out_static_path()
        # if 0 <= lane_rad < math.pi / 4 or math.pi * 7 / 4 < lane_rad <= math.pi * 2:
        if rad_tag==1:
            static_paths_sorted=sorted(static_paths_n_points,key=lambda x:x[1][0],reverse=True)
        #向东行驶，y越大，车道idx越小
        # if math.pi/4<lane_rad<math.pi*3/4:
        if rad_tag==2:
            static_paths_sorted=sorted(static_paths_n_points,key=lambda x:x[0][0],reverse=False)
        # if math.pi*3 / 4 < lane_rad < math.pi * 5 / 4:
        if rad_tag == 3:
            static_paths_sorted=sorted(static_paths_n_points,key=lambda x:x[1][0],reverse=False)
        # if math.pi*5/4<lane_rad<math.pi*7/4:
        if rad_tag == 4:
            static_paths_sorted=sorted(static_paths_n_points,key=lambda x:x[0][0],reverse=True)
        # self.static_paths_sorted=static_paths_sorted
        return static_paths_sorted

    def get_idx_list(self):
        idx_num=len(self.static_paths)
        idx_0=self.current_idx
        idx_list=[]

        if idx_num==3:
            idx_list=[1,2,3]
        else:
            if idx_0==1 or idx_0==2:
                idx_list = [1,2]
            elif idx_0==3:
                idx_list = [2,3]

        return idx_list

    def eval_traffic(self,vehs_relation,lane_num):
        """

        :return: I_tr
        """
        vehs = self.traffic_info
        #todo
        v_d = 7
        v_d_lane=22
        v_d=max(v_d,v_d_lane)
        idx_list=self.get_idx_list()

        value_traffic=[]
        if bool(vehs_relation):

            for i in idx_list:
                car_leader=vehs_relation['veh_'+str(i)+'l']
                if len(car_leader)>0:
                    car_leader=car_leader[0]
                v_d_c=self.eval_traffic_lane(car_leader,vehs,v_d )
                # print(f'{i}#路径的期望速度为{v_d_c}')
                value_traffic.append(v_d_c)
        else:
            value_traffic=[1,1,1]

        self.value_traffic=value_traffic
        return value_traffic

    def eval_traffic_lane(self,car_leader:dict,vehs,v_d):
        if not isinstance(car_leader,list):
            car_leader = car_leader
            veh_id=list(car_leader.keys())[0]
            length_leader = self.traffic_info[veh_id][1].Length
            dis=list(car_leader.values())[0]
            dis=abs(dis[2])- (self.length + length_leader) / 2
            if dis>5:
                v_l = vehs[veh_id][0].u
                d_b = max(10, 3.6 * v_l)
                v_d0 = min(dis / d_b, 1) * v_d
                v_d0 = dis / d_b * v_d
                # v_d0=np.clip(v_d0,4.5,v_d)

                c_l = min(v_l / v_d, 1)
                v_d0 = round(v_d0 * c_l, 2)
            else:
                v_d0=0

        else:
            v_d0 = v_d  # 前方无车

        v_d0 = round(v_d0 , 2)

        return v_d0

    def get_track_speed(self,vehs_relation):
        """
        get vehcle track speed
        :return:
        """

        #todo
        v_static_path = model_params.target_v
        path_idx_str='veh_'+str(self.current_idx)+'l'
        if bool(vehs_relation):
            if len(vehs_relation[path_idx_str])>0:
                hasleader=1
                car_leader = vehs_relation[path_idx_str][0]
                veh_id = list(car_leader.keys())[0]
                length_leader = self.traffic_info[veh_id][1].Length
                dis = abs(car_leader[veh_id][2])-0.5*(self.length+length_leader)

                v_l = round(self.traffic_info[veh_id][0].u,2)
            else:
                hasleader = 0
                v_l = 22
        else:
            v_l = 22
            hasleader = 0

        v_A = round(self.u,2)

        a=10
        if hasleader :
            if dis > 50:
                v_A =v_static_path
            elif dis>5 and dis<50:
                acc=a/(dis)*(v_l-v_A)
                max_dec=-self.veh_info.base_info.max_dec
                max_acc=3
                if acc<0:
                    acc=max(acc,max_dec)
                else:
                    acc=min(acc,max_acc)

                print("加速度：",acc)
                v_A=v_A+acc
                v_A=max(0,v_A) if v_A<0 else min(v_A,v_static_path)


            else:
                v_A=0
        else:
            v_A=v_static_path

        print("A车期望速度：",v_A)

        return v_A

    def eval_safety(self,vehs_relation,lane_num):
        """

        :return: I_s
        """
        if bool(vehs_relation):
            ov_concern_1l = vehs_relation['veh_1l']
            ov_concern_1f = vehs_relation['veh_1f']
            ov_concern_2l = vehs_relation['veh_2l']
            ov_concern_2f = vehs_relation['veh_2f']
            ov_concern_3l = vehs_relation['veh_3l']
            ov_concern_3f = vehs_relation['veh_3f']


            ov_concern_1 = ov_concern_1l + ov_concern_1f
            ov_concern_2 = ov_concern_2l + ov_concern_2f
            ov_concern_3 = ov_concern_3l + ov_concern_3f

            ov_concern_list=[ov_concern_1,ov_concern_2,ov_concern_3]
            y_lane_list = self.y_lane_list

            risk_list = []
            #todo
            for i in range(lane_num):
                risk = self.eval_podar(i, y_lane_list, ov_concern_list[i])
                risk_list.append(risk)
        else:

            risk_list=[0]*lane_num


        return risk_list



    def get_y_lane_list(self,):
        static_paths=self.static_paths
        y_lane_list=[]
        for item in static_paths:
            y=item.point[0].y
            y_lane_list.append(y)
        y_lane_list=sorted(y_lane_list,reverse=True)

        return y_lane_list



    def eval_podar(self,i,y_lane_list,ov_concern):


        vehs = self.traffic_info

        waypoints=self.static_paths[i].point[0]
        l = self.veh_info.base_info.Length
        w = self.veh_info.base_info.Width
        ms = self.veh_info.base_info.Weight / 1000
        ms1 = self.veh_info.base_info.Weight
        md = self.veh_info.base_info.max_dec

        x = self.veh_info.moving_info.position.point.x
        y = self.veh_info.moving_info.position.point.y
        v = self.veh_info.moving_info.u
        phi = self.veh_info.moving_info.position.phi
        # print('phi:',phi)
        acc_lon = self.veh_info.moving_info.lon_acc
        mileage = self.veh_info.moving_info.position.dis_to_link_end


        podar = PODAR()
        # [step 1] set static parameters (only once)
        podar.add_ego(name=self.ctrl_veh, length=l, width=w, mass=ms1, max_dece=md)
        # [step 2] set dynamic information-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+++++++++++++++++++++++++------------------------------------------------------------------------------------


        #todo
        podar.update_ego(name=self.ctrl_veh, x0=waypoints.x, y0=waypoints.y, speed=v, phi0=phi, a0=acc_lon,
                     mileage=mileage)  # dynamic information
        ov_concern=ov_concern

        # [step 3] here, the ego_surr_list determines which surrounding vehicle will participate the risk calculation
        # podar.ego_surr_list = {'ego_1_name': ['ego1_surr_veh_1_id', 'ego1_surr_veh_2_id', ...],
        #                        'ego_2_name': ['ego2_surr_veh_1_id', 'ego2_surr_veh_2_id', ...]}
        #     podar.ego_surr_list = {ego_name: ov_concern_0}

        # [step 4] for surrounding vehicles:

            # podar.ego_surr_list = {ego_name: ov_concern_1}
        risk=self.cul_risk(vehs,podar,ov_concern)

        return risk

    def cul_risk(self,vehs,podar,ov_concern):
        surr_list = []
        for _ in ov_concern:
            for key in _:
                surr_list.append(key)
        podar.ego_surr_list = {self.ctrl_veh: surr_list}

        for _ in ov_concern:
            for key in _:
                ov = vehs[key]
                podar.add_obj(name=key, x0=ov[0].position.point.x, y0=ov[0].position.point.y, speed=ov[0].u, phi0=ov[0].position.phi, a0=ov[0].lon_acc,
                              length=ov[1].Length,
                              width=ov[1].Width)
        # podar.ego_surr_list={ego_name:surr_list}
        risk, if_real_collision, if_predicted_collision = podar.estimate_risk(self.ctrl_veh)
        risk = round(risk, 2) if risk > 1e-2 else 0
        risk = 2.3 - risk
        risk = round(risk/2.3,2)
        podar.reset()
        return risk

    def eval_energy(self,vehs_relation,lane_num):
        """

        :return:I_e
        """
        J_energy_list=[]
        idx_list = self.get_idx_list()

        for i in idx_list:
            P = self.get_P_value(vehs_relation, i)
            J_energy = -0.47 - 0.15 * P
            J_energy=J_energy+1.67
            J_energy=max(J_energy,0)
            J_energy=round(min(J_energy,1),2)

            J_energy_list.append(J_energy)



        return J_energy_list

    def get_P_value(self,vehs_relation,i):
        if bool(vehs_relation):
            v_l = vehs_relation['veh_'+str(i)+'l']

            ms = self.veh_info.base_info.Weight
            if len(v_l) > 0:
                v_l_item = v_l[0]
                v_l_name = list(v_l_item.keys())[0]
                length_leader = self.traffic_info[v_l_name][1].Length
                velocity_l = round(self.traffic_info[v_l_name][0].u, 2)
                dis = v_l_item[v_l_name][2] -  (self.length + length_leader) / 2

            else:
                #todo
                velocity_l = 20
                dis=3.6*velocity_l

            v_A = self.veh_info.moving_info.u
            a = 10
            acc = a / (dis) * (velocity_l - v_A)

            P = ms * abs(acc) * v_A


        else:
            P=0

        return P


    def eval_comfort(self,vehs_relation,lane_num):
        """

        :return:I_c
        """
        J_comfort_list=[]
        idx_list=self.get_idx_list()
        for i in idx_list:
            acc_lon=self.get_acc_value(vehs_relation,i)
            acc_lat=0
            w=0
            self.acc_lon_list.append(acc_lon)
            if len(self.acc_lon_list)>2:
                self.acc_lon_list = self.acc_lon_list[-2:]
                jerk=self.acc_lon_list[1]-self.acc_lon_list[0]
            else:
                jerk=0
            J_comfort=-(abs(acc_lon)+abs(acc_lat)+abs(0.66*w)+abs(0.66*jerk))
            J_comfort = J_comfort / 100 + 1
            J_comfort=np.clip(J_comfort,0,1)
            J_comfort=round(J_comfort,2)
            J_comfort_list.append(J_comfort)

        return J_comfort_list



    def get_acc_value(self,vehs_relation,i):
        if bool(vehs_relation):
            v_l = vehs_relation['veh_' + str(i) + 'l']

            if len(v_l) > 0:
                v_l_item = v_l[0]
                v_l_name = list(v_l_item.keys())[0]
                length_leader = self.traffic_info[v_l_name][1].Length
                velocity_l = round(self.traffic_info[v_l_name][0].u, 2)
                dis = v_l_item[v_l_name][2] - 0.5 - (4.5 + length_leader) / 2
            else:
                #todo
                velocity_l = 20
                dis = 3.6 * velocity_l



            v_A = self.veh_info.moving_info.u
            a = 20
            acc = a / (dis) * (velocity_l - v_A)
        else:
            #todo
            acc=2
        return acc




    def eval_regularity(self,vehs_relation,lane_num):
        """

        :return:I_r
        """
        #todo

        points_list=[]
        # y_lane_list=self.y_lane_list
        rad_tag=self.rad_tag
        x_lane_list=self.x_lane_list
        y_lane_list=self.y_lane_list

        if rad_tag in [1,3]:
            apply_list=y_lane_list
        else:
            apply_list = x_lane_list


        width_vehicle=self.veh_info.base_info.Width
        width_road=3.5
        if len(apply_list)!=0:
            for y in apply_list:
                points=self.get_points_value(y,width_road,width_vehicle)
                points=points/2
                points_list.append(points)
        else:
            points_list=[0,0,0]

        J_regularity=points_list

        return J_regularity


    def get_points_value(self,y,width_road,width_vehicle):
        points=0
        if y<-width_road:
            if y>=-2*width_road:
                y_bias_cal=y+1.5*width_road
                if y_bias_cal <= 0.5 * (width_vehicle - width_road):
                    points = 2
                else:
                    points = 0
            else:
                points = 2
        else:
            if y>=0:
                points = 2
            else:
                y_bias_cal = y + 0.5 * width_road
                if y_bias_cal>=0.5 * ( width_road-width_vehicle):
                    points = 2

        return points

    def eval_strategy(self,vehs_relation,lane_num,d_behavior):
        """

        :return:I_r
        """
        tag=global_value.tag_getValue()
        if d_behavior==2 and tag==2:
            target_lane_idx=2
        else:
            target_lane_idx='none'

        if lane_num==2:
            if target_lane_idx==0:
                J_strategy = [1, 0]
            elif target_lane_idx==1:
                J_strategy = [0, 1]
            else:
                J_strategy = [1, 1]
        elif lane_num==3:
            if target_lane_idx == 0:
                J_strategy = [1, 0,0]
            elif target_lane_idx == 1:
                J_strategy = [0, 1,0]
            elif target_lane_idx == 2:
                J_strategy=[0,0,1]
            else:
                J_strategy=[1,1,1]


        return J_strategy


    def LC_f(self,list,idx):
        '''
        keep current lane as possible as it can,add a friction force to it.
        :return:
        '''

        LC_f=1
        final_idx=self.current_idx-1 if self.current_idx!=3 else self.current_idx-2


        for i in range(len(list)):
            if i==final_idx:
                continue
            final_idx = idx if list[i]-LC_f>=list[final_idx] else final_idx
            if final_idx == idx:
                break


        return final_idx





    def get_common_routh(self,vehs_relation):
        """
        evaluate different static routh
        :return:output static routh in common driving behavior
        """
        J_common=self.eval_J_common(vehs_relation)


        idx = J_common.index(max(J_common))
        final_idx = self.LC_f(J_common, idx)

        print(f'静态路径{final_idx+1},评估值{J_common}')
        return final_idx

    def get_special_routh(self,vehs_relation,d_behavior):
        """
        evaluate different static routh
        :return:output static routh in common driving behavior
        """
        J_special=self.eval_J_special(vehs_relation,d_behavior)
        J_special=np.array(J_special[0])
        J_special_list=[item for sub in J_special for item in sub]




        idx = J_special_list.index(max(J_special_list))
        final_idx=self.LC_f(J_special_list,idx)

        print(f'静态路径{final_idx+1},评估值{J_special}')
        return final_idx

    def sensity_tune_for_routh(self):
        """

        :return:
        """




    def judge_overtake(self):
        '''

        :return: static path index
        '''
        static_path_idx = 0
        return static_path_idx
    def judge_cutin(self):
        '''

        :return: static path index
        '''
        static_path_idx = 0
        return static_path_idx

    def judge_brake(self):
        '''

        :return: static path index
        '''
        static_path_idx = 0
        return static_path_idx

    def get_cornercase_routh(self):
        """

        :return:output static routh in corner case
        """
    def gen_new_static_path(self,path,target_v,target_phi):
        # path=self.get_n_points()

        # route={'path':path,'v':target_v*np.ones(41),'phi':target_phi**np.ones(41)}

        path.append(target_phi * np.ones(41))
        path.append(target_v* np.ones(41))

        return path
    def get_n_points(self):
        path=[]
        return path


    def output_satic_path(self,path,target_v,target_idx,is_safe_LC,vehs_relation,bias=[0,0]):
        '''

        :return:selected static path
        '''
        route=self.gen_new_static_path(path,target_v,0)


        is_safe_LC_l=is_safe_LC[0]
        is_safe_LC_r=is_safe_LC[1]
        cl_dis=self.get_nearby_veh_realdis(vehs_relation,direction="veh_cl")

        if self.current_idx==target_idx :#跟驰时用跟驰速度模型
            if cl_dis<model_params.g_attention:
                route[3]=np.ones(len(route[3]))*target_v
            else:
                route[3] = np.ones(len(route[3])) * model_params.target_v*model_params.coef_v
        else:
            route[3] = np.ones(len(route[3])) * model_params.target_v

            if is_safe_LC_l and self.current_idx<target_idx:
                route[2] = np.ones(len(route[2])) * 0.1
            if is_safe_LC_r and self.current_idx>target_idx:
                route[2] = np.ones(len(route[2])) * (-0.1)

        return route



    def sampler_behavior(self,timeinterval:int=1):
        """

        :return:
        """
        behavior_list=['std_driving','overtake_right','brake_suddenly','cut_in','lane_changes','flane_slow','turtle_speed','flane_parking','slane_high','reversing','serpentine_driving','off_center']
        behavior_numlist=np.arange(12)+np.ones(12)
        # print(nou_list)
        d_behavior=random.choices(behavior_numlist,weights=[89,1,1,1,1,1,1,1,1,1,1,1])[0]
        d_behavior=int(d_behavior)
        d_behavior = 1
        print(d_behavior,behavior_list[d_behavior-1])
        return d_behavior
    def get_nearby_veh_realdis(self,vehs_relation,direction):
        veh_direction_dis = 100
        if bool(vehs_relation):
            veh_direction = vehs_relation[direction]
            if len(veh_direction)>0:
                vehId_around=list(veh_direction[0].keys())[0]
                length_veh_around=self.traffic_info[vehId_around][1].Length
                if len(veh_direction) > 0:
                    veh_direction_dis = abs(list(veh_direction[0].values())[0][2])- (self.length + length_veh_around) / 2

        return veh_direction_dis

    def judge_LC_conditon(self,vehs_relation):
        is_safe_LC_l=False
        is_safe_LC_r=False


        veh_ll_dis=self.get_nearby_veh_realdis(vehs_relation,direction='veh_ll')
        veh_lf_dis =self.get_nearby_veh_realdis(vehs_relation,direction='veh_lf')
        veh_rl_dis = self.get_nearby_veh_realdis(vehs_relation, direction='veh_rl')
        veh_rf_dis = self.get_nearby_veh_realdis(vehs_relation, direction='veh_rf')


        dis_turn_base=model_params.turn_dis_base



        if self.current_idx!=self.left_1st and abs(veh_ll_dis)>dis_turn_base and abs(veh_lf_dis)>dis_turn_base:
            is_safe_LC_l=True
            print("可以左转")

        if self.current_idx!=self.right_1st and abs(veh_rl_dis)>dis_turn_base and abs(veh_rf_dis)>dis_turn_base:
            is_safe_LC_r=True
            print("可以右转")

        is_safe_LC=[is_safe_LC_l,is_safe_LC_r]
        print("能否换道：",is_safe_LC)
        return is_safe_LC

    def is_in_left_lane(self):
        lane2_left_tag=False
        if self.current_idx == self.left_1st:
            lane2_left_tag = True
        return lane2_left_tag
    def is_behavior_completed(self,tag):
        completed_tag=False
        if tag==2:
            completed_tag=True

        return completed_tag

    def output_correct_pathandidx(self,is_safe_LC,static_path_idx,diff_path,d_behavior):
        real_static_path_idx = static_path_idx + 1
        case_safe_turnright = self.current_idx < real_static_path_idx and is_safe_LC[1]
        case_safe_turnleft = self.current_idx > real_static_path_idx and is_safe_LC[0]

        if case_safe_turnleft or case_safe_turnright:
            print("real_static_path_idx:", real_static_path_idx)
            #最左侧

            ref_path = diff_path[static_path_idx]
            idx = real_static_path_idx
        else:

            ref_path = diff_path[self.current_idx - 1]
            idx = self.current_idx
            print("不换道", idx)


        tag=global_value.tag_getValue()
        if d_behavior!=1 and tag==2:
            ref_path = diff_path[static_path_idx ]
            idx = static_path_idx + 1

        return [ref_path,idx]

    veh_concern=''

    def IDC_decide(self,vehs_relation,d_behavior):
        '''
        logic about making decision，using valuation funtion
        to decide to select the optimal static routh

        :return:static routh
        '''

        vehs=self.traffic_info

        if bool(vehs_relation):
            sort_path = self.static_paths_sort()
            is_safe_LC = self.judge_LC_conditon(vehs_relation)
        else:
            sort_path=self.static_paths_sort()
            is_safe_LC=True
        ref_path=[]
        # lane2_left_tag=False



        if d_behavior==1:
            print("选择标准驾驶行为")
            static_path_idx=self.get_common_routh(vehs_relation)

            if bool(vehs_relation):
                ref_data = self.output_correct_pathandidx( is_safe_LC, static_path_idx, sort_path
                                                            ,d_behavior)
            else:
                ref_data=[self.static_paths,static_path_idx]
            ref_path = ref_data[0]
            idx = ref_data[1]



        elif d_behavior==2:
            print("选择异常驾驶行为：右则超车")
            static_path_idx = self.get_special_routh(vehs_relation,d_behavior)
            print("static_path_idx:", static_path_idx)

            #异常行为不返回原车道 需要修改下面参数，加入异常行为类别判断
            ref_data=self.output_correct_pathandidx(is_safe_LC,static_path_idx,
                                                    sort_path,d_behavior,tag=0)
            ref_path=ref_data[0]
            idx=ref_data[1]

            if len(vehs_relation['veh_2l']) != 0:
                leader_id=list(vehs_relation['veh_2l'][0].keys())[0]
                global veh_concern
                veh_concern=leader_id
            else:
                leader_id=''

            if len(vehs_relation['veh_2f']) != 0:
                follower_id = list(vehs_relation['veh_2f'][0].keys())[0]
                veh_2f=list(vehs_relation['veh_2f'][0].keys())
            else:
                follower_id=''
                veh_2f =[]


            if len(vehs_relation['veh_ll'])!=0:
                veh_ll=list(vehs_relation['veh_ll'][0].keys())
            else:
                veh_ll=[]
            if len(vehs_relation['veh_lf'])!=0:
                veh_lf=list(vehs_relation['veh_lf'][0].keys())
            else:
                veh_lf=[]
            if len(vehs_relation['veh_cf'])!=0:
                veh_cf=list(vehs_relation['veh_cf'][0].keys())
            else:
                veh_cf=[]

            if leader_id != '':
                if leader_id in veh_ll:
                    global_value.tag_setValue(1)

            if veh_concern in veh_lf:
                print(f"veh_concern:{veh_concern}")

                global_value.tag_setValue(2)
                dis = vehs[follower_id].lane_position - vehs[self.ctrl_veh].lane_position + 5
                if  dis < -5:
                    tag=global_value.tag_getValue()

                    ref_data = self.output_correct_pathandidx(is_safe_LC, static_path_idx,
                                                              sort_path, d_behavior, tag)
                    ref_path=ref_data[0]
                    idx=self.current_idx+1
                    print("准备返回原车道:", idx)
            tag = global_value.tag_getValue()
            if veh_concern in veh_cf and tag==2:
                global_value.tag_setValue(0)
                print('超车已完成，已到原车道:', static_path_idx)

        elif d_behavior==3:
            print("选择异常驾驶行为：急刹车")
            static_path=self.get_cornercase_routh()
            # ref_path=self.output_satic_path()
            ref_path = sort_path[0]
        elif d_behavior==4:
            print("选择异常驾驶行为：加塞")
            ref_path = sort_path[0]
        elif d_behavior==5:
            print("选择异常驾驶行为：连续变道")
            ref_path = sort_path[0]
        elif d_behavior==6:
            print("选择异常驾驶行为：快车道慢速行驶")
            ref_path = sort_path[0]
        elif d_behavior==7:
            print("选择异常驾驶行为：龟速行驶")
            ref_path = sort_path[0]
        elif d_behavior==8:
            print("选择异常驾驶行为：快车道停车")
            ref_path = sort_path[0]
        elif d_behavior==9:
            print("选择异常驾驶行为：慢车道高速行驶")
            ref_path = sort_path[0]
        elif d_behavior==10:
            print("选择异常驾驶行为：车道内倒车")
            ref_path = sort_path[0]
        elif d_behavior==11:
            print("选择异常驾驶行为：蛇形驾驶")
            ref_path = sort_path[0]
        elif d_behavior==12:
            print("选择异常驾驶行为：偏离道路中心线驾驶")
            ref_path = sort_path[0]
        ref_path=[ref_path,idx]
        return ref_path



if __name__ == "__main__":
    surr=Surr()
    surr.test(info_update)




