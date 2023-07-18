import grpc
import copy
from traffic_model.traffic_model import Surr,IDC_decision
from matplotlib import pyplot as plt

from risenlighten.lasvsim.process_task.api.cosim.v1 import simulation_pb2
from risenlighten.lasvsim.process_task.api.cosim.v1 import simulation_pb2_grpc
from dataclasses import dataclass
import time
import random
import numpy as np
import math

@dataclass
class parametes_set():
    desir_dis:float=50
    veh_num:int=10

    def update_decay_param(self,value):
        self.desir_dis=value



def run():
    with grpc.insecure_channel('8.146.201.197:31244') as channel:
        # with grpc.insecure_channel('120.46.203.61:9001') as channel:
        # jfdd:[4034,1775],[4035,1777],[4027,1753][4036,1780][4036,1783][4040,1791][4039,1790][4061,2011][4062,2014]

        #复杂交通:[4070,2035]
        #水平复杂场景：[4084,2070]
        #垂直场景：[4085,2075]
        #弯曲道路（南大干线）:[4091,2093]
        #金枫大道:[4092,2100]
        params = parametes_set()

        #[npc25,npc10,npc60]
        task_id = 4092
        record_id = 2100
        vehicle_id = 'ego'
        progress_times = 0
        choosed_list = []

        stub = simulation_pb2_grpc.CosimStub(channel)
        startResp = stub.Start(simulation_pb2.StartSimulationReq(task_id=task_id,record_id=record_id))

        UpdateVehicleInfoReult = stub.UpdateVehicleInfo(simulation_pb2.UpdateVehicleInfoReq(
            simulation_id=startResp.simulation_id, vehicle_id=vehicle_id,
            ))
        # parametes_set=params.parametes_set()


        for i in range(250):

            start_time=time.time()

            # 获取自车以及自车的周车列表
            vehicle = stub.GetVehicle(simulation_pb2.GetVehicleReq(
                simulation_id=startResp.simulation_id, vehicle_id=vehicle_id))
            checkError(vehicle.error)
            # print(vehicle)
            veh_around=vehicle.vehicle.around_moving_objs
            # veh_id_around=[x.id for x in veh_around]
            veh_id_around = list(map(lambda x: x.id, veh_around))
            #得到指定范围内周车
            nearest_vehs=get_nearest_vehs(vehicle,veh_around,params.desir_dis,params.veh_num)
            nearest_veh_id_around = list(map(lambda x: x.id, nearest_vehs))

            if len(nearest_veh_id_around)>30:
                params.update_decay_param(value=30)
            else:
                params.update_decay_param(value=50)

            stepResult = stub.NextStep(simulation_pb2.NextStepReq(
                simulation_id=startResp.simulation_id))
            print('*'*100)
            print('stepResult:',stepResult.state.progress)

            end_time_nextstep = time.time()
            print(f'单个step中nextstep花费时间:{round((end_time_nextstep - start_time), 2)}s')

            if stepResult.state.progress==1:
                progress_times+=1
            # if stepResult.state.progress in [0,200] or i>10:
            #     break
            #     tag=True

            if checkError(stepResult.error):
                break

            if i > 1:
                input_params = process_input_params(vehicle,stub,startResp,nearest_veh_id_around)
                # print(input_params)
                # control_value = [[-2], [0]]
                veh_info=input_params['veh_info']
                traffic_info=input_params['traffic_info']
                if not bool(traffic_info):
                    break
                #筛选相同路段的车辆
                new_vehs_list=get_same_seg_vehs(veh_info,nearest_veh_id_around,traffic_info)

                print(f"考虑的周车的数量:{len(nearest_veh_id_around)},相同路段的周车数量:{len(new_vehs_list)}")
                surr = Surr()
                control_value = surr.update(input_params,new_vehs_list)
                # if i>50:
                #     control_value=[[-2],[0.1]]
                print(f"control_value:{control_value}")

                end_time_for_model=time.time()
                print(f'单个step中交通流模型花费时间:{round((end_time_for_model - start_time),2)}s')

                leader_list=surr.get_leader_list()
                follower_list=surr.get_follower_list()
                # print(f'leader_list:{leader_list}')
                # print(f'follower_list:{follower_list}')

                leader_info_list=surr.get_veh_acc(leader_list,target_v=6)
                print(f'leader_info_list:{leader_info_list}')
                follower_info_list = surr.get_veh_acc(follower_list, target_v=3)
                print(f'follower_info_list:{follower_info_list}')

                vehicleControleReult = stub.SetVehicleControl(simulation_pb2.SetVehicleControlReq(
                    simulation_id=startResp.simulation_id, vehicle_id=vehicle_id,
                    lon_acc=control_value[0][0], ste_wheel=control_value[1][0]))
                checkError(vehicleControleReult.error)


                if i<100 and i%10==0:
                    if len(follower_info_list)>2:
                        choosed_list=random.sample(follower_info_list,2)
                    elif len(leader_info_list)>2:
                        choosed_list=random.sample(leader_info_list,2)
                    else:
                        choosed_list=leader_info_list
                if i>=100:
                    choosed_list = leader_info_list+follower_info_list
                choosed_list=choosed_list if  choosed_list else follower_info_list
                print(f'choosed_list:{choosed_list}')
                for leader_info in choosed_list:
                    leader_veh=leader_info[0]
                    leader_acc=leader_info[1]
                    vehicleControleReult_others = stub.SetVehicleControl(simulation_pb2.SetVehicleControlReq(
                        simulation_id=startResp.simulation_id, vehicle_id=leader_veh,
                        lon_acc=leader_acc, ste_wheel=0))
                    checkError(vehicleControleReult_others.error)



            print('progress_times:', progress_times)

            end_time=time.time()
            print(f'单个step花费时间:{round((end_time - start_time),2)}s')
                # 状态不正确，结束
            if (stepResult.state.progress <= 0) or (stepResult.state.progress >= 100) :
                    # or progress_times>10:

                print(
                    f"仿真结束,状态：{stepResult.state.msg}")
                break

        print(f"id：{startResp.simulation_id}")

        stub.Stop(simulation_pb2.StopSimulationReq(
            simulation_id=startResp.simulation_id))
        result = stub.GetResults(simulation_pb2.GetResultsReq(
            simulation_id=startResp.simulation_id))
        # print(f"仿真结束,结果:{result.results}")


def checkError(err):
    if err is None:
        return False

    if err.code != 0:
        print(err.msg)
        return True
    return False

#主要获取车辆真实位置信息
def get_allvehs_info(stub,startResp,veh_list):
    # veh_list = stub.GetVehicleIdList(simulation_pb2.GetVehicleIdListReq(
    #     simulation_id=startResp.simulation_id))
    # print(veh_list)

    all_vehs={}

    for veh in veh_list:
        vehicle = stub.GetVehicle(simulation_pb2.GetVehicleReq(
            simulation_id=startResp.simulation_id, vehicle_id=veh))
        checkError(vehicle.error)
        id=vehicle.vehicle.info.id
        all_vehs[id]=[vehicle.vehicle.info.moving_info,vehicle.vehicle.info.base_info]

    return all_vehs

def process_input_params(vehicle,stub,startResp,veh_list):
    start_time=time.time()
    vehicle_info = vehicle.vehicle.info
    # around_moving_objs = vehicle.vehicle.around_moving_objs
    print("=" * 100)
    # print('around_moving_objs:', around_moving_objs)

    static_paths = vehicle_info.static_path
    print("=" * 100)
    # print('static_paths:', static_paths)

    all_moving_objs=get_allvehs_info(stub,startResp,veh_list)
    # print(all_moving_objs)

    input_params={'veh_info':vehicle_info,'traffic_info':all_moving_objs,'static_paths':static_paths}

    end_time_process_input=time.time()
    print(f'单个step中process_input花费时间:{round((end_time_process_input - start_time), 2)}s')


    return input_params

def order_paths(static_paths):

    y_list=[]
    for path in static_paths:
        y_value=path.point[-1].y
        y_list.append(y_value)
    idx1=y_list.index(max(y_list))
    new_paths=[]
    new_paths.append(static_paths[idx1])
    m_paths=copy.deepcopy(y_list)
    m_paths=m_paths.pop(y_list[idx1])
    if len(m_paths)==1:
        idx2=y_list.index(min(y_list))
        new_paths.append(static_paths[idx2])


    y_2=max(m_paths)
    idx2=y_list.index(y_2)
    new_paths.append(static_paths[idx2])

    while len(m_paths)>1:
        y_value_next=max(m_paths)
        idx_next=y_list.index(y_value_next)
        new_paths.append(static_paths[idx_next])
        m_paths.pop(y_value_next)

    return new_paths

def get_nearest_vehs(veh_info,around_vehs,desir_dis=50,veh_num=10):
    host_veh_point=veh_info.vehicle.info.moving_info.position.point
    host_veh_point=np.array([host_veh_point.x,host_veh_point.y])

    nearest_vehs=[item for item in around_vehs
                  if np.linalg.norm(host_veh_point-np.array([item.moving_info.position.point.x,item.moving_info.position.point.y]))<desir_dis]
    # nearest_vehs = [item for item in around_vehs
    #                 if math.sqrt((item.moving_info.position.point.x-host_veh_point.x)**2+
    #                              (item.moving_info.position.point.y-host_veh_point.y)**2) < desir_dis]
    nearest_vehs=sorted(nearest_vehs,key=lambda item:math.sqrt((item.moving_info.position.point.x-host_veh_point[0])**2+
                                  (item.moving_info.position.point.y-host_veh_point[1])**2))[:min(len(nearest_vehs),veh_num)]
    return nearest_vehs

def get_same_seg_vehs(host_veh,around_vehs,traffic_info):
    host_veh_link=host_veh.moving_info.position.link_id
    # list_traffic=[traffic_info[x][0].position.link_id for x in around_vehs]
    # new_list=[x for x in list_traffic if x==host_veh_link]
    new_vehs=[x for x in around_vehs if traffic_info[x][0].position.link_id==host_veh_link]
    # print(list_traffic,new_list)
    print(f"new_vehs:{new_vehs}")
    return new_vehs


def pyplot(data):
    pass



if __name__ == '__main__':
    run()
