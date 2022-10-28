"""
This code is an implementation of power system behavior simulation under extreme events for the under review paper 
"Real-time Resilience Assessment for Power Systems under Extreme Disasters: A Spatial Temporal Graph-based Approach".

浙江大学，
朱禹泓，
2022.10.26

"""



# 导入依赖的模块
import sys 
sys.path.append(r"../")
import pypower
import pypower.case118 as case118
import pypower.runpf as runpf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import copy

# 环境初始化

font = {
        	    'family' : 'MicroSoft YaHei', #宋体 Simsun
           	 'weight' : 'bold',
            
      	  }
matplotlib.rc('font',**font)


# constant
v_dtower = 35 # 塔的设计风速
v_dline = 35 #线路的设计风速
gamma = 0.2 #风速影响因子
dt = 15 #时间间隔
dl = 1 #线路长度标幺值
suggested_vt_range = [40,70] #建议设置的台风风速范围
simulation_times = 100




#functions
def lambda_mk(v_mk):
    """
    
    Parameters
    ----------
    v_mk : TYPE
        wind speed at tower k and transmission m .

    Returns
    -------
    failure rate lambda_mkt

    """
    
    if v_mk <= v_dtower:
        lambda_mkt = 0
    elif v_mk <= 2*v_dtower and v_mk > v_dtower:
        lambda_mkt = np.e**(gamma*(v_mk-2*v_dtower))
    elif v_mk > 2*v_dtower:
        lambda_mkt = 1
    
    return lambda_mkt

"""
'''
test for lambda_mk

'''
vs = []
failure_rates = []
for v in range(0,3*v_dtower):
    vs.append(v)
    failure_rates.append(lambda_mk(v))
plt.plot(vs,failure_rates)
"""

def p_mk(v_mk):
    """
    

    Parameters
    ----------
    v_mk : TYPE
        wind speed at tower k and transmission m .
    dt : TYPE
        time period.

    Returns
    -------
    p_mkt : TYPE
       tower failure rate.

    """
    if lambda_mk(v_mk) != 1:
        p_mkt = 1-np.exp(-dt*lambda_mk(v_mk)/(1-lambda_mk(v_mk)))
    else:
        p_mkt = 1
    return p_mkt

"""
'''
test for p_mk

'''
vs = []
tower_failure_rates = []
for v in range(0,3*v_dtower):
    vs.append(v)
    tower_failure_rates.append(p_mk(v))
plt.plot(vs,tower_failure_rates)
"""

def lambda_ml(v_ml):
    """
    
    Parameters
    ----------
    v_ml : TYPE
        wind speed at line l and transmission m .

    Returns
    -------
    failure rate lambda_mlt

    """
    lambda_mlt = min(np.exp(11*v_ml/v_dline - 18) * dl,1)

    
    return lambda_mlt
"""
'''
test for lambda_ml

'''
vs = []
line_failure_rates = []
for v in range(0,3*v_dtower):
    vs.append(v)
    line_failure_rates.append(lambda_ml(v))
plt.plot(vs,line_failure_rates)
"""

def p_ml(v_ml):
    """
    

    Parameters
    ----------
    v_ml : TYPE
        wind speed at tower k and line m .
    dt : TYPE
        time period.

    Returns
    -------
    p_mkt : TYPE
       line failure rate.

    """

    p_mlt = 1-np.exp(-dt*lambda_ml(v_ml))  
    return p_mlt
"""
'''
test for p_mk

'''
vs = []
line_failure_rates = []
for v in range(0,3*v_dtower):
    vs.append(v)
    line_failure_rates.append(p_ml(v))
plt.plot(vs,line_failure_rates,label='线路时间段内故障概率')
plt.legend()
"""

def p_branch(v):
    """
    

    Parameters
    ----------
    v : TYPE
        wind speed at branch b.

    Returns
    -------
    p : TYPE
        failure rate.

    """
    p = 1-(1-p_ml(v))*(1-p_mk(v))*(1-p_mk(v))
    return p
"""
'''
test for p_mk

'''
vs = []
branch_failure_rates = []
for v in range(0,3*v_dtower):
    vs.append(v)
    branch_failure_rates.append(p_branch(v))
plt.plot(vs,branch_failure_rates,label='线路时间段内故障总概率')
plt.legend()
"""

def time2repair(v_t):
    """
    

    Parameters
    ----------
    v : TYPE
        wind speed at the moment that the branch is broken.

    Returns
    -------
    ttr : TYPE
        time to repair for the branch.

    """
    ttr = int(random.uniform(4,8)*v_t*2/(v_dtower+v_dline))
    return ttr

def get_topo(case):
    branches = case['branch']
    buses = case['bus']
    bus_num = buses.shape[0]
    #print(bus_num)
    A = np.zeros((118,118))
    for branch_num in range(branches.shape[0]):
        from_bus_label,to_bus_num = branches[branch_num,0], branches[branch_num,1]
        #print(from_bus_label,to_bus_num)
        A[int(from_bus_label-1),int(to_bus_num-1)] = 1
    return A

def get_feat(case):
    buses = case['bus']
    bus_num = buses.shape[0]
    F = np.zeros((118,5))
    for bus_num in range(buses.shape[0]):
        bus_label = buses[bus_num,0]
        act_bus_num = int(bus_label-1)
        F[act_bus_num,2] = buses[bus_num,2]
        F[act_bus_num,3] = buses[bus_num,3]
        F[act_bus_num,0] = buses[bus_num,7]
        F[act_bus_num,1] = buses[bus_num,8]
    for gen_num in range(case['gen'].shape[0]):
        bus_label = case['gen'][gen_num,0]
        act_bus_num = int(bus_label-1)
        F[act_bus_num,4] = case['gen'][gen_num,1]
    return F

def run_a_typhoon(case,typhoon):

    
    return 0

# classes



class Initcase:
    def __init__(self,casedata):
        self.case = casedata
        self.branches = []
        for branch_num in range(self.case['branch'].shape[0]):
            self.branches.append(Branch(self.case['branch'][branch_num,:],branch_num))
    
    def bus_label2branch_num(self,bus_num1,bus_num2):
        bus_num1,bus_num2 = int(bus_num1),int(bus_num2)
        for branch in self.branches:
            if bus_num1 == branch.from_bus_num and bus_num2 == branch.to_bus_num:
                return branch.branch_num
            elif bus_num2 == branch.from_bus_num and bus_num1 == branch.to_bus_num:
                return branch.branch_num
            else:
                continue
        return None
    
    def branch_num2bus_label(self,branch_num):
        return (self.branches[branch_num].from_bus_num,self.branches[branch_num].to_bus_num)
    
    def buslist2branchlist(self,buslist):
        branchlist = []
        for buses in buslist:
            branchlist.append(self.bus_label2branch_num(buses[0],buses[1]))
        return branchlist
    
    @property
    def branchsize(self):
        return len(self.branches)
    
    @property
    def availablebranchsize(self):
        number = 0
        for branch in self.branches:
            if branch.available == True:
                number += 1
        return number
    
    def update_branch_state(self,cur_typhoon_traj, cur_typhoon_speed):
        # 判定是否可修复
        for branch in self.branches:
            if branch.branch_num in cur_typhoon_traj:
                branch.repairable = False
            else:
                branch.repairable = True
        
        
        for branch_num in cur_typhoon_traj:
            if self.branches[branch_num].available == True:
                p_bran = p_branch(cur_typhoon_speed)
                random_num = random.uniform(0,1)
                
                if p_bran>=random_num:
                    self.branches[branch_num].available = False
                
                else:
                    continue
        del_branch_list = []
        for branch in self.branches:
            if branch.available == False:
                del_branch_list.append(branch.branch_num)
        
        self.case['branch'] = copy.deepcopy(ori_case118['branch'])
        self.case['branch'] = np.delete(self.case['branch'],del_branch_list,axis=0)
        #print('-'*20,time_step,'-'*20)
        #print(self.availablebranchsize)
        #print(self.case['branch'].shape[0])
        
    def topo_conn_check(self):
        bus_dict = self.case['bus']
        sets = []
        all_bus_list = copy.deepcopy(bus_dict[:,0])
        remain_bus_list = copy.deepcopy(bus_dict[:,0])
        all_bus_list = [int(all_bus_list[i]) for i in range(len(all_bus_list))]
        remain_bus_list = [int(remain_bus_list[i]) for i in range(len(remain_bus_list))]
        nodes = self.nodes().nodes
        for node_label_ in nodes:
            node = nodes[node_label_]
            if node.nodelabel in remain_bus_list:
                sets.append(node.all_conn_labels())
                for buslabel in node.all_conn_labels():
                    remain_bus_list.remove(buslabel)
            else:
                continue
        return sets
    
    def island_handle(self):
        sets = self.topo_conn_check()
        #lengths = [len(sets[i]) for i in range(len(sets))]   
        if len(sets) == 1: return None
        self.case['bus'] = copy.deepcopy(ori_case118['bus'])
        set_num = 1
        # 设置分区号与断面号
        for set_ in sets:
            for buslabel in set_:
                self.case['bus'][buslabel-1,6] = set_num
                self.case['bus'][buslabel-1,10] = set_num
            set_num += 1
        #检查是否同时有发电机和负荷:
        del_bus_list = []
        for set_ in sets:
            if len(set_)>1:
                load_exist = False
                gen_exist = False
                slack_exist = False
                for buslabel in set_:
                    if int(self.case['bus'][buslabel-1,1]) == 1:
                        load_exist = True
                    if int(self.case['bus'][buslabel-1,1]) == 2:
                        gen_exist = True
                    if int(self.case['bus'][buslabel-1,1]) == 3:
                        slack_exist = True    
                    if load_exist and gen_exist and slack_exist:
                        break
                if  (not gen_exist) and (not slack_exist):
                    for buslabel in set_:
                        self.case['bus'][buslabel-1,2] = 0
                        self.case['bus'][buslabel-1,3] = 0
                        
                if (gen_exist) and (not slack_exist):
                    for buslabel in set_:
                        if int(self.case['bus'][buslabel-1,1]) == 2:
                            self.case['bus'][buslabel-1,1] = 3
                            break
            else:
                row_num = set_[0]-1
                del_bus_list.append(row_num)
        self.case['bus'] = np.delete(self.case['bus'],del_bus_list,axis=0)
        #print(self.case['bus'].shape)
        
    def nodes(self):
        brancharray = copy.deepcopy(self.case['branch'])
        busarray = copy.deepcopy(self.case['bus'])
        nodes = NodesGraph(brancharray,busarray)
        return nodes
    
    def runpf(self):
        #print(self.case['bus'].shape)
        return runpf.runpf(self.case)
    
    def busarray_restore(self):
        self.case['bus'] = copy.deepcopy(ori_case118['bus'])
        
class NodesGraph:
    def __init__(self,brancharray,busarray):
        node_num = busarray.shape[0]
        nodes = {}
        for node_num_ in range(node_num):
            node = Node(int(busarray[node_num_,0]))
            nodes[int(busarray[node_num_,0])] = node
        for branch_num in range(brancharray.shape[0]):
            node_label1, node_label2 = int(brancharray[branch_num,0]), int(brancharray[branch_num,1])
            nodes[node_label1].conn_nodes.append(nodes[node_label2])
            nodes[node_label2].conn_nodes.append(nodes[node_label1])
        self.nodes = nodes
            
        
class Node:
    def __init__(self,nodelabel):
        self.nodelabel = nodelabel
        self.conn_nodes = []
        
    def direct_conn_labels(self):
        labels = []
        for node in self.conn_nodes:
            labels.append(node.nodelabel)
        return labels
    
    def all_conn_labels(self):
        labels = [self.nodelabel]
        nodes = [self]
        while 1:
            bef_label_length = len(labels)
            for node in nodes:
                temp_conn_nodes = node.conn_nodes
                for tempnode in temp_conn_nodes:
                    if tempnode.nodelabel not in labels:
                        labels.append(tempnode.nodelabel)
                        nodes.append(tempnode)
                    pass
            aft_label_length = len(labels)
            if bef_label_length == aft_label_length:
                break
        return labels
            

class Branch:
    def __init__(self,branch_data,branch_num):
        self.branch_num = branch_num
        self.from_bus_num = int(branch_data[0])
        self.to_bus_num = int(branch_data[1])
        self.available = True
        self.repairable = False
        self.ttr = None

if __name__ == "__main__":
    ori_case118 = case118.case118()
    
    ori_A = get_topo(ori_case118)
    initcase = Initcase(ori_case118)
    preset_typhoon_traj = [        
            initcase.buslist2branchlist([[1,3],[3,5],[5,6],[6,7],[3,12],[5,11],[5,4]]),#1
            initcase.buslist2branchlist([[5,4],[4,11],[5,11],[6,5],[6,7],[12,11],[11,13]]),#2
            initcase.buslist2branchlist([[4,11],[13,11],[16,12],[16,17],[17,15],[17,31],[17,113]]),#3
            initcase.buslist2branchlist([[16,17],[17,18],[17,30],[8,30],[30,38],[17,15],[17,113],[17,31]]),#4
            initcase.buslist2branchlist([[18,19],[19,20],[30,38]]),#5
            initcase.buslist2branchlist([[24,70],[70,71],[70,74],[70,75],[70,69]]),#6
            initcase.buslist2branchlist([[70,71],[70,24],[75,74],[70,75],[71,73]]),#7
            initcase.buslist2branchlist([[74,75],[75,118],[118,76],[75,77],[77,76],[77,82]]),#8
            initcase.buslist2branchlist([[118,76],[76,77],[77,78],[78,79],[77,82],[83,82],[82,96]]),#9            
            initcase.buslist2branchlist([[78,79],[82,96],[96,97],[96,94],[96,95],[94,95],[93,94],[94,100]]),#10
            initcase.buslist2branchlist([[94,95],[94,93],[96,94],[100,94],[100,98],[96,97]]),#11
            initcase.buslist2branchlist([[92,100],[100,101],[100,103],[100,104],[103,104],[103,105],[110,111],[109,110],[108,109]]),#12
        ]
    
    
    #whole-process simulation
    results = {}
    for sim_time in range(simulation_times):
        preset_typhoon_windspeed = [round(random.uniform(suggested_vt_range[0],suggested_vt_range[1])) for i in range(len(preset_typhoon_traj))]
        preset_typhoon_case = {'traj':preset_typhoon_traj,'windspeed':preset_typhoon_windspeed}
        sim_case = copy.deepcopy(initcase)
        unconv_exist = False
        state_series = []
        is_end = False
        for time_step in range(len(preset_typhoon_traj)):
            
        #for time_step in range(len(preset_typhoon_traj)):
            cur_typhoon_traj, cur_typhoon_speed =  preset_typhoon_case['traj'][time_step],preset_typhoon_case['windspeed'][time_step]
            sim_case.update_branch_state(cur_typhoon_traj, cur_typhoon_speed)
            #xxx = sim_case.topo_conn_check()
            sim_case.island_handle()
            pfres = sim_case.runpf()
            if pfres[1] == 0 :unconv_exist = True
            cur_A = get_topo(pfres[0])
            cur_F = get_feat(pfres[0])
            cur_state = [cur_A,cur_F]
            state_series.append(cur_state)
            #yyy = copy.deepcopy(sim_case.case)
            if time_step == len(preset_typhoon_traj)-1:is_end = True
            if not is_end:sim_case.busarray_restore()
        end_load = pfres[0]['bus'].sum(axis=0)[2]
        total_load = ori_case118['bus'].sum(axis=0)[2]
        load_loss_rate = (total_load-end_load)/total_load
        result = {'input':state_series,'output':load_loss_rate}
        results[sim_time] = result
    np.save('results', results, allow_pickle=True, fix_imports=True)
        
        
