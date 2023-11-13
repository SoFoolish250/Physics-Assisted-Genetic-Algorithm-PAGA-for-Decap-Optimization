import numpy as np
from modOnlineGA_paper import geneticalgorithm as ga
import matplotlib.pyplot as plt
import copy
import os
import skrf as rf
from tqdm import tqdm
from numba import jit
import time




def short_1port(input_net, map2orig_input=[0, 1], shorted_port=1):
    short_net = input_net.s11.copy()
    short_net.s = -1*np.ones(short_net.f.shape[0])
    output_net = rf.network.connect(input_net, shorted_port, short_net, 0)
    map2orig_output = map2orig_input.copy()
    del map2orig_output[shorted_port]
    return output_net, map2orig_output

def init_decap_list(freq):
    decap_list = []
    for filename in os.listdir("./data/decap/"):
        temp,_ = short_1port(rf.Network("./data/decap/" + filename).interpolate(freq))
        decap_list.append(temp.z)
    return decap_list

def get_custom_target(freq):
    ztarget = np.full(freq.shape[0],10,dtype=np.float32)
    for i in range(freq.shape[0]):
        if freq[i] <= 1e6:
            ztarget[i] = 0.5e-3            
    return ztarget




def connect_1decap(input_net_z, map2orig_input, connect_port, decap_z11):
    a_ports = list(range(input_net_z.shape[1]))
    del a_ports[connect_port]
    p_ports = [connect_port]
    Zaa = input_net_z[np.ix_(list(range(0,input_net_z.shape[0])), a_ports, a_ports)]
    Zpp = input_net_z[np.ix_(list(range(0,input_net_z.shape[0])), p_ports, p_ports)]
    Zap = input_net_z[np.ix_(list(range(0,input_net_z.shape[0])), a_ports, p_ports)]
    Zpa = input_net_z[np.ix_(list(range(0,input_net_z.shape[0])), p_ports, a_ports)]
    Zqq = decap_z11
    z_connect = Zaa - np.matmul(np.matmul(Zap,np.linalg.inv(Zpp+Zqq)),Zpa)
    map2orig_output = map2orig_input.copy()
    del map2orig_output[connect_port]
    return z_connect, map2orig_output
 

def connect_ndecap_together(input_z, map2orig_input, connect_port_list, decap_num_list):
    a_ports = list(range(input_z.shape[1]))
    a_ports = [i for i in a_ports if i not in connect_port_list]
    p_ports = connect_port_list
    Zaa = input_z[np.ix_(list(range(0,input_z.shape[0])), a_ports, a_ports)]
    Zpp = input_z[np.ix_(list(range(0,input_z.shape[0])), p_ports, p_ports)]
    Zap = input_z[np.ix_(list(range(0,input_z.shape[0])), a_ports, p_ports)]
    Zpa = input_z[np.ix_(list(range(0,input_z.shape[0])), p_ports, a_ports)]
    Zqq = np.zeros((input_z.shape[0], len(connect_port_list), len(connect_port_list)),dtype = 'complex')
    for i in range(len(connect_port_list)):
        Zqq[:,i,i] = DECAP_LIST[decap_num_list[i]].reshape((input_z.shape[0]))
    z_connect = Zaa - np.matmul(np.matmul(Zap,np.linalg.inv(Zpp+Zqq)),Zpa)
    map2orig_output = [map2orig_input[i] for i in range(len(map2orig_input)) if map2orig_input[i] not in connect_port_list]
    return z_connect, map2orig_output



# most time consuming part, can be accelerated by numba jit
@jit(nopython = True)
def connect_ndecap_together_calc_znn(input_z, connect_port_list, decap_num_list):
    a_ports = np.arange(1)
    p_ports = connect_port_list
    
    Zaa = input_z[:,a_ports][:,:,a_ports]   #n*1*1
    Zpp = input_z[:,p_ports][:,:,p_ports]   #n*p*p
    Zap = input_z[:,a_ports][:,:,p_ports]   #n*1*p
    Zpa = input_z[:,p_ports][:,:,a_ports]   #n*p*1
    
    
    Zqq = np.zeros((input_z.shape[0], len(connect_port_list), len(connect_port_list)),dtype = np.complex128)
    inv = np.zeros((input_z.shape[0], len(connect_port_list), len(connect_port_list)),dtype = np.complex128)
    z_connect = np.zeros((input_z.shape[0],len(a_ports),len(a_ports)),dtype = np.complex128)
    for i in range(len(connect_port_list)):
        Zqq[:,i,i] = DECAP_LIST2[decap_num_list[i]]
    for i in range(input_z.shape[0]):
        inv[i] = np.linalg.inv(Zpp[i]+Zqq[i])
        z_connect[i] = Zaa[i] - np.dot(np.dot(Zap[i],inv[i]),Zpa[i])

    return z_connect



def connect_z_short(z1, map2orig_input, shorted_port):
    a_ports = list(range(z1.shape[1])) 
    del a_ports[shorted_port]
    p_ports = [shorted_port]
    Zaa = z1[np.ix_(a_ports, a_ports)]
    Zpp = z1[np.ix_(p_ports, p_ports)]
    Zap = z1[np.ix_(a_ports, p_ports)]
    Zpa = z1[np.ix_(p_ports, a_ports)]
    z_connect = Zaa - np.matmul(np.matmul(Zap,np.linalg.inv(Zpp)),Zpa)
    map2orig_output = map2orig_input.copy()
    del map2orig_output[shorted_port]
    return z_connect,map2orig_output



def connect_z_short_calc_znn(z1, map2orig_input, shorted_port, observation_port = [0]):
    a_ports = observation_port
    p_ports = [shorted_port]
    Zaa = z1[a_ports][:,a_ports]
    Zpp = z1[p_ports][:,p_ports]
    Zap = z1[a_ports][:,p_ports]
    Zpa = z1[p_ports][:,a_ports]
    Z11 = Zaa[0,0] - Zap[0,0]/Zpp*Zpa[0,0]
    map2orig_output = map2orig_input.copy()
    del map2orig_output[shorted_port]
    return Z11, map2orig_output


def get_priority_of_each_decap(port_priority, varbound, DecapTypeNum):
    port_priority_of_each_decap = []
    for i in range(1,DecapTypeNum+1):
        port_priority_of_each_decap.append([port for port in port_priority if (varbound[port-1][0] <= i <= varbound[port-1][1])]) # port-1 to exclude IC port
    
    sub_priority = []
    temp = 0
    sub_priority.append(port_priority_of_each_decap[0])
    for i in range(varbound.shape[0]):
        if varbound[i][0] == varbound[temp][0] and varbound[i][1] == varbound[temp][1]:
            continue
        else:
            sub_priority.append(port_priority_of_each_decap[varbound[i][0]-1])
            temp = i
            
    return port_priority_of_each_decap,sub_priority


  
# prioritize ports by shorting ports respectively and looking at inductance of Z11. Shorting multiple times
def prioritize_ports_2(z_orig, map2orig_input):
    ic_port = 0
    
    
    compare_ports = list(range(0, z_orig.shape[0]))  # ports that need to be prioritized
    compare_ports.remove(ic_port)


    port_sort = []
    port_map = list(range(0, z_orig.shape[0]))

    for a in tqdm(range(0, len(compare_ports))):
        l11s = []
        for b in compare_ports:
            z11, port_map_tmp = connect_z_short_calc_znn(z1 = z_orig, map2orig_input = port_map, shorted_port = port_map.index(b))
            l11s.append(np.imag(z11))
        
        minl11idx = l11s.index(min(l11s))
        port_sort.append(compare_ports[minl11idx])
        z_orig, port_map = connect_z_short(z1=z_orig, map2orig_input=port_map, shorted_port=port_map.index(compare_ports[minl11idx]))
        del compare_ports[minl11idx]
        

    # map2orig_output = [map2orig_input[i - 1] for i in port_sort]
    map2orig_output = map2orig_input
    return port_sort, map2orig_output


def get_target_z_RL(R, Zmax, fstart=0.01e6, fstop=20e6, nf=201, interp='log'):
    f_transit = fstop * R / Zmax
    if interp == 'log':
        freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
    elif interp == 'linear':
        freq = np.linspace(fstart, fstop, nf)
    ztarget_freq = np.array([fstart, f_transit, fstop])
    ztarget_z = np.array([R, R, Zmax])
    ztarget = np.interp(freq, ztarget_freq, ztarget_z)
    return ztarget

def get_RL_target(freq,R,L):
    return 2*np.pi*freq*L + R




def f(X,state_z = None, map2orig = None):
    dim = len(X)
    port_list = []
    decap_list = []

    for i in range(0, dim):
        if X[i] > 0:
            port_list.append(map2orig.index(i)+1) # +1 to exclude IC port
            decap_list.append(int(X[i]-1)) # -1 since 0 stand for no decap rather than "decap type 0"
            
    z_w_decap = connect_ndecap_together_calc_znn(input_z = state_z, 
                                    connect_port_list = np.array(port_list), 
                                    decap_num_list = np.array(decap_list))
    
    z_solution = np.abs(z_w_decap[:,0,0])
    if np.count_nonzero(np.greater_equal(z_solution, z_target)) == 0:
        reward = z_orig.shape[1] - 1 - len(decap_list)
    else:
        reward = -np.max((z_solution - z_target) / z_target)
    return -reward






def connect_1decap_calc_znn(input_net_z, map2orig_input, connect_port, decap_z11, observation_port = [0]):
    a_ports = observation_port
    p_ports = [connect_port]
    
    Zaa = input_net_z[:,a_ports][:,:,a_ports]
    Zpp = input_net_z[:,p_ports][:,:,p_ports]
    Zap = input_net_z[:,a_ports][:,:,p_ports]
    Zpa = input_net_z[:,p_ports][:,:,a_ports]
    
    Zqq = decap_z11
    Z11 = Zaa[:,0,0] - Zap[:,0,0]/(Zpp+Zqq).reshape(input_net_z.shape[0])*Zpa[:,0,0]
    map2orig_output = map2orig_input.copy()
    del map2orig_output[connect_port]
    return Z11, map2orig_output





def evaluate2(z_state,decap,port,z_target,map2orig):
    z_w_decap,_ = connect_1decap_calc_znn(input_net_z=z_state, map2orig_input=map2orig, connect_port=map2orig.index(port), decap_z11 = DECAP_LIST[decap])
    z_diff = abs(z_w_decap) - z_target
    z_diff[z_diff<0] = 0

    # return max(z_diff)
    return np.sum(z_diff)

def get_initial_solution(z,port_priority_of_each_decap,z_target):
    z_state = copy.deepcopy(z)
    map2orig = list(range(z_orig.shape[1]))
    best_decap_list = []
    best_port_list = []
    lowest_cost = 1e8
    optimal = 0


    
    # -1 to exclude IC port
    for i in tqdm(range(z.shape[1] - 1)):
        best_decap_list.append(-1)
        best_port_list.append(-1)
        improve = 0
        for j in range(len(DECAP_LIST)):
            port = -1
            for k in port_priority_of_each_decap[j]:
                if k not in best_port_list:
                    port = k
                    break
            if port == -1:
                cost = 99999
            else:
                cost = evaluate2(z_state = z_state, decap = j, port = port, z_target = z_target, map2orig = map2orig)
            if (cost < lowest_cost):
                best_decap = j
                best_port = port
                lowest_cost = cost
                improve = 1
            if (cost == 0):
                optimal = 1
                break
        if improve == 0:
            best_decap = j
            best_port = port
        best_decap_list[i] = best_decap
        best_port_list[i] = best_port
        if (optimal == 1):
            break
        # print(lowest_cost)
        
        z_state,map2orig = connect_1decap(input_net_z = z_state, map2orig_input = map2orig, connect_port = map2orig.index(best_port_list[i]), decap_z11 = DECAP_LIST[best_decap_list[i]])
    return best_decap_list, best_port_list





def plot_impedance(z_orig, solution, map2orig_input, z_target, freq):
    dim = len(solution)
    port_list = []
    decap_list = []

    for i in range(0, dim):
        if solution[i] > 0:
            # refer to f(X)
            port_list.append(map2orig_input.index(i)+1)
            # refer to f(X)
            decap_list.append(int(solution[i]-1))
    
    z_w_decap, _ = connect_ndecap_together(input_z=z_orig, map2orig_input=list(range(0,z_orig.shape[1])), connect_port_list=port_list, decap_num_list=decap_list)
    
    
    plt.loglog(freq,z_target,'b')
    plt.loglog(freq,abs(z_w_decap[:,0,0]),'r')
    plt.legend(['target','GA'])
    plt.grid(which='both')
    plt.ylabel('Impedance (Ω)')
    plt.xlabel('Frequency (Hz)')
    plt.tight_layout()
    


if __name__ == '__main__':     
    #######################################################################################
    # read S-parameter or Z-parameter
    ######################################################################################
    # Sfilename = 'your_PDN'
    # net_orig = rf.Network('./data/' + Sfilename)
    # z_orig = net_orig.z
    
    z_orig = np.load('example.npy')
    
    
    #########################################################################################
    # define frequency range of interest and load decaps library
    ########################################################################################
    # frequency = net_orig.frequency
    # freq = net_orig.frequency.f
    
    freq = np.load('example-freq.npy')
    frequency = rf.Frequency.from_f(freq,unit='Hz')
    
    
    # these two variables are exactly the same except the data structure
    DECAP_LIST = init_decap_list(frequency)
    DECAP_LIST2 = np.array([DECAP_LIST[i].reshape((z_orig.shape[0])) for i in range(len(DECAP_LIST))])
    

    
    ####################################################################################
    # define target impedance
    #####################################################################################
    # z_target = get_target_z_RL(R, Zmax, fstart=freq[0], fstop=freq[-1], nf=len(freq), interp='log')
    # z_target = get_custom_target(freq)
    z_target = get_RL_target(freq, R=1e-3, L=0.075e-9)
    
    
    ##########################################################################################
    # hyperparameters
    ########################################################################################
    suspend_strategy = True
    elite_port_ratio = 0.5
    size_variation = 2
    max_num_iteration = 100
    population_size = 50
    mutation_probability = 0.05
    elit_ratio = 0.01
    crossover_probability = 0.5
    parents_portion = 0.3
    crossover_type = 'uniform'
    max_iteration_without_improv = None
    
    
    
    ##################################################################################
    # size constraint
    ################################################################################
    num_ports = z_orig.shape[1] - 1 # -1 to exclude IC port
    varbound = np.array([[1,3]]*24 + [[4,6]]*36 + [[7,10]]*36 + [[11,13]]*24) # define size constraint for each decap type
    # varbound = np.array([[0,13]]*num_ports) # if no size constraint
    
    

    ##########################################################################################
    # port prioritization
    ########################################################################################
    print("sorting the ports………………………………\n")
    sorting_frquency_point = 120
    port_priority, _ = prioritize_ports_2(z_orig = z_orig[sorting_frquency_point,:,:], map2orig_input=list(range(0, z_orig.shape[0])))
    port_priority_of_each_decap, sub_priority = get_priority_of_each_decap(port_priority, varbound, DecapTypeNum = len(DECAP_LIST))
    
    
    ##########################################################################################
    # initial solution determination
    ########################################################################################
    print("getting the initial solution………………………………\n")
    seed_best_decap, seed_best_port = get_initial_solution(z=z_orig, port_priority_of_each_decap=port_priority_of_each_decap, z_target=z_target)
    seed_solution_ga = np.zeros(z_orig.shape[1]-1)
    for i in range(len(seed_best_decap)): 
        # +1 -1 refer to f(X)
        seed_solution_ga[seed_best_port[i]-1] = seed_best_decap[i]+1
    port_priority_ga = [i-1 for i in port_priority]
    
    
    ##########################################################################################
    # GA parameter
    ########################################################################################
    algorithm_param = {'max_num_iteration': max_num_iteration,
                       'population_size': population_size,
                       'mutation_probability': mutation_probability,
                       'elit_ratio': elit_ratio,
                       'crossover_probability': crossover_probability,
                       'parents_portion': parents_portion,
                       'crossover_type': crossover_type,
                       'max_iteration_without_improv': max_iteration_without_improv,
                       'port_priority': port_priority_ga,
                       'sub_priority': sub_priority,
                       'elite_port_ratio':elite_port_ratio, 
                       'z_orig' : z_orig,
                       'suspend_strategy' : suspend_strategy,
                       'size_variation' : size_variation
                       }
    
    
    ##########################################################################################
    # 调用GA模型
    ########################################################################################
    model = ga(function=f,
                dimension=num_ports,
                variable_type='int',
                variable_boundaries=varbound,
                algorithm_parameters=algorithm_param,
                seed_sol= seed_solution_ga
                # seed_sol = None
                )
    
    model.run()



    # plot_impedance(z_orig = z_orig,map2orig_input=list(range(z_orig.shape[1])), solution = model.best_variable, z_target = z_target, freq = freq)