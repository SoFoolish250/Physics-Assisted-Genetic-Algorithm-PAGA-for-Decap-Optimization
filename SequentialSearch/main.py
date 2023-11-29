from pdn_class import PDN,connect_1decap,connect_1decap2
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import time
import skrf as rf
from numba import jit


def get_custom_target(freq):
    ztarget = np.full(freq.shape[0],10,dtype=np.float32)
    # for i in range(freq.shape[0]):
    #     if freq[i] > 500e3 and freq[i] < 2e6:
    #         ztarget[i] = 0.3e-3
    #     elif freq[i] > 2e6 and freq[i] < 3.65e6:
    #         ztarget[i] = 0.45e-3
    for i in range(freq.shape[0]):
        if freq[i] <= 1e6:
            ztarget[i] = 0.5e-3
        # else:
        #     ztarget[i] = 2*np.pi*freq[i]*150e-12
        # elif freq[i] > 1e5 and freq[i] < 6e6:
        #     ztarget[i] = 0.0022
        # elif freq[i] >= 6e6 and freq[i] <= 1e7:
        #     ztarget[i] = 0.004
            
    return ztarget


def get_target_z_RL(R, Zmax, fstart=1e4, fstop=100e6, nf=201, interp='log'):
    f_transit = fstop * R / Zmax
    if interp == 'log':
        freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
    elif interp == 'linear':
        freq = np.linspace(fstart, fstop, nf)
    ztarget_freq = np.array([fstart, f_transit, fstop])
    ztarget_z = np.array([R, R, Zmax])
    ztarget = np.interp(freq, ztarget_freq, ztarget_z)
    return ztarget

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
    return np.array([decap_list[i].reshape(len(freq)) for i in range(len(decap_list))])

@jit(nopython = True)
def connect_ndecap_together(input_z, connect_port_list, decap_num_list):
    a_ports = list(range(input_z.shape[1]))
    a_ports = np.array([i for i in a_ports if i not in connect_port_list])
    p_ports = connect_port_list
    

    Zaa = input_z[:,a_ports][:,:,a_ports]
    Zpp = input_z[:,p_ports][:,:,p_ports]
    Zap = input_z[:,a_ports][:,:,p_ports]
    Zpa = input_z[:,p_ports][:,:,a_ports]
    
    Zqq = np.zeros((input_z.shape[0], len(connect_port_list), len(connect_port_list)),dtype = np.complex128)
    inv = np.zeros((input_z.shape[0], len(connect_port_list), len(connect_port_list)),dtype = np.complex128)
    z_connect = np.zeros((input_z.shape[0],len(a_ports),len(a_ports)),dtype = np.complex128)
    for i in range(len(connect_port_list)):
        Zqq[:,i,i] = DECAP_LIST[decap_num_list[i]]
    for i in range(input_z.shape[0]):
        inv[i] = np.linalg.inv(Zpp[i]+Zqq[i])
        z_connect[i] = Zaa[i] - np.dot(np.dot(Zap[i],inv[i]),Zpa[i])

    return z_connect

def evaluate(z, decap_list, port_list):
    z_w_decap = connect_ndecap_together(input_z=z,connect_port_list=np.array(port_list), decap_num_list=np.array(decap_list))
    z_solution = np.abs(z_w_decap[:,0,0])
    z_diff = z_solution - z_target
    z_diff[z_diff<0] = 0
    return np.sum(z_diff)/z_orig.shape[0]

def evaluate2(z_state,decap,port,z_target,map2orig):
    z_w_decap,_ = connect_1decap2(input_net_z=z_state, map2orig_input=map2orig, connect_port=map2orig.index(port), decap_z11 = brd.decap_list[decap])
    z_solution = np.abs(z_w_decap[:,0,0])
    z_diff = z_solution - z_target
    z_diff[z_diff<0] = 0
    return np.sum(z_diff)/z_orig.shape[0]


# load PDN S(Z)-parameter
z_orig = np.load('./example.npy')
brd = PDN(frequency_start=2e5,frequency_stop=1e8)




# set frequency of interest and init decaps library
fstart = 2e5
fstop = 1e8
nf = 201
frequency = rf.frequency.Frequency(start=fstart, stop=fstop, npoints=nf, unit = 'Hz', sweep_type='log')
freq = frequency.f
DECAP_LIST = init_decap_list(frequency)

# define target impedance
z_target = get_custom_target(freq)


decap_list = []
port_list = []
optimal = 0
lowest_cost = z_orig.shape[0]+1
best_decap_list = []
best_port_list = []
min_decap_num = 0
cost_list = []


map2orig = list(range(z_orig.shape[1]))
z_state = z_orig.copy()
# i: the num of decap
# j: consider all kinds of decap
# k: consider all locations

t1 = time.time()
for i in range(z_orig.shape[1] - 1):
    decap_list.append(0)
    port_list.append(1)
    improve = 0
    
    for j in range(len(brd.decap_list)):
        decap_list[i] = j
        
        for k in range(1,z_orig.shape[1]):
            if k in port_list:
                continue
            port_list[i] = k
            
            cost = evaluate(z_orig, decap_list, port_list)
            # cost = evaluate2(z_state = z_state, decap=j, port=k, z_target=z_target, map2orig=map2orig)
            
            
            if (cost < lowest_cost):
                best_decap_list = decap_list.copy()
                best_port_list = port_list.copy()
                lowest_cost = cost
                min_decap_num = i+1
                improve = 1
            cost_list.append(lowest_cost)
            
            if (cost == 0):
                optimal = 1
                break
             
        print("num of decap:",i+1)
        print("decap_list:",best_decap_list)
        print("port_list:",best_port_list)
        print("cost:",lowest_cost)
        print("")
    
    if (improve == 0):
        print("no improvement after adding one more decap")
        break
    z_state,map2orig = connect_1decap2(input_net_z = z_state, map2orig_input = map2orig, connect_port = map2orig.index(best_port_list[i]), decap_z11 = brd.decap_list[best_decap_list[i]])
    decap_list[i] = best_decap_list[i]
    port_list[i] = best_port_list[i]
    
    
    z, _ = brd.connect_n_decap(input_z=z_orig, map2orig_input=list(range(0,z_orig.shape[1])), connect_port_list=port_list, decap_num_list=decap_list)
    test = abs(z[:,0,0])
    # plt.plot(z_target)
    # plt.plot(test)
    
    
    if (optimal == 1):
        break
    
t2 = time.time()

print("best_decap_list is\n",best_decap_list)
print("best_port_list is\n",best_port_list)
print("min_decap_num is:",min_decap_num)

plt.loglog(brd.freq.f,z_target)
plt.loglog(brd.freq.f,test)

print(t2-t1)
