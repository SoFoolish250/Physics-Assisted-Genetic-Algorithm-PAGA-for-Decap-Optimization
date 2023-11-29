from pdn_class import PDN
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import skrf as rf
from copy import deepcopy
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


def evaluate2(z_state,decap,port,z_target,map2orig):
    z_w_decap,_ = connect_1decap2(input_net_z=z_state, map2orig_input=map2orig, connect_port=map2orig.index(port), decap_z11 = brd.decap_list[decap])
    z_solution = np.abs(z_w_decap[:,0,0])
    global_cost = 1
    if np.count_nonzero(np.greater_equal(z_solution, z_target)) == 0:
        global_cost = 0
    return (z_solution[fmax_index] - z_target[fmax_index]), global_cost


def evaluate(z, decap_list, port_list, z_target, fmax_index):
    z_w_decap = connect_ndecap_together(input_z=z, connect_port_list=np.array(port_list), decap_num_list=np.array(decap_list))
    z_solution = np.abs(z_w_decap[:,0,0])
    global_cost = 1
    if np.count_nonzero(np.greater_equal(z_solution, z_target)) == 0:
        global_cost = 0
    return (z_solution[fmax_index] - z_target[fmax_index]), global_cost


def get_custom_target(freq):
    ztarget = np.full(freq.shape[0],100,dtype=np.float32)
    for i in range(freq.shape[0]):
        if freq[i] <= 1e6:
            ztarget[i] = 0.5e-3
    return ztarget



def connect_1decap2(input_net_z, map2orig_input, connect_port, decap_z11):
    aa = list(range(input_net_z.shape[1]))
    del aa[connect_port]
    Zaa = input_net_z[np.ix_(list(range(0,input_net_z.shape[0])),aa,aa)]
    Zpp = input_net_z[:, connect_port, connect_port].reshape((input_net_z.shape[0],1,1))
    Zqq = decap_z11
    Zap = input_net_z[np.ix_(list(range(0,input_net_z.shape[0])),aa,[connect_port])]
    Zpa = input_net_z[np.ix_(list(range(0,input_net_z.shape[0])),[connect_port],aa)]

    
    
    inv = np.linalg.inv(Zpp+Zqq)
    second = np.einsum('rmn,rkk->rmn', Zap, inv)
    second = np.einsum('rmn,rnd->rmd', second, Zpa)
    output_net_z = Zaa - second
    map2orig_output = deepcopy(map2orig_input)
    del map2orig_output[connect_port]
    return output_net_z, map2orig_output

# load PDN S(Z)-parameter
z_orig = np.load('./example.npy')


# set frequency of interest and init decaps library
fstart = 2e5
fstop = 1e8
nf = 201
frequency = rf.frequency.Frequency(start=fstart, stop=fstop, npoints=nf, unit = 'Hz', sweep_type='log')
freq = frequency.f
DECAP_LIST = init_decap_list(frequency)


# define target impedance
z_target = get_custom_target(freq)


brd = PDN(frequency_start=fstart,frequency_stop=fstop)


decap_list = []
port_list = []


optimal = 0
lowest_cost = 100000
min_decap_num = 0


# find resonance frequency of each decap type
resonance_freq = []
for i in range(len(brd.decap_list)):
    res_index = np.argmin(brd.decap_list[i])
    resonance_freq.append(freq[res_index])




opt_index = (freq <= 1e6)
z_state = z_orig.copy()
map2orig = list(range(z_orig.shape[1]))


start = time.time()
for i in range(z_orig.shape[1] - 1):
    """
    step1: find the max-z and its corresponding frequency f_max
    step2: find the decap whose resonance frequency is closest to the f_max
    step3: find the best location of the decap (here we  traverse all locations)
    step5: if z-target is satisfied for all frequency, terminate the algorithm, otherwise, add one more decap and repeate step1-step4
    """
    z_ic = z_state[:,0,0][opt_index]
    fmax_index = np.argmax(abs(z_ic))
    fmax = freq[fmax_index]
    decap_kind = np.argmin(np.abs([i - fmax for i in resonance_freq]))
    decap_list.append(decap_kind)
    
    port_list.append(1)
    lowest_cost = 100000
    
    for loc in range(1,z_orig.shape[1]):
        if loc in port_list:
            continue
        port_list[i] = loc
        cost, global_cost = evaluate(z_orig, decap_list, port_list, z_target, fmax_index)
        # cost, global_cost = evaluate2(z_state = z_state, decap=decap_kind, port=loc, z_target=z_target, map2orig=map2orig)
        
        if cost<lowest_cost:
            port_best = loc
            lowest_cost = cost

    z_state,map2orig = connect_1decap2(input_net_z = z_state, map2orig_input = map2orig, connect_port = map2orig.index(port_best), decap_z11 = brd.decap_list[decap_kind])
    
    
    port_list[i] = port_best

    print("num of decap:",i+1)
    print("decap_list:",decap_list)
    print("port_list:",port_list)
    print("cost:",lowest_cost)
    print("")
    if global_cost == 0:
        break


print("best_decap_list is\n",decap_list)
print("best_port_list is\n",port_list)
print("min_decap_num is:",len(decap_list))
end = time.time()

print(end-start)


z_final, _ = brd.connect_n_decap(input_z=z_orig, map2orig_input=list(range(0,z_orig.shape[1])), connect_port_list=port_list, decap_num_list=decap_list)

plt.loglog(brd.freq.f[:],abs(z_final[:,0,0]),'g-.')
plt.loglog(brd.freq.f[:],z_target,'r--')
plt.grid(which='both')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Impedance(Ohm)')    
