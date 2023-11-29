import numpy as np
from GA import geneticalgorithm as ga
import os
import skrf as rf
from numba import jit
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def get_target_z_RL(R, Zmax, opt, fstart=0.1e6, fstop=1e8, nf=201, interp='log'):
    f_transit = fstop * R / Zmax
    if interp == 'log':
        freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
    elif interp == 'linear':
        freq = np.linspace(fstart, fstop, nf)
    ztarget_freq = np.array([fstart, f_transit, fstop])
    ztarget_z = np.array([R, R, Zmax])
    ztarget = np.interp(freq, ztarget_freq, ztarget_z)
    return ztarget


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
    





def f(X):
    dim = len(X)
    port_list = []
    decap_list = []
    map2orig_input=list(range(0,z.shape[1]))
    for i in range(0, dim):
        if X[i] > 0:
            port_list.append(map2orig_input.index(i)+1)
            decap_list.append(int(X[i]-1))
    
    z_w_decap = connect_ndecap_together(input_z=z, connect_port_list=np.array(port_list), decap_num_list=np.array(decap_list))
    
    z_solution = np.abs(z_w_decap[:,0,0])
    
    if np.count_nonzero(np.greater_equal(z_solution, z_target)) == 0:
        reward = z.shape[1] - 1 - len(decap_list)
    else:
        reward = -np.max((z_solution - z_target) / z_target)
    return -reward


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


def get_custom_target(freq):
    ztarget = np.full(freq.shape[0],10,dtype=np.float32)
    for i in range(freq.shape[0]):
        if freq[i] <= 1e6:
            ztarget[i] = 0.5e-3
            
    return ztarget


def get_RL_target(freq,R,L):
    return 2*np.pi*freq*L + R



# load PDN S(Z)-parameter
z = np.load('./example.npy')

#Set frequency of interest and init decaps library
fstart = 2e5
fstop = 1e8
nf = 201
frequency = rf.frequency.Frequency(start=fstart, stop=fstop, npoints=nf, unit = 'Hz', sweep_type='log')
freq = frequency.f
DECAP_LIST = init_decap_list(frequency)

# define target impedance
z_target = get_custom_target(freq)

# Set GA up
num_ports = z.shape[1] - 1
varbound = np.array([[0, 13]] * num_ports)
algorithm_param = {'max_num_iteration': 300,
                   'population_size': 50,
                   'mutation_probability': 0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': None}



seed_sol = None
model = ga(function=f,
            dimension=num_ports,
            variable_type='int',
            variable_boundaries=varbound,
            algorithm_parameters=algorithm_param,
            # seed_sol = seed_sol
            )

model.run()