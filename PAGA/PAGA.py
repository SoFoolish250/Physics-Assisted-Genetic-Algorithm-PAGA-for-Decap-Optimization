import numpy as np
import sys
import time
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed


class geneticalgorithm():
    '''  Genetic Algorithm (Elitist version) for Python

    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.



    Implementation and output:

        methods:
                run(): implements the genetic algorithm

        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }

                report: a list including the record of the progress of the
                algorithm over iterations
    '''

    #############################################################
    def __init__(self, function, dimension, variable_type='bool',
                 variable_boundaries=None,
                 variable_type_mixed=None,
                 function_timeout=100,
                 algorithm_parameters={'max_num_iteration': None, \
                                       'population_size': 100, \
                                       'mutation_probability': 0.1, \
                                       'elit_ratio': 0.01, \
                                       'crossover_probability': 0.5, \
                                       'parents_portion': 0.3, \
                                       'crossover_type': 'uniform', \
                                       'max_iteration_without_improv': None},
                 seed_sol = None):

        '''
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function.
        (For maximization multiply function by a negative sign: the absolute
        value of the output would be the actual objective function)

        @param dimension <integer> - the number of decision variables

        @param variable_type <string> - 'bool' if all variables are Boolean;
        'int' if all variables are integer; and 'real' if all variables are
        real value or continuous (for mixed type see @param variable_type_mixed)

        @param variable_boundaries <numpy array/None> - Default None; leave it
        None if variable_type is 'bool'; otherwise provide an array of tuples
        of length two as boundaries for each variable;
        the length of the array must be equal dimension. For example,
        np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first
        and upper boundary 200 for second variable where dimension is 2.

        @param variable_type_mixed <numpy array/None> - Default None; leave it
        None if all variables have the same type; otherwise this can be used to
        specify the type of each variable separately. For example if the first
        variable is integer but the second one is real the input is:
        np.array(['int'],['real']). NOTE: it does not accept 'bool'. If variable
        type is Boolean use 'int' and provide a boundary as [0,1]
        in variable_boundaries. Also if variable_type_mixed is applied,
        variable_boundaries has to be defined.

        @param function_timeout <float> - if the given function does not provide
        output before function_timeout (unit is seconds) the algorithm raise error.
        For example, when there is an infinite loop in the given function.

        @param algorithm_parameters:
            @ max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int>
            @ mutation_probability <float in [0,1]>
            @ elit_ration <float in [0,1]>
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ crossover_type <string> - Default is 'uniform'; 'one_point' or
            'two_point' are other options
            @ max_iteration_without_improv <int> - maximum number of
            successive iterations without improvement. If None it is ineffective

        for more details and examples of implementation please visit:
            https://github.com/rmsolgi/geneticalgorithm

        '''
        self.__name__ = geneticalgorithm
        #############################################################
        # input function
        assert (callable(function)), "function must be callable"

        self.f = function
        #############################################################
        # dimension

        self.dim = int(dimension)
        self.param = algorithm_parameters


        # a limiter to control max # of capacitors in a solution
        self.size = self.dim                 # max # of capacitors allowed
        self.size_variation = self.param['size_variation']         # allows for solutions in size - size_variation to exist so solutions actually improve


        # seed sol variable for giving an initial solution
        if seed_sol is not None:
            self.seed_sol = seed_sol.copy()

        else:
            self.seed_sol = None

        #############################################################
        # input variable type

        assert (variable_type == 'bool' or variable_type == 'int' or \
                variable_type == 'real'), \
            "\n variable_type must be 'bool', 'int', or 'real'"
        #############################################################
        # input variables' type (MIXED)

        if variable_type_mixed is None:

            if variable_type == 'real':
                self.var_type = np.array([['real']] * self.dim)
            else:
                self.var_type = np.array([['int']] * self.dim)


        else:
            assert (type(variable_type_mixed).__module__ == 'numpy'), \
                "\n variable_type must be numpy array"
            assert (len(variable_type_mixed) == self.dim), \
                "\n variable_type must have a length equal dimension."

            for i in variable_type_mixed:
                assert (i == 'real' or i == 'int'), \
                    "\n variable_type_mixed is either 'int' or 'real' " + \
                    "ex:['int','real','real']" + \
                    "\n for 'boolean' use 'int' and specify boundary as [0,1]"

            self.var_type = variable_type_mixed
        #############################################################
        # input variables' boundaries

        if variable_type != 'bool' or type(variable_type_mixed).__module__ == 'numpy':

            assert (type(variable_boundaries).__module__ == 'numpy'), \
                "\n variable_boundaries must be numpy array"

            assert (len(variable_boundaries) == self.dim), \
                "\n variable_boundaries must have a length equal dimension"

            for i in variable_boundaries:
                assert (len(i) == 2), \
                    "\n boundary for each variable must be a tuple of length two."
                assert (i[0] <= i[1]), \
                    "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound = variable_boundaries
        else:
            self.var_bound = np.array([[0, 1]] * self.dim)

        #############################################################
        # Timeout
        self.funtimeout = float(function_timeout)

        #############################################################
        # input algorithm's parameters

        

        self.pop_s = int(self.param['population_size'])

        assert (self.param['parents_portion'] <= 1 \
                and self.param['parents_portion'] >= 0), \
            "parents_portion must be in range [0,1]"

        self.par_s = int(self.param['parents_portion'] * self.pop_s)
        trl = self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s += 1

        self.prob_mut = self.param['mutation_probability']

        assert (self.prob_mut <= 1 and self.prob_mut >= 0), \
            "mutation_probability must be in range [0,1]"

        self.prob_cross = self.param['crossover_probability']
        assert (self.prob_cross <= 1 and self.prob_cross >= 0), \
            "mutation_probability must be in range [0,1]"

        assert (self.param['elit_ratio'] <= 1 and self.param['elit_ratio'] >= 0), \
            "elit_ratio must be in range [0,1]"

        trl = self.pop_s * self.param['elit_ratio']
        if trl < 1 and self.param['elit_ratio'] > 0:
            self.num_elit = 1
        else:
            self.num_elit = int(trl)

        assert (self.par_s >= self.num_elit), \
            "\n number of parents must be greater than number of elits"

        if self.param['max_num_iteration'] == None:
            self.iterate = 0
            for i in range(0, self.dim):
                if self.var_type[i] == 'int':
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) * self.dim * (100 / self.pop_s)
                else:
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) * 50 * (100 / self.pop_s)
            self.iterate = int(self.iterate)
            if (self.iterate * self.pop_s) > 10000000:
                self.iterate = 10000000 / self.pop_s
        else:
            self.iterate = int(self.param['max_num_iteration'])

        self.c_type = self.param['crossover_type']
        assert (self.c_type == 'uniform' or self.c_type == 'one_point' or \
                self.c_type == 'two_point'), \
            "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"

        self.stop_mniwi = False
        if self.param['max_iteration_without_improv'] == None:
            self.mniwi = self.iterate + 1
        else:
            self.mniwi = int(self.param['max_iteration_without_improv'])
        
        self.elite_port_ratio = self.param['elite_port_ratio']
        self.port_priority = self.param['port_priority']
        self.sub_priority = self.param['sub_priority']
        self.suspended = 0
        self.z_orig = self.param['z_orig'].copy()
        self.state_z = self.param['z_orig'].copy()
        self.map_to_orig = list(range(self.dim))
        self.suspend_strategy = self.param['suspend_strategy']
        self.sub_decap_num = np.zeros(len(self.sub_priority),dtype = int)


        #############################################################

    def run(self):

        data_saved = np.zeros((self.iterate,3))
        start_time = time.time()



        
        #############################################################
        # Initial Population

        self.integers = np.where(self.var_type == 'int')
        self.reals = np.where(self.var_type == 'real')

        #initialize population
        pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)

        # stores each solutio
        solo = np.zeros(self.dim + 1)

        #Each var is a solution
        var = np.zeros(self.dim)
        

        
        
        if self.seed_sol is not None:

            print("Initial Solution Accepted")
            print("Initial Solution is", self.seed_sol)


            obj = self.sim(self.seed_sol)
            self.best_function = obj
            self.best_variable = self.seed_sol.copy()
            self.size = np.count_nonzero(self.best_variable)

            print("Score given as", self.best_function)
            print("# of Capacitors Per Solution Starting at", self.size)
            if self.suspend_strategy == True:
                self.initial_parameters_size_control()
            
        # generate initial population
        for p in range(0, self.pop_s):
            if self.seed_sol is not None:
                if p ==0:                               #1% copy seed solution
                    var = self.seed_sol.copy()
                else:
                    var = self.seed_sol.copy()
                    # var = self.mutate2(var)
                    var = self.mod_mutate(var)

            else:
                for i in self.integers[0]:
                    var[i] = np.random.randint(self.var_bound[i][0], \
                                               self.var_bound[i][1] + 2)
                    if var[i] == self.var_bound[i][1] + 1:
                        var[i] = 0
                    #solo[i] = var[i].copy()


            for i in range(np.shape(var)[0]):
                solo[i] = int(var[i])
            obj = self.sim(var)
            solo[self.dim] = obj
            pop[p] = solo.copy()

        #############################################################


        #############################################################
        # Report
        self.report = []
        self.test_obj = obj
        self.best_variable = var.copy()
        self.best_function = obj
        ##############################################################

        t = 0
        counter = 0
        while t < self.iterate:

            self.progress(t, self.iterate, status="GA is running...")
            #############################################################
            # Sort pop according to F(x)
            pop = pop[pop[:, self.dim].argsort()]

            if pop[0, self.dim] < self.best_function:
                counter = 0
                self.best_function = pop[0, self.dim].copy()
                self.best_variable = pop[0, : self.dim].copy()
                    
                if self.suspend_strategy == True and self.best_function<0:
                    self.parameters_size_control(self.best_variable)
                    pop = self.pop_suspend_control(pop)

                if np.count_nonzero(self.best_variable) < self.size and self.best_function < 0:
                    self.size = np.count_nonzero(self.best_variable)
                    print('Number of Capacitors has Decreased. Minimum Decap Number =', self.size)
            else:
                counter += 1
            print('here', self.size)
            #############################################################



            # Report

            self.report.append(pop[0, self.dim])  # the best score of each generation

            ##############################################################
            # Normalizing objective function

            normobj = np.zeros(self.pop_s)

            minobj = pop[0, self.dim]
            if minobj < 0: # would be < 0 if your goal is to maximize (check the objective function section)
                normobj = pop[:, self.dim] + abs(minobj)

            else:
                normobj = pop[:, self.dim].copy()

            maxnorm = np.amax(normobj)
            normobj = maxnorm - normobj + 1  # the smallest value would still be the smallest

            #############################################################
            # Calculate probability

            sum_normobj = np.sum(normobj)
            prob = np.zeros(self.pop_s)
            prob = normobj / sum_normobj
            cumprob = np.cumsum(prob)

            #############################################################
            # Select parents
            par = np.array([np.zeros(self.dim + 1)] * self.par_s)

            for k in range(0, self.num_elit):
                par[k] = pop[k].copy()
                
            for k in range(self.num_elit, self.par_s):
                index = np.searchsorted(cumprob, np.random.random())
                par[k] = pop[index].copy()

            ef_par_list = np.array([False] * self.par_s)
            par_count = 0

            while par_count == 0:
                for k in range(0, self.par_s):
                    if np.random.random() <= self.prob_cross:
                        ef_par_list[k] = True
                        par_count += 1

            ef_par = par[ef_par_list].copy()

            #############################################################
            # New generation
            pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)


            for k in range(0, self.par_s):
                pop[k] = par[k].copy()

            for k in range(self.par_s, self.pop_s, 2):

                r1 = np.random.randint(0, par_count)
                r2 = np.random.randint(0, par_count)
                pvar1 = ef_par[r1, : self.dim].copy() 
                pvar2 = ef_par[r2, : self.dim].copy() 
                ch = self.cross(pvar1, pvar2, self.c_type)

                ch1 = ch[0].copy()
                ch2 = ch[1].copy()

                # ch1 = self.mutate2(ch1)
                # ch2 = self.mutate2(ch2)  
                ch1 = self.mod_mutate(ch1)
                ch2 = self.mod_mutate(ch2)
               


                if self.size != 0:
                    n_caps1 = np.count_nonzero(ch1)
                    n_caps2 = np.count_nonzero(ch2)
                    if n_caps1 < (self.size - self.size_variation) or n_caps1 > self.size:
                        ch1 = self.pop_size_control(ch1, self.size, random_change=True)
                    if n_caps2 < (self.size - self.size_variation) or n_caps2 > self.size:
                        ch2 = self.pop_size_control(ch2, self.size, random_change= True)




                # ######################################### 
                solo[: self.dim] = ch1.copy()   # copy the genes over to solo
                obj = self.sim(ch1)             # calculate score/check if score calculatable
                solo[self.dim] = obj            # store score
                pop[k] = solo.copy()            # copy member of population
                solo[: self.dim] = ch2.copy()   # do the same for the second child
                obj = self.sim(ch2)
                solo[self.dim] = obj
                pop[k + 1] = solo.copy()
                # #############################################
                

            #############################################################

            t += 1

            # if score does not improve within some # of generations
            if counter > self.mniwi:
                pop = pop[pop[:, self.dim].argsort()]
                if pop[0, self.dim] >= self.best_function:
                    t = self.iterate
                    self.progress(t, self.iterate, status="GA is running...")
                    time.sleep(2)
                    t += 1
                    self.stop_mniwi = True
            
            data_saved[t-1][0] = t
            data_saved[t-1][1] = self.size
            data_saved[t-1][2] = time.time() -start_time
        #############################################################
        # Sort
        pop = pop[pop[:, self.dim].argsort()]

        if pop[0, self.dim] < self.best_function:
            self.best_function = pop[0, self.dim].copy()
            self.best_variable = pop[0, : self.dim].copy()
        #############################################################
        # Report

        self.report.append(pop[0, self.dim])

        self.output_dict = {'variable': self.best_variable, 'function': \
            self.best_function}
        show = ' ' * 100
        sys.stdout.write('\r%s' % (show))

        if self.stop_mniwi == True:
            sys.stdout.write('\nWarning: GA is terminated due to the' + \
                             ' maximum number of iterations without improvement was met!')
        
        
        
        finish_time = time.strftime("%Y%m%d-%H%M%S")
        hm = finish_time[9:13]
        np.save("./result/solutions"+hm+".npy",self.best_variable)
        np.save("./result/data_saved"+hm+".npy",data_saved)

    ##############################################################################
    ##############################################################################
    def cross(self, x, y, c_type):

        ofs1 = x.copy()
        ofs2 = y.copy()

        if c_type == 'one_point':
            ran = np.random.randint(0, self.dim)
            for i in range(0, ran):
                ofs1[i] = y[i].copy()
                ofs2[i] = x[i].copy()

                # picks cross over point
                # copies makes two children first, with the single point crossover applied to each

        if c_type == 'two_point':

            ran1 = np.random.randint(0, self.dim)
            ran2 = np.random.randint(ran1, self.dim)

            for i in range(ran1, ran2):
                ofs1[i] = y[i].copy()
                ofs2[i] = x[i].copy()

                # picks cross over points
                # copies makes two children first, with the two point crossover applied to each

        if c_type == 'uniform':

            for i in range(0, self.dim):
                ran = np.random.random()
                if ran < 0.5:
                    ofs1[i] = y[i].copy()
                    ofs2[i] = x[i].copy()

                    # Uniformly picks crossover points
                    # copies makes two children first, with the uniform crossover applied to each

        return np.array([ofs1, ofs2])

    ###############################################################################






    ###############################################################################
    def evaluate(self):
        return self.f(self.temp, state_z = self.state_z, map2orig = self.map_to_orig)

    ###############################################################################
    def sim(self, X):
        self.temp = X.copy()
        obj = None

        try: 
            # obj = func_timeout(self.funtimeout, self.evaluate)
            obj = self.evaluate()
        except FunctionTimedOut:
            print("given function is not applicable")

        assert (obj != None), "After " + str(self.funtimeout) + " seconds delay " + \
                              "func_timeout: the given function does not provide any output"
        return obj

    ###############################################################################
    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()
    ###############################################################################

###############################################################################

    ################### Size Variation control ###############

    def pop_size_control(self, pop, size, random_change = True):

        # new_pop = copy.deepcopy(pop)
        new_pop = pop.copy()

        if np.ndim(new_pop) == 1:
            new_pop = np.array([new_pop])

        for i in range(np.shape(new_pop)[0]):
            new_chrome = new_pop[i][0:self.dim].copy()              # copies the genes
            current_size = np.count_nonzero(new_chrome)             # number of caps in solutions

            caps_locations = np.nonzero(new_chrome)[0]              # gets locations where ports are not empty
            caps = new_chrome[caps_locations]                       # gets capacitor at non-empty ports


            if current_size > size:
                num_to_empty = current_size - size
                ports_to_empty = np.random.choice(caps_locations, size= num_to_empty,replace=False)
                for j in ports_to_empty:
                    new_chrome[j] = 0
                new_pop[i][0:self.dim] = new_chrome.copy()


            elif current_size < size:
                empty_locations = np.nonzero(new_chrome == 0)[0]  # gets empty port locations
                num_to_fill = size - current_size
                # num_to_fill = size - current_size - 1
                # ports_can_be_fill = list(set(empty_locations) & set(self.port_priority[0:(self.dim - self.suspended)]))
                ports_can_be_fill = list(set(empty_locations) & set(self.map_to_orig))
                ports_to_fill = np.random.choice(ports_can_be_fill, size=num_to_fill, replace=False)
                if random_change:
                    for j in ports_to_fill:
                        new_cap = np.random.randint(np.min(self.var_bound[j]), np.max(self.var_bound[j]) + 1)
                        new_chrome[j] = new_cap
                    new_pop[i][0:self.dim] = new_chrome.copy()
                else:
                    # replace with a cap already in solution rather than randomly out of the entire range
                    for j in ports_to_fill:
                        new_cap = caps[np.random.randint(0,current_size)]
                        new_chrome[j] = new_cap
                    new_pop[i][0:self.dim] = new_chrome.copy()    
        if np.shape(new_pop)[0] == 1:
            new_pop = new_pop[0]
            
        return new_pop
    
    def mutate(self,x):
        for k in range(self.dim):
            if k in self.map_to_orig:
                ran = np.random.random()
                if ran < self.prob_mut:
                    randint = np.random.randint(self.var_bound[k][0], \
                                             self.var_bound[k][1] + 3)
                    if randint >= self.var_bound[k][1] + 1:
                        randint = 0
                    x[k] = randint
        return x
    

    # modified mutation, unremoved ports only
    def mod_mutate(self,x):
        for i in range(len(self.sub_priority)):
            for j in range(len(self.sub_priority[i])):

                if j <= int(self.elite_port_ratio*self.sub_decap_num[i]):
                    ran = np.random.random()
                    if ran < self.prob_mut:
                        port = self.sub_priority[i][j] - 1
                        x[port] = np.random.randint(self.var_bound[port][0],
                                                    self.var_bound[port][1] + 1)
                elif self.sub_priority[i][j]-1 in self.map_to_orig:
                    ran = np.random.random()
                    if ran < self.prob_mut:
                        port = self.sub_priority[i][j] - 1
                        randint = np.random.randint(self.var_bound[port][0],
                                                    self.var_bound[port][1] + 3)
                        if randint >= self.var_bound[port][1] + 1:
                            randint = 0
                        x[port] = randint
                else:
                    break
        return x

    
    # port removal, suspended ports stand for removed ports
    def parameters_size_control(self,best_variable):
        self.suspended = int(self.dim - np.count_nonzero(best_variable))
        
        sub_decap_num = np.zeros(len(self.sub_priority),dtype = int)
        for i in range(len(self.sub_priority)):
            for j in range(len(self.sub_priority[i])):
                if best_variable[self.sub_priority[i][j]-1] != 0:
                    sub_decap_num[i] += 1
        self.sub_decap_num = sub_decap_num
        
        map_temp = list(range(self.dim))
        for i in range(len(self.sub_priority)):
            for port in self.sub_priority[i][self.sub_decap_num[i]:]:
                del map_temp[map_temp.index(port-1)]
        index = [0] + [i+1 for i in map_temp]
        self.state_z = self.param['z_orig'].copy()[np.ix_(list(range(0,self.z_orig.shape[0])),index,index)] 
        self.map_to_orig = map_temp

    
    
    # def parameters_size_control2(self,best_variable):
    #     self.map_to_orig = [i for i in range(self.dim) if best_variable[i] != 0]
    #     index = [0] + [i+1 for i in self.map_to_orig] #这里0代表IC端口
    #     self.state_z = self.param['z_orig'].copy()[np.ix_(list(range(0,self.z_orig.shape[0])),index,index)]
    
    
    # initial port removal according to initial solution
    def initial_parameters_size_control(self):        
        self.suspended = self.dim - np.count_nonzero(self.seed_sol)
        self.map_to_orig = [i for i in range(self.dim) if self.seed_sol[i] != 0]
        index = [0] + [i+1 for i in self.map_to_orig] # port 0 stand for IC port
        self.state_z = self.param['z_orig'].copy()[np.ix_(list(range(0,self.z_orig.shape[0])),index,index)]
        
        sub_decap_num = np.zeros(len(self.sub_priority),dtype = int)
        for i in range(len(self.sub_priority)):
            for j in range(len(self.sub_priority[i])):
                if self.seed_sol[self.sub_priority[i][j]-1] != 0:
                    sub_decap_num[i] += 1
        self.sub_decap_num = sub_decap_num
    
    
        # set variables corresponding to removed ports to 0 
    def pop_suspend_control(self,pop):
        for i in range(self.dim):
            if i not in self.map_to_orig:
                pop[:,i].fill(0)
        return pop