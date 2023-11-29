
'''
Copyright 2020 Ryan (Mohammad) Solgi
Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
'''

###############################################################################
###############################################################################
###############################################################################

import numpy as np
import sys
import time
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt
import copy
import time

###############################################################################
###############################################################################
###############################################################################

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
                 function_timeout=10,
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


        # a limiter to control max # of capacitors in a solution
        self.size = self.dim                 # max # of capacitors allowed
        self.size_variation = 2         # allows for solutions in size - size_variation to exist so solutions actually improve


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

        self.param = algorithm_parameters

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

        # randomly generate population
        for p in range(0, self.pop_s):


            if p == 0 and self.seed_sol is not None:
                var = self.seed_sol.copy()

            else:
                for i in self.integers[0]:
                    var[i] = np.random.randint(self.var_bound[i][0], \
                                               self.var_bound[i][1] + 1)
                    #solo[i] = var[i].copy()
                for i in self.reals[0]:
                    var[i] = self.var_bound[i][0] + np.random.random() * \
                             (self.var_bound[i][1] - self.var_bound[i][0])
                    #solo[i] = var[i].copy()



            # here sim(var) seems to just check if var is an appropriate input argument for input function
            # (if input function can use var to calculate something)
            # also checks if the input argument actually returns something before time funciton time out
            # if it works, and can return stuff, it will return the score I believe.

            ###### Initial population check for pop size -- Jack Jank ######
            var = self.pop_size_control(var, self.size)
            for i in range(np.shape(var)[0]):
                solo[i] = var[i].copy()
            obj = self.sim(var)

            # so it looks like the last entry stores obj
            # everyother entry before it stores the population member gene
            # obj appears to be the output of the input function (objective function) so it is the score
            # so each entry of pop contains [var1,var2,var3,..... the genes the member, score of the member]
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

            # self.progress(t, self.iterate, status="GA is running...")
            #############################################################
            # Sort
            pop = pop[pop[:, self.dim].argsort()]
            # Argsort:
            # 'Returns an array of indices of the same shape as 'array'that index data along the given axis in sorted order'
            # basically argsort sorts the function, but instead of moving the values to its new location, it moves the INDICES
            # to its new position. Returns an array of indices
            #   x = np.array([3, 1, 2])
            #   np.argsort(x)
            # so the sort would be [1,2,3], which if you replace with their original INDEX and not value,
            #   you get [1,2,0]
            # by default i think it sorts least - greatest

            # in this specific case,
            # it looks like it does the argsort based on the output value of the objective function
            # then population reorders itself based on that sort
            # Basically it reorders it so that population is sorted based on scores, lowest to highest

            if pop[0, self.dim] < self.best_function:

                # if best scoring member of the population is better than current best score,
                # overwrite the best score
                # overwrite the best performing member with the new best member
                counter = 0
                self.best_function = pop[0, self.dim].copy()
                self.best_variable = pop[0, : self.dim].copy()

                
                

                # update the largest possible size of the solution
                # based on # of decaps of the best soluiton, if target is met

                # print('Score Improved. Best Score So Far Is', self.best_function)

                if np.count_nonzero(self.best_variable) < self.size and self.best_function < 0:
                    self.size = np.count_nonzero(self.best_variable)
                    print('Number of Capacitors has Decreased. Minimum Decap Number =', self.size)

                # print('Solution is', self.best_variable)
                # print('time used:',nth_time)
            else:
                counter += 1
            print('here', self.size)
            #############################################################

            ###### Initial population check for pop size -- Jack Jank ######
            if t == 0:
                pop = self.pop_size_control(pop, self.size)
                # for i in range(self.pop_s):
                #     print('old', list(old_pop[i]))
                #     print('new', list(pop[i]))
                ######

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
                # copy some elite members over. num of elite members determined by elite ratio
                par[k] = pop[k].copy()
            for k in range(self.num_elit, self.par_s):

                index = np.searchsorted(cumprob, np.random.random())
                # Basically random.random() generates a probability
                # and the searchsorted() takes the probability, and find what index that probability would have,
                # if it was placed into a sorted version of cumprob and without changing order of cum prob

                # ---- np.searchsorted([1, 2, 3, 4, 5], 3)
                # ---- 2
                # look online for other arguments

                # this is basically a roulette search.
                par[k] = pop[index].copy()
                # gets the parent to copy

                # in addition, I believe the parents selected here based on the searchsorted, are also
                # the parents who will survive to the next generation (based on par_s)


            ef_par_list = np.array([False] * self.par_s)
            par_count = 0

            # Basically a loop to decide which of the selected parents will actually crossover
            # Denoted as a True/False array
            while par_count == 0:
                for k in range(0, self.par_s):
                    if np.random.random() <= self.prob_cross:
                        ef_par_list[k] = True
                        par_count += 1

            # If i am not mistaken, par[ef_par_list] will return an array comprising of only those indices
            # where ef_par_list is true.
            # ie par[ef_par_list] will include only the selected parents (with index = true) that will cross over
            # ef_par is a list of parents that will actually breed
            # ELITE parents are included here
            # ef_par is a list of parents that CAN actually breed, not a list of all parents
            ef_par = par[ef_par_list].copy()

            #############################################################
            # New generation
            pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)


            for k in range(0, self.par_s):

                # put in elite parents and portion of surviving parents over to the new population.
                pop[k] = par[k].copy()

            for k in range(self.par_s, self.pop_s, 2):

                r1 = np.random.randint(0, par_count)   # from the parents chosen for breeding, grab one
                r2 = np.random.randint(0, par_count)
                pvar1 = ef_par[r1, : self.dim].copy()  # pretty sure it copies the variables of the parents at r1. Excludes the score
                pvar2 = ef_par[r2, : self.dim].copy()  # same here for second parent

                ch = self.cross(pvar1, pvar2, self.c_type)
                # cross over returns an array of containing two children (go check def cross()


                ch1 = ch[0].copy()
                ch2 = ch[1].copy()

                ch1 = self.mut(ch1)                     # mutate one of the child 1 in the classical way
                ch2 = self.mutmidle(ch2, pvar1, pvar2)  # mutates child 2 in a 'fancy' way
                                                        # tries to keep genes mutated within a range based on parents

                # population control for GA
                # ensure only some # of caps are in a solution
                if self.size != 0:
                    n_caps1 = np.count_nonzero(ch1)
                    n_caps2 = np.count_nonzero(ch2)
                    if n_caps1 < (self.size - self.size_variation) or n_caps1 > self.size:
                        #print('less than')
                        ch1 = self.pop_size_control(ch1, self.size, random_change=False)
                    if n_caps2 < (self.size - self.size_variation) or n_caps2 > self.size:
                        #print('less than')
                        ch2 = self.pop_size_control(ch2, self.size, random_change= False)

                solo[: self.dim] = ch1.copy()   # copy the genes over to solo
                obj = self.sim(ch1)             # calculate score/check if score calculatable
                solo[self.dim] = obj            # store score
                pop[k] = solo.copy()            # copy member of population
                solo[: self.dim] = ch2.copy()   # do the same for the second child
                obj = self.sim(ch2)
                solo[self.dim] = obj
                pop[k + 1] = solo.copy()
            #############################################################

            t += 1

            # if score does not improve within some # of generations
            if counter > self.mniwi:
                pop = pop[pop[:, self.dim].argsort()]
                if pop[0, self.dim] >= self.best_function:
                    t = self.iterate
                    self.progress(t, self.iterate, status="GA is running...")
                    time.sleep(1)
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
        sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        # sys.stdout.flush()
        # re = np.array(self.report)

        # plt.plot(re)
        # plt.xlabel('Iteration')
        # plt.ylabel('Objective function')
        # plt.title('Genetic Algorithm')
        # plt.show()
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

    def mut(self, x):

        for i in self.integers[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                x[i] = np.random.randint(self.var_bound[i][0], \
                                         self.var_bound[i][1] + 1)

        for i in self.reals[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                x[i] = self.var_bound[i][0] + np.random.random() * \
                       (self.var_bound[i][1] - self.var_bound[i][0])

        return x

    ###############################################################################
    def mutmidle(self, x, p1, p2):
        for i in self.integers[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                if p1[i] < p2[i]:
                    x[i] = np.random.randint(p1[i], p2[i])
                elif p1[i] > p2[i]:
                    x[i] = np.random.randint(p2[i], p1[i])
                else:
                    x[i] = np.random.randint(self.var_bound[i][0], \
                                             self.var_bound[i][1] + 1)

        for i in self.reals[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                if p1[i] < p2[i]:
                    x[i] = p1[i] + np.random.random() * (p2[i] - p1[i])
                elif p1[i] > p2[i]:
                    x[i] = p2[i] + np.random.random() * (p1[i] - p2[i])
                else:
                    x[i] = self.var_bound[i][0] + np.random.random() * \
                           (self.var_bound[i][1] - self.var_bound[i][0])
        return x

    ###############################################################################
    def evaluate(self):
        return self.f(self.temp)

    ###############################################################################
    def sim(self, X):
        self.temp = X.copy()
        obj = None

        ## How func_timeout works
        # def func_timeout(timeout, func, args=(), kwargs=None
        # func_timeout runs function 'func' with input arguments 'args' for up to 'timeout' seconds
        # At the end of 'timeout' seconds,
        # it will return/raise anything the passed function would otherwise return or raise
        # if 'timeout' is exceeded before any other return/raise from the function would occur (ie function not done)
        # it will return a time out error (FunctionTimedOut)
        try:
            obj = func_timeout(self.funtimeout, self.evaluate)
            # in this case, timeout is funtimeout
            # the function to be evaluated is the input function generally (objective function, self.f, contextually)
            # where the input argument for the input function is self.temp (a copy of something passed to sim)

            # so it seems like funtimeout is a max limit on how long the objective function is allowed to run

            #if it does not time out, and its evaluateable, it returns the scores
        except FunctionTimedOut:
            print("given function is not applicable")
            # is time is exceeded, the objective function (or just input function) is not appropriate for use?

        assert (obj != None), "After " + str(self.funtimeout) + " seconds delay " + \
                              "func_timeout: the given function does not provide any output"
            # assert:, check if a condition is true, if not, raise an error
            # if the function successfully runs all the way through but does not have a return argument, lets you
            # know you need a return argument.

        # So basically it seems like this function checks if the input argument to the input function won't raise errors
        # Checks if the input function will run in a reasonable amount of time
        # checks if the input function actually has a return argument
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

    ################### Functions Jack Jank Added ###############

    def pop_size_control(self, pop, size, random_change = True):

        # this function will take a population, pop, and limits the size of the population
        # to a particular size.

        # for the PDN capacitor selection, a 0 represents an empty port.
        # we want to control how many capacitors (non-zero) can be in the solution
        # there will be 'size' number of capacitors

        # later may change so size is a range but not rn

        # Also to note, from default code, pop has dimension [0:self.dim+1]
        # 0 to self.dim contains the genes
        # the self.dim + 1 should be the evaluation/score

        # pop is the population object
        # size is an integer from 1 to self.dim



        if not isinstance(size, int):
            raise TypeError('The "Size" of the Solution Must be an Integer In [1, #of Genes]')

        if size < 1 or size > self.dim:
            raise ValueError('Size of a Solution Must Be Between 1 or the # of Genes (# of Ports')



        new_pop = copy.deepcopy(pop)

        if np.ndim(new_pop) == 1: # if you are passing only one solution and not the entire population
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
                num_to_empty = size - current_size
                ports_to_fill = np.random.choice(empty_locations, size=num_to_empty, replace=False)

                if random_change:
                    for j in ports_to_fill:
                        new_cap = np.random.randint(np.min(self.var_bound[0])+1, np.max(self.var_bound[0]) + 1)
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