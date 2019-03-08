from collections import defaultdict

import numpy as np
import scipy as sp

#from pymake.frontend.frontend import Object

from .utils import *
from .math import *

import logging
lgg = logging.getLogger('root')

# Optimize Algothim (cython, etc) ?
# frontend -- data copy -- integration ?

class Algo(object):
    def __init__(self, *args, **kwargs):
        pass

    def get_clusters(self, state=dict(), true_order=True):
        """ return cluster of each element in the original order:
            pi: the linear partition of clusters
            labels: the labels of the original orders
            C: number of clusters
        """
        pi = state.get('pi', self.pi)

        C = len(pi)
        # ordered elements
        clusters = np.asarray(sum([ [i] * len(pi[i]) for i in np.arange(C) ], []))

        if true_order is True:
            # Original order
            return clusters[self.labels]
        else:
            return clusters

    def partition(self, state=dict()):
        clusters = self.get_clusters(state)
        K = self.K
        return dict(zip(*[np.arange(K), clusters]))

try:
    import community as pylouvain
except ImportError as e:
    print('Import Error: %s' % e)

class Louvain(Algo):
    # Ugh : structure ?
    def __init__(self, data, iterations=100, **kwargs):
        super(Louvain, self).__init__(**kwargs)
        #nxG do a copy of data ?
        self.data = data
        #self.data = data.copy()

    def search(self):
        g = nxG(self.data)
        self._partition = pylouvain.best_partition(g, resolution=1)
        return list(self._partition.values())

    def partition(self):
        return self._partition

    def _get_clusters(self):
        return list(self._partition.values())

    @staticmethod
    def get_clusters(data, resolution=0.5):
        g = nxG(data)
        _partition = pylouvain.best_partition(g, resolution=resolution)
        _c = list(_partition.values())
        c_i = list(np.unique(_c))
        clusters = [c_i.index(v) for v in _c]
        return clusters


#import warnings
#warnings.filterwarnings('error')

from copy import deepcopy
#from pymake.frontend.frontendnetwork import frontendNetwork

class Annealing(Algo):
#class Annealing(Algo, frontendNetwork):
    """ Find near optimal partitionning of a square matrix (0,1),
        by using a modularity that maximize communities detection.
        This is a kind of Simulated Annealing (SA) that maximize
        the so called energy (instead of minimizing in SA, get crazy ;)
            * maximize inner-clusters energy I
            * minimize inter-cluster energy O
            * Modularity = O - I

        @TODO: Local Optima escape:
            * multinomial instead of argmax ?
            * convex ?

        parameters
        ---------
        data: np.array (K,K)
            A square matrix
        C_init: int
            Number of clusters at start
        iterations: int
            Iterations for boundary search
        grow_rate : int
            Number of clusters to add a each super-iteration
            if 0, just one super-iteration.
    """

    def __init__(self, data, iterations=200, C_init=2, grow_rate=1):
        super(Annealing, self).__init__(data=data)
        self.data = data.copy()
        self.grow_rate = grow_rate
        # Number of initial classes
        self.K = data.shape[0]
        # Keep track of class labels
        self.labels = np.arange(self.K)

        # Number of clusters
        C = C_init

        ### Dichotomy init:
        # Clusters Boundary
        # Store for speed purpose
        self.B = np.linspace(0,self.K, C+1).astype(int)
        # Cluters partitionning
        self.pi = self.get_partitions(self.B)
        # Total Energy
        self.E = self.energy()
        # degree
        self.degree = np.asarray([data[k,:].sum() + data[:,k].sum() - data[k,k] for k in range(self.K)])

        # not used...
        self.iterations = iterations

    def set_state(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def get_state(self):
        state = {'pi': self.pi,
                 'B': self.B,
                 'data': self.data,
                 'labels': self.labels}
        return state

    def get_C(self):
        # or len(pi)
        return len(self.B)-1

    def get_partitions(self, B):
        """ Clusters Partitionning in a linear label """
        C = len(B) - 1
        return [np.arange(B[i], B[i+1]) for i in range(C)]

    def modularity(self, state=dict()):
        g = self.getG()
        part = self.partition(state)
        modul = pylouvain.modularity(part, g)
        return modul

    #def energy(self, state=dict(), get_params=False): return self.modularity(state)
    def energy(self, state=dict(), get_params=False):
        data = state.get('data', self.data)
        B = state.get('B', self.B)
        pi = state.get('pi', self.get_partitions(B))
        K = self.K
        C = len(pi)

        # Inner Energy
        #blocks = [ np.ix_(*[slice]*2) for slice in pi ]
        diag_blocks = [data[np.ix_(*[slice]*2)] for slice in pi]
        I, L_i = np.asarray(list(zip(* [(a.sum(), float(a.size)) for a in diag_blocks])))

        # Outer Energy
        O = []
        for c in range(C):
            b_low = B[c]
            b_high = B[c+1]
            O.append( data[b_low:b_high, :].sum() + data[:, b_low:b_high].sum() - 2 * I[c])
        L_o = 2 * (K * np.asarray(list(map(len, pi))).astype(float) - L_i)

        if get_params is True:
            return I, L_i, np.asarray(O), L_o

        # Total Energy
        E = 1/float(C) * ((I/L_i).sum() - np.square((O/L_o).sum()))
        #return 1/(1+np.exp(-E))
        #regul =  1 / (np.array(len(pi)).astype(float).sum())**2
        regul = 0
        return E + regul

    def boundary_sample(self, new=0, it=None):
        """ Sample new boundary for clusters
            return new Energy.
        """
        K = self.K
        C = self.get_C() + new

        if it == 0:
            B_new = np.linspace(0,self.K, C+1).astype(int)
        else:
            # Min separation of boundary set to 2.
            B_new = np.sort(np.random.choice(np.arange(2, K-1, 2), C-1, replace=False))
            B_new = np.hstack((0, B_new, K))

        # reload partitioning
        pi_new = self.get_partitions(B_new)

        state = {'B': B_new, 'pi':pi_new}
        return state

    def concentrate_clases(self, state=dict()):
        """ Reorder data by Assignin classes to cluster
            that maximise energy. (descending order).

            Classe will be moving the clusters of choice
            by increasing its size (B updates)
            * Inplace changes.
        """
        data = state.get('data', self.data)
        B = state.get('B', self.B)
        pi = state.get('pi', self.pi)

        K = self.K
        C = len(pi)

        size = np.asarray(list(map(len, pi)))
        clusters = self.get_clusters(state, true_order=False, )

        e_k_in_c = np.zeros((K, C))
        cursor = np.arange(K)
        I, L_i, O, L_o = self.energy(get_params=True)
        for k in np.random.choice(range(K), K, replace=False):
            _k = np.argmax(cursor == k)
            old_c = clusters[k]
            old_i_c = data[_k, B[old_c]:B[old_c+1]].sum() + data[B[old_c]:B[old_c+1], _k].sum() - data[_k,_k]
            degree_k = self.degree[self.labels[k]]
            #E_old = self.energy()
            params_E_tmp = np.zeros((C,4,2))
            for c in np.arange(C):
                #if c == old_c:
                #    #e_k_in_c[k, old_c] = 0
                #    continue
                b_low = B[c]
                b_high = B[c+1]

                ## True update (local optima !)
                #i_c = data[_k, b_low:b_high].sum() + data[b_low:b_high, _k].sum() + data[_k,_k]
                #o_c = degree_k - 2*i_c + data[_k,_k]
                #old_o_c = degree_k - 2*old_i_c + data[_k,_k]
                ##print """ c= %d (true %d), degree_k = %d
                ##        i_c = %d, old_i_c = %d
                ##        o_c = %d, old_o_c = %d
                ##        """ % (c, old_c, degree_k, i_c, old_i_c, o_c, old_o_c)

                #_I   = I[[c  , old_c]] + [i_c                  , - old_i_c]
                #_O   = O[[c  , old_c]] + [o_c                  , - old_o_c]
                #_L_i = L_i[[c, old_c]] + [2*size[c]+1          , - (2*size[old_c] - 1)]
                #_L_o = L_o[[c, old_c]] + [2*(K - 2*size[c] - 1), - 2*(K - 2*size[old_c] + 1)]
                #params_E_tmp[c, 0] = _I
                #params_E_tmp[c, 1] = _L_i
                #params_E_tmp[c, 2] = _O
                #params_E_tmp[c, 3] = _L_o
                ## Delta Energy
                #pt = [c, old_c]
                #e_k_in_c[k, c] = 1/float(C) * ((_I/_L_i).sum() - np.square((_O/_L_o).sum()) - ((I[pt]/L_i[pt]).sum() - np.square((O[pt]/L_o[pt]).sum()) ))

                # Approx update
                sgn = [-1, 1]
                inc = bool(c != old_c)
                i_c = data[_k, b_low:b_high].sum() + data[b_low:b_high, _k].sum() + sgn[inc] *  data[_k,_k]
                o_c = degree_k - i_c
                e_k_in_c[k, c] = i_c / ((size[c] + inc)**2) - o_c / ((K - (size[c] +inc))**2)

            # Keep the clusters with highest energy
            new_c = np.argmax(e_k_in_c[k])
            # new_c == old_c
            if e_k_in_c[k, new_c] == 0: continue
            # update data, size, B and cursor
            new_k = B[new_c]
            shiftpos(cursor, _k, new_k)
            shiftpos(data, _k, new_k, axis=0)
            shiftpos(data, _k, new_k, axis=1)
            size[old_c] -= 1
            size[new_c] += 1
            # @todo: check if cluster disapear...
            #print 'B cluster update %d -> %d (B=%d)' % (old_c, new_c, B[1])
            if old_c < new_c:
                B[1:C][old_c:new_c] -= 1
            else:
                B[1:C][new_c:old_c] += 1
            ## Approx update
            ##I, L_i, O, L_o = self.energy(get_params=True)
            #I[[new_c, old_c]] = params_E_tmp[new_c, 0]
            #L_i[[new_c, old_c]] = params_E_tmp[new_c, 1]
            #O[[new_c, old_c]] = params_E_tmp[new_c, 2]
            #L_o[[new_c, old_c]] = params_E_tmp[new_c, 3]

            #delta_e = (self.energy() - E_old) - e_k_in_c[k, new_c]
            #print 'assert delta_e = 0 --', delta_e
            #print 'I', I[[new_c,old_c]], params_E_tmp[new_c,0]
            #print 'O', O[[new_c,old_c]], params_E_tmp[new_c,2]
            #print 'L_i', L_i[[new_c,old_c]], params_E_tmp[new_c,1]
            #print 'L_o', L_o[[new_c,old_c]], params_E_tmp[new_c,3]
            #if delta_e != 0:
            #    exit()

        #c_hist[::-1].sort() # sort in place
        self.pi = self.get_partitions(B)
        self.labels = self.labels[cursor]

        if self.get_C() != C:
            print( 'heeeere, some clusters are empty!!!!\n need to manage that and eventually remove clusters ?!! %d %d' % (self.get_C(), C))

        #return  1/float(C) * ((I/L_i).sum() - (O/L_o).sum())
        return self.energy()


    def sample_B(self):
        """ Sample Boundary """
        #self.iterations = self.K * self.get_C() / 2
        self.iterations = 100
        self.iterations = int(np.log(self.K) * self.get_C())
        self.iterations = 10

        for it in range(self.iterations): # Broadcast ?!
            print( it, end='')

            state = self.boundary_sample(it=it)
            E_new = self.energy(state)
            #if E_new > self.E :
            if E_new > self.E or self.anneal_transition(E_new, it):
                print (' acceptance %f, %f' %( E_new, self.E))
                self.set_state(state)
                self.E = self.concentrate_clases()
                print ('total Energy %f, B: %s' % (self.E, self.B))
        return

    def search(self):
        """ Sample partionning, until energy reach a maximum
            or reach max iterations
        """
        E_old = self.E
        old_state = self.get_state()
        while (self.get_C() < self.K):
            ### altern Reorder Boundaries - Reorder membership
            self.sample_B()

            #break
            if self.E > E_old and self.grow_rate > 0:
                ### Insert new cluster or keep the current state
                print ('adding new clusters')
                old_state = deepcopy(self.get_state())
                E_old = self.E
                self.set_state(self.boundary_sample(new=self.grow_rate))
            else:
                self.set_state(old_state)
                print ('replacing previsous state (%d cluster)' % self.get_C())
                #self.sample_B()
                break

        return self.get_clusters()

    def anneal_transition(self, E_new, it):
        if it == 0: return True
        T = float(self.iterations - it)
        #T = T / (self.K**2)
        T = T / (self.K**2 / self.get_C())
        delta = self.E - E_new
        p = np.exp(-(delta)/T * np.log(1000))
        #print 'anneal proba %f' %p
        return bool(p > np.random.random())

    def add_cluster(self, n=1):
        """ Add a new clusters in the partitionning """
        pass

    def hi_phi(self):
        """ Return the reordered data with high energy clusters """
        return self.data
    def get_labels(self):
        """ Return the reordered labels"""
        return self.labels

    def stop_criteria(self):
        return False

from scipy.stats import kstest, ks_2samp
from scipy.special import zeta
# Ref: Clauset, Aaron, Cosma Rohilla Shalizi, and Mark EJ Newman. "Power-law distributions in empirical data."
# Debug
# @todo: Estimation of x_min instead of the "max" heuristic.
# @todo: cut-off
def gofit(x, y, model='powerlaw', precision=0.03):
    """ (x, y): the empirical distribution with x the values and y **THE COUNTS** """
    y = y.astype(float)
    #### Power law Goodness of fit
    # Estimate x_min
    y_max = y.max()

    # Reconstruct the data samples
    data = degree_hist_to_list(x, y)
    #x = np.arange(1, y.sum()) ?

    # X_min heuristic /Estim
    index_min = len(y) - np.argmax(y[::-1]) # max from right
    #index_min = np.argmax(y) # max from left
    #x_min = x[index_min]
    index_min = 0
    x_min = x[index_min]
    while x_min == 0:
        index_min += 1
        x_min = x[index_min]

    ### cutoff ?
    x_max = x.max()

    # Estimate \alpha
    N = int(y.sum())
    n_tail = y[index_min:].sum()
    if n_tail < 25 or len(y) < 5:
        # no enough point
        lgg.error('Not enough samples %s' % n_tail)
        return
    #elif n_tail / N < 3/4.0:
    #    # tail not relevant
    #    index_min = len(y) - np.argmax(y[::-1]) # max from left
    #    #index_min = 0 # all the distribution
    #    x_min = x[index_min]
    #    n_tail = y[index_min:].sum()

    alpha = 1 + n_tail * (np.log(data[data>x_min] / (x_min -0.5)).sum())**-1

    ### Build The hypothesis
    if model == 'powerlaw':
        ### Discrete CDF (gives worse p-value)
        cdf = lambda x: 1 - zeta(alpha, x) / zeta(alpha, x_min)
        ### Continious CDF
        #cdf = lambda x:1-(x/x_min)**(-alpha+1)
    else:
        lgg.error('Godfit: Hypothese Unknow %s' % model)
        return

    # Number of synthetic datasets to generate #precision = 0.03
    S = int(0.25 * (precision)**-2)
    pvalue = []

    # Ignore head data
    if False:
        N = n_tail # bad effect
        ks_d = kstest(data[data>=x_min], cdf)
    else:
        ks_d = kstest(data, cdf)

    for s in range(S):
        ### p-value with Kolmogorov-Smirnov, for each synthetic dataset
        # Each synthetic dataset has following size:
        powerlaw_samples_size = np.random.binomial(N, n_tail/N)
        # plus random sample from before the cut
        out_empirical_samples_size = N - powerlaw_samples_size

        # Generate synthetic dataset
        ratio_plaw = 1
        ratio_random = 1
        powerlaw_samples = random_powerlaw(alpha, x_min, powerlaw_samples_size*ratio_plaw)

        if len(data[data<=x_min]) > 0:
            out_samples = np.random.choice((data[data<=x_min]), size=out_empirical_samples_size*ratio_random)
            sync_samples = np.hstack((out_samples, powerlaw_samples))
        else:
            sync_samples = powerlaw_samples

        ### Cutoff ?!
        #powerlaw_samples = powerlaw_samples[powerlaw_samples <= x_max]
        #ratio =  powerlaw_samples_size / len(powerlaw_samples)
        #if ratio > 1:
        #    supplement = random_powerlaw(alpha, x_min, powerlaw_samples_size * (ratio -1))
        #    supplement = supplement[supplement <= x_max]
        #    sync_samples = np.hstack((out_samples, powerlaw_samples, supplement))
        #else:
        #    sync_samples = np.hstack((out_samples, powerlaw_samples))


        #ks_2 = ks_2samp(sync_samples, d)
        ks_s = kstest(sync_samples, cdf)
        pvalue.append(ks_s.statistic >= ks_d.statistic)

    #frontend.DataBase.save(d, 'd.pk')
    #frontend.DataBase.save(sync_samples, 'sc.pk')

    pvalue = sum(pvalue) / len(pvalue)
    estim = {'alpha': alpha, 'x_min':x_min, 'y_max':y_max,
             'n_tail': int(n_tail),'n_head':N - n_tail,
             'pvalue':pvalue}
    print ('KS data: ', ks_d)
    print ('KS synthetic: ', ks_s)
    print (estim)
    estim.update({'sync':sync_samples.astype(int)})
    return estim



######################################
### External
#####################################

try:
    from sklearn.cluster import KMeans
    from sklearn.datasets.samples_generator import make_blobs
except:
    pass
def kmeans(M, K=4):
    km = KMeans( init='k-means++',  n_clusters=K)
    km.fit(M)
    clusters = km.predict(M.astype(float))
    return clusters.astype(int)

from matplotlib import pyplot as plt
def kmeans_plus(X=None, K=4):
    if X is None:
        centers = [[1, 1], [-1, -1], [1, -1]]
        K = len(centers)
        X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

    ###################################
    # Compute clustering with Means

    k_means = KMeans(init='k-means++', n_clusters=K, n_init=10)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)

    ###################################
    # Plot result

    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#4E9A97']
    plt.figure()
    plt.hold(True)
    for k, col in zip(range(K), colors):
    #for k in range(K):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], 'w',
                 markerfacecolor=col, marker='.')
        plt.plot(cluster_center[0], cluster_center[1], 'o',
                 markeredgecolor='k', markersize=6, markerfacecolor=col)
    plt.title('KMeans')
    plt.grid(True)
    plt.show()

