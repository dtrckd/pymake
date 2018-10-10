import os
import numpy as np
import scipy as sp
from numpy import ma
from collections import defaultdict
import itertools
import networkx as nx

from pymake import ExpTensor, GramExp, ExpeFormat, ExpSpace
from pymake.frontend.manager import ModelManager, FrontendManager

from pymake.util.algo import gofit, Louvain, Annealing
from pymake.util.math import reorder_mat, sorted_perm, categorical, clusters_hist
from pymake.util.utils import Now, nowDiff, colored
import matplotlib.pyplot as plt
from pymake.plot import plot_degree, degree_hist, adj_to_degree, plot_degree_poly, adjshow, plot_degree_2, random_degree,  draw_graph_circular, draw_graph_spectral, draw_graph_spring, _markers
from pymake.core.format import tabulate

from sklearn.metrics import roc_curve, auc, precision_recall_curve


class BurstProcess(ExpeFormat):

    _default_expe = dict(
        block_plot = False,
        _write  = False,
        _do            = ['burstiness', 'global'], # default
        _mode         = 'generative',
        gen_size      = 1000,
        epoch         = 30 , # Gen,eration epoch
        limit_gen   = 5, # Local superposition ! Ignored ?
        limit_class   = 15, # Ignored ?
    )

    def _preprocess(self):
        expe = self.expe

        frontend = FrontendManager.load(expe)
        if frontend:
            self._N = frontend.getN()
            expe.symmetric = frontend.is_symmetric()
        else:
            self._N = expe.N
            expe.symmetric = True


        if expe._mode == 'predictive':
            ### Generate data from a fitted model
            model = ModelManager.from_expe(expe, load=True)

            try:
                # this try due to mthod modification entry in init not in picke object..
                expe.hyperparams = model.get_hyper()
            except Exception as e:
                self.log.warning('loading hyperparam error: %s' % e)
                if model is not None:
                    model._mean_w = 0
                    expe.hyperparams = 0

        elif expe._mode == 'generative':
            ### Generate data from a un-fitted model

            expe.alpha = 1
            expe.gmma = 1/2
            expe.delta = [0.5,0.5]

            if 'ilfm' in expe.model:
                keys_hyper = ('alpha','delta')
                hyper = (expe.alpha, expe.delta)
            else:
                keys_hyper = ('alpha','gmma','delta')
                hyper = (expe.alpha, expe.gmma, expe.delta)
            expe.hyperparams = dict(zip(keys_hyper, hyper))
            expe.hyper = 'fix' # dummy
            model = ModelManager.from_expe(expe, load=False)

        else:
            raise NotImplementedError('What generation context ? predictive/generative..')

        self.log.info('=== GenNetworks === ')
        self.log.info('Mode: %s' % expe._mode)
        self.log.info('===')
        self.log.info('hyper: %s' % (str(expe.hyperparams)))

        self.frontend = frontend
        self.model = model

        if model is None:
            raise FileNotFoundError('No model for Expe at :  %s' % self.output_path)



    def _generate(self):
        ''' Generate data. '''
        expe = self.expe
        model = self.model

        Y = []
        Theta = []
        Phi = []
        now = Now()
        for i in range(expe.epoch):
            y, theta, phi = model.generate(mode=expe._mode,
                                           N=self._N, K=expe.K,
                                           symmetric=expe.symmetric,
                                          )
            Y.append(y)
            Theta.append(theta)
            Phi.append(phi)
        self.log.info('Data Generation : %s second' % nowDiff(now))

        #if expe.symmetric:
        #    for y in Y:
        #        frontend.symmetrize(y)

        self._Y = Y
        self._Theta = Theta
        self._Phi = Phi

    def _local_likelihood(self, theta, phi, k1, k2=None):
        ''' Expected local adjacency matrix more that likelihood. '''

        if not k2:
            k2 = k1

        model = self.expe.model

        if 'mmsb' in model:
            ll = np.outer(theta[:,k1], theta[:,k2]) * phi[k1, k2]
        elif 'ilfm' in model:
            ll = theta.dot(phi).dot(theta.T)
            ll = 1 / (1 + np.exp(-1 * ll))
            ll = sp.stats.bernoulli.rvs(ll) * np.outer(theta[:,k1], theta[:,k2])
        else:
            raise ValueError('Model unknow for local likelihood computation')
        return ll

    @ExpeFormat.raw_plot('corpus')
    def burst_process_me(self, frame):

        expe = self.expe

        # Force ONE epoch # bernoulli variance...
        expe.epoch = 1
        self._generate()

        Y = self._Y
        Theta = self._Theta
        Phi = self._Phi

        for i in range(expe.epoch):

            theta = Theta[i]
            phi = Phi[i]
            N = theta.shape[0]
            process = np.zeros((N, N))

            likelihood = self.model.likelihood(theta, phi)
            adj = sp.stats.bernoulli.rvs(likelihood)
            process = np.cumsum(adj, 1)


        ax = frame.ax()
        legend = expe.model
        step = 3
        ax.errorbar(np.arange(N)[0::step],
                    process.mean(0)[0::step],
                    yerr=process.std(0)[0::step],
                    fmt='o', label=legend,
                    elinewidth=1, capsize=0, alpha=0.3,
                   )

        ax.legend(loc=2,prop={'size':10})
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cumulative Sum')

    @ExpeFormat.raw_plot('corpus')
    def burst_process_mg(self, frame):

        expe = self.expe

        self._generate()

        Y = self._Y
        Theta = self._Theta
        Phi = self._Phi

        processes = []

        for i in range(expe.epoch):

            theta = Theta[i]
            phi = Phi[i]
            N = theta.shape[0]

            process = np.zeros((N, N))
            likelihood = self.model.likelihood(theta, phi)
            adj = sp.stats.bernoulli.rvs(likelihood)

            process = np.cumsum(adj, 1)
            processes.append(process)

        # Mean-Variance of the count process expectation
        mean_process = np.vstack((p.mean(0) for p in processes))
        # Std-Variance of the count process expectation
        var_process = np.vstack((p.var(0) for p in processes))

        legend = expe.model
        step = 3

        ax1 = frame.fig.add_subplot(1,2,1)
        ax2 = frame.fig.add_subplot(1,2,2)

        ax1.errorbar(np.arange(N)[0::step],
                    mean_process.mean(0)[0::step],
                    yerr=mean_process.std(0)[0::step],
                    fmt='o', label=legend,
                    elinewidth=1, capsize=0, alpha=0.3,)

        ax2.errorbar(np.arange(N)[0::step],
                    var_process.mean(0)[0::step],
                    yerr=var_process.std(0)[0::step],
                    fmt='o', label=legend,
                    elinewidth=1, capsize=0, alpha=0.3,)

        ax1.legend(loc=2,prop={'size':10})
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Expectation')
        ax2.set_ylabel('variance')
        frame.title = 'degree Cumutative Sum'

        default_size = frame.fig.get_size_inches()
        frame.fig.set_size_inches((default_size[0]*2, default_size[1]))


    @ExpeFormat.raw_plot('model')
    def burst_process_local_me(self, frame):

        expe = self.expe

        # Force ONE epoch # bernoulli variance...
        expe.epoch = 1
        self._generate()

        Y = self._Y
        Theta = self._Theta
        Phi = self._Phi

        for i in range(expe.epoch):

            theta = Theta[i]
            phi = Phi[i]

            N = theta.shape[0]
            K = theta.shape[1]
            process = np.zeros((K, N, N))

            for _k in range(K):
                process[_k] = self._local_likelihood(theta, phi, _k)

            process = np.cumsum(process, 2)

        ax = frame.ax()
        step = 3


        for _k in range(K):
            _process = process[_k]
            legend = expe.model + ' K=%d'%(_k)
            ax.errorbar(np.arange(N)[0::step],
                        _process.mean(0)[0::step],
                        yerr=_process.std(0)[0::step],
                        fmt='o', label=legend,
                        elinewidth=1, capsize=0, alpha=0.3,
                       )

        ax.legend(loc=2,prop={'size':10})
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cumulative Sum')


    @ExpeFormat.raw_plot('model')
    def burst_process_local_mg(self, frame):

        expe = self.expe

        self._generate()

        Y = self._Y
        Theta = self._Theta
        Phi = self._Phi

        Ks = [o.shape[1] for o in Theta]
        k_min = min(Ks)
        k_max= max(Ks)

        processes = []

        for i in range(expe.epoch):

            theta = Theta[i]
            phi = Phi[i]

            N = theta.shape[0]
            K = theta.shape[1]
            #if K > k_min:
            #    print('warning: not all block converted.')
            #    continue
            process = np.zeros((K, N, N))

            for _k in range(K):
                process[_k] = self._local_likelihood(theta, phi, _k)

            if K < k_max:
                # Ks are ordered !
                _pad = k_max - K
                process = np.pad(process, [(0,_pad),(0,0),(0,0)], mode='constant', constant_values=0)

            process = np.cumsum(process, 2)
            processes.append(process)

        # Mean-Variance of the count process expectation
        mean_process = np.stack((p.mean(1) for p in processes))
        # Std-Variance of the count process expectation
        var_process = np.stack((p.var(1) for p in processes))

        ax1 = frame.fig.add_subplot(1,2,1)
        ax2 = frame.fig.add_subplot(1,2,2)
        step = 3

        for _k in range(K):
            _process = mean_process[:,_k]
            legend = expe.model + ' K=%d'%(_k)
            ax1.errorbar(np.arange(N)[0::step],
                        _process.mean(0)[0::step],
                        yerr=_process.std(0)[0::step],
                        fmt='o', label=legend,
                        elinewidth=1, capsize=0, alpha=0.3,)

            _process = var_process[:,_k]
            legend = expe.model + ' K=%d'%(_k)
            ax2.errorbar(np.arange(N)[0::step],
                        _process.mean(0)[0::step],
                        yerr=_process.std(0)[0::step],
                        fmt='o',# label=legend,
                        elinewidth=1, capsize=0, alpha=0.3,)

        #ax1.set_xscale('log'); ax1.set_yscale('log')
        #ax2.set_xscale('log'); ax2.set_yscale('log')
        ax1.legend(loc=2,prop={'size':10})
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Expectation')
        ax2.set_ylabel('Variance')
        frame.title = 'Degree Cumulative Sum'

        default_size = frame.fig.get_size_inches()
        frame.fig.set_size_inches((default_size[0]*2, default_size[1]))


#
#
# Second phase ("temporal process")
#
#
#
    @ExpeFormat.raw_plot()
    #@ExpeFormat.raw_plot('corpus')
    def prop_process_me(self, frame, p=250):
        p = int(p)
        expe = self.expe

        # Force ONE epoch # bernoulli variance...
        expe.epoch = 1
        self._generate()

        Y = self._Y
        Theta = self._Theta
        Phi = self._Phi

        theta = Theta[0]
        phi = Phi[0]

        N = theta.shape[0]

        likelihood = self.model.likelihood(theta, phi)
        adj = sp.stats.bernoulli.rvs(likelihood)

        n_to_zeros = N-p
        _idx = np.random.choice(np.arange(N**2), n_to_zeros, replace=False)
        idx = np.unravel_index(_idx, (N,N))

        adj_n = adj.copy()
        adj_n[idx] = 0

        g_n = nx.from_numpy_array(adj_n)
        degree_n = dict(g_n.degree())
        d_n, dc_n = degree_hist(degree_n, filter_zeros=False)

        g = nx.from_numpy_array(adj)
        degree = dict(g.degree())
        d, dc = degree_hist(degree, filter_zeros=False)


        x = d_n
        y = []
        burstiness = defaultdict(lambda:0)
        normalize_deg = defaultdict(lambda:0)
        # Compute p(d(N)>n+1 | p(p)=n)
        for node, deg_n in degree_n.items():
            if deg_n == 0:
                continue

            deg_N = degree[node]
            if deg_N > deg_n:
                burstiness[deg_n] += 1

            normalize_deg[deg_n] +=1

        # Normalize
        for deg, total in normalize_deg.items():
            burstiness[deg] = burstiness[deg] / total

        for deg in x:
            y.append(burstiness[deg])

        ax = frame.ax()
        ax.plot(x, y, label=expe.model)

        ax.legend(loc=1,prop={'size':10})
        ax.set_xlabel('n')
        ax.set_ylabel('Cumulative Sum')
        frame.title = expe.corpus + ' p=%s' % p


    @ExpeFormat.raw_plot()
    #@ExpeFormat.raw_plot('corpus')
    def prop_process_local_me(self, frame, p=250):
        p = int(p)
        expe = self.expe

        # Force ONE epoch # bernoulli variance...
        expe.epoch = 1
        self._generate()

        Y = self._Y
        Theta = self._Theta
        Phi = self._Phi

        theta = Theta[0]
        phi = Phi[0]

        N = theta.shape[0]
        K = theta.shape[1]

        for _k1 in range(K):
            #for _k2 in range(K):

                n_to_zeros = N-p

                adj = self._local_likelihood(theta, phi, _k1)
                #if adj.dtype == np.dtype(float):
                #    adj = sp.stats.bernoulli.rvs(adj)

                #    _id1 = np.arange(N**2).reshape((N,N))[adj==1]
                #    _id0 = np.arange(N**2).reshape((N,N))[adj==0]
                #    nn1 = len(_id1) // 3
                #    nn0 = n_to_zeros - nn1
                #    if nn1 > 0:
                #        _idx1 = np.random.choice(_id1, nn1, replace=False)
                #        _idx0 = np.random.choice(_id0, nn0, replace=False)
                #        _idx = np.hstack((_idx0, _idx1))
                #    else:
                #        _idx = _id0

                #    idx = np.unravel_index(_idx, (N,N))
                _idx = np.random.choice(np.arange(N**2), n_to_zeros, replace=False)
                idx = np.unravel_index(_idx, (N,N))


                adj_n = adj.copy()
                adj_n[idx] = 0

                if 'ilfm' in expe.model:
                    g_n = nx.from_numpy_array(adj_n)
                    degree_n = dict(g_n.degree())
                else:
                    degree_n = dict((i,int(round(d))) for i,d in enumerate(adj_n.sum(1)))
                d_n, dc_n = degree_hist(degree_n, filter_zeros=False)

                if 'ilfm' in expe.model:
                    g = nx.from_numpy_array(adj)
                    degree = dict(g.degree())
                else:
                    degree = dict((i,int(round(d))) for i,d in enumerate(adj.sum(1)))
                d, dc = degree_hist(degree, filter_zeros=False)

                x = d_n
                y = []
                burstiness = defaultdict(lambda:0)
                normalize_deg = defaultdict(lambda:0)
                # Compute p(d(N)>n+1 | p(p)=n)
                for node, deg_n in degree_n.items():
                    if deg_n == 0:
                        continue

                    deg_N = degree[node]
                    if deg_N > deg_n:
                        burstiness[deg_n] += 1

                    normalize_deg[deg_n] +=1

                # Normalize
                for deg, total in normalize_deg.items():
                    burstiness[deg] = burstiness[deg] / total

                for deg in x:
                    y.append(burstiness[deg])

                ax = frame.ax()

                label = '%s K=%d' % (expe.model, _k1)
                ax.plot(x, y, label=label)

        ax.legend(loc=1,prop={'size':8})
        ax.set_xlabel('n')
        ax.set_ylabel('Cumulative Sum')
        frame.title = expe.corpus + ' p=%s' % p



    @ExpeFormat.raw_plot()
    #@ExpeFormat.raw_plot('corpus')
    def prop2_process_me(self, frame, p=90):
        p = int(p)
        expe = self.expe

        # Force ONE epoch # bernoulli variance...
        expe.epoch = 1
        self._generate()

        Y = self._Y
        Theta = self._Theta
        Phi = self._Phi

        theta = Theta[0]
        phi = Phi[0]

        N = theta.shape[0]

        likelihood = self.model.likelihood(theta, phi)
        adj = sp.stats.bernoulli.rvs(likelihood)

        n_to_zeros = int( N**2 * (1-p/100))
        _idx = np.random.choice(np.arange(N**2), n_to_zeros, replace=False)
        idx = np.unravel_index(_idx, (N,N))

        adj_n = adj.copy()
        adj_n[idx] = 0

        g_n = nx.from_numpy_array(adj_n)
        degree_n = dict(g_n.degree())
        d_n, dc_n = degree_hist(degree_n, filter_zeros=False)

        g = nx.from_numpy_array(adj)
        degree = dict(g.degree())
        d, dc = degree_hist(degree, filter_zeros=False)


        x = d_n
        y = []
        burstiness = defaultdict(lambda:0)
        normalize_deg = defaultdict(lambda:0)
        # Compute p(d(N)>n+1 | p(p)=n)
        for node, deg_n in degree_n.items():
            if deg_n == 0:
                continue

            deg_N = degree[node]
            if deg_N > deg_n:
                burstiness[deg_n] += 1

            normalize_deg[deg_n] +=1

        # Normalize
        for deg, total in normalize_deg.items():
            burstiness[deg] = burstiness[deg] / total

        for deg in x:
            y.append(burstiness[deg])

        ax = frame.ax()

        w = 0.2
        opacity = 0.8
        y = np.array(y)
        index = 0
        ticks = []
        ticks_label = []
        c = self.colors.next()
        for n, v in enumerate(y):
            if v == 0:
                continue
            rects = ax.bar(index + w, v, w,
                           alpha=opacity,
                           label=None, color=c)
            ticks.append(index)
            ticks_label.append(x[n])
            index += 1

        ax.set_xticklabels(ticks_label)
        ticks = np.array(ticks) + w
        ax.set_xticks(ticks)
        ax.xaxis.set_tick_params(labelsize=8)

        ax.legend(loc=4,prop={'size':10})
        ax.set_xlabel('n')
        ax.set_ylabel('Probability of new links')
        frame.title = '%s, %s,  p=%s' % (self.specname(expe.corpus),
                                         self.specname(expe.model), p)

    @ExpeFormat.raw_plot()
    def prop2_process_local_me(self, frame, p=90):
        p = int(p)
        expe = self.expe

        # Force ONE epoch # bernoulli variance...
        expe.epoch = 1
        self._generate()

        Y = self._Y
        Theta = self._Theta
        Phi = self._Phi

        theta = Theta[0]
        phi = Phi[0]

        N = theta.shape[0]
        K = theta.shape[1]

        ticks = []
        ticks_label = []

        nb_class = 0
        for _k1 in range(K):
            if 'ilfm' in expe.model:
                K =1
            for _k2 in range(K):
                if 'ilfm' in expe.model:
                    _k2 = _k1

                if nb_class >= 4:
                    break

                n_to_zeros = int( N**2 * (1-p/100))

                adj = self._local_likelihood(theta, phi, _k1, _k2)
                #if adj.dtype == np.dtype(float):
                #    adj = sp.stats.bernoulli.rvs(adj)

                #    _id1 = np.arange(N**2).reshape((N,N))[adj==1]
                #    _id0 = np.arange(N**2).reshape((N,N))[adj==0]
                #    nn1 = len(_id1) // 3
                #    nn0 = n_to_zeros - nn1
                #    if nn1 > 0:
                #        _idx1 = np.random.choice(_id1, nn1, replace=False)
                #        _idx0 = np.random.choice(_id0, nn0, replace=False)
                #        _idx = np.hstack((_idx0, _idx1))
                #    else:
                #        _idx = _id0

                #    idx = np.unravel_index(_idx, (N,N))
                _idx = np.random.choice(np.arange(N**2), n_to_zeros, replace=False)

                idx = np.unravel_index(_idx, (N,N))

                adj_n = adj.copy()
                adj_n[idx] = 0

                if 'ilfm' in expe.model:
                    g_n = nx.from_numpy_array(adj_n)
                    degree_n = dict(g_n.degree())
                else:
                    degree_n = dict((i,int(round(d))) for i,d in enumerate(adj_n.sum(1)))
                d_n, dc_n = degree_hist(degree_n, filter_zeros=False)

                if 'ilfm' in expe.model:
                    g = nx.from_numpy_array(adj)
                    degree = dict(g.degree())
                else:
                    degree = dict((i,int(round(d))) for i,d in enumerate(adj.sum(1)))
                d, dc = degree_hist(degree, filter_zeros=False)

                x = d_n

                y = []
                burstiness = defaultdict(lambda:0)
                normalize_deg = defaultdict(lambda:0)
                # Compute p(d(N)>n+1 | p(p)=n)
                for node, deg_n in degree_n.items():
                    if deg_n == 0:
                        continue

                    deg_N = degree[node]
                    if deg_N > deg_n:
                        burstiness[deg_n] += 1

                    normalize_deg[deg_n] +=1

                # Normalize
                for deg, total in normalize_deg.items():
                    burstiness[deg] = burstiness[deg] / total

                for deg in x:
                    y.append(burstiness[deg])

                y = np.array(y)
                if len(y[y>0]) <= 3:
                    continue

                ax = frame.ax()

                w = 0.4
                opacity = 0.8
                y = np.array(y)
                _index = 0
                index = _index + len(ticks)*1.25
                c = self.colors.next()
                label = '%s K=%d' % (expe.model, _k1)
                one_count = 0
                first = True
                for n, v in enumerate(y):
                    if v == 0:
                        continue
                    if v == 1:
                        one_count += 1
                    else:
                        one_count = 0

                    if one_count >=4:
                        break

                    if first:
                        label = 'classe %s' % nb_class
                    else:
                        label = None


                    rects = ax.bar(index + w/2, v, w,
                                   alpha=opacity,
                                   label=label, color=c)
                    ticks.append(index)
                    ticks_label.append(x[n])
                    index += 1
                    first = False

                nb_class +=1

        ax.set_xticklabels(ticks_label)
        ticks = np.array(ticks) + w
        ax.set_xticks(ticks)
        ax.xaxis.set_tick_params(labelsize=8)

        ax.legend(loc=4,prop={'size':10})
        ax.set_xlabel('n')
        ax.set_ylabel('Probability of new links')
        frame.title = '%s, %s,  p=%s' % (self.specname(expe.corpus),
                                         self.specname(expe.model), p)

    @ExpeFormat.raw_plot()
    def power_law_mm(self, frame, p=90):
        p = int(p)
        expe = self.expe

        # Force ONE epoch # bernoulli variance...
        expe.epoch = 1
        self._generate()

        Y = self._Y
        Theta = self._Theta
        Phi = self._Phi

        theta = Theta[0]
        phi = Phi[0]

        N = theta.shape[0]
        K = theta.shape[1]

        ticks = []
        ticks_label = []

        alpha = np.exp(self.model.s.zsampler.log_alpha_beta)[:K]

        for _i in range(N):

            if _i >= 3:
                break
            plt.figure()

            fi_params = (2*N + alpha.sum()) * theta[_i]

            samples = np.random.dirichlet(fi_params, 1000)

            for _k in range(K):
                plt.hist(samples[:, _k], bins=100)


            plt.ylim(0, 100)
            plt.xlabel('f_i %d values'%_i)
            plt.ylabel('count')




