import os
import numpy as np
import scipy as sp
from numpy import ma
from pymake import ExpTensor, GramExp, ExpeFormat, ExpSpace
from pymake.frontend.manager import ModelManager, FrontendManager



import itertools
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
        write  = False,
        _do            = ['burstiness', 'global'], # default
        _mode         = 'generative',
        gen_size      = 1000,
        epoch         = 30 , # Gen,eration epoch
        limit_gen   = 5, # Local superposition ! Ignored ?
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
            model = ModelManager.from_expe(expe)

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
            model = ModelManager.from_expe(expe, init=True)

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

    @ExpeFormat.raw_plot('corpus')
    def burst_process_me(self, frame):

        expe = self.expe

        # Force ONE epoch # bernoulli variance...
        expe.epoch = 1
        self._generate()

        Y = self._Y
        Theta = self._Theta
        Phi = self._Phi

        ax = frame.ax


        for i in range(expe.epoch):

            theta = Theta[i]
            phi = Phi[i]

            N = theta.shape[0]
            process = np.zeros((N, N))

            likelihood = self.model.likelihood(theta, phi)
            adj = sp.stats.bernoulli.rvs(likelihood)

            process = np.cumsum(adj, 1)

            legend = expe.model
            title = expe.corpus

            step = 3
            ax.errorbar(np.arange(N)[0::step],
                        process.mean(0)[0::step],
                        yerr=process.std(0)[0::step],
                        fmt='o', label=legend,
                        elinewidth=1, capsize=0, alpha=0.3,
                       )

        ax.legend(loc=1,prop={'size':10})
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cumulative Sum')

    @ExpeFormat.raw_plot('corpus')
    def burst_process_mg(self, frame):

        expe = self.expe

        self._generate()

        Y = self._Y
        Theta = self._Theta
        Phi = self._Phi

        ax = frame.ax

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


        # Mean-Variance of the count process expectation
        mean_process = np.vstack((p.mean(0) for p in processes))

        # Std-Variance of the count process expectation
        var_process = np.vstack((p.std(0) for p in processes))

        legend = expe.model
        title = expe.corpus

        step = 3
        _process = mean_process
        ax.errorbar(np.arange(N)[0::step],
                    _process.mean(0)[0::step],
                    yerr=_process.std(0)[0::step],
                    fmt='o', label=legend,
                    elinewidth=1, capsize=0, alpha=0.3,
                   )

        ax.legend(loc=1,prop={'size':10})
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cumulative Sum')


