from collections import defaultdict
import numpy as np
from graph_tool.all import *

from .modelbase import ModelBase


class SBM_BASE(ModelBase):
    __abstractmethods__ = 'model'

    def _init_params(self, frontend):
        frontend = self.frontend

        # Save the testdata
        data_test = np.transpose(self.frontend.data_test.nonzero())
        frontend.reverse_filter()
        weights = []
        for i,j in data_test:
            weights.append(frontend.weight(i,j))
        frontend.reverse_filter()
        self.data_test = np.hstack((data_test, np.array(weights)[None].T))

        _len = {}
        _len['K'] = self.expe.get('K')
        _len['N'] = frontend.num_nodes()
        _len['E'] = frontend.num_edges()
        _len['nnz'] = frontend.num_nnz()
        #_len['nnz_t'] = frontend.num_nnz_t()
        _len['dims'] = frontend.num_neighbors()
        _len['nnz_ones'] = frontend.num_edges()
        _len['nnzsum'] = frontend.num_nnzsum()
        self._len = _len

        self._K = self._len['K']
        self._is_symmetric = frontend.is_symmetric()

    def _reduce_latent(self):
        theta = self._state.get_blocks()
        phi = self._state.get_matrix()

        return theta, phi

    def _equilibrate(self):
        K = self.expe.K

        g = self.frontend.data

        measures = defaultdict(list)
        def collect_marginals(s):
            measures['entropy'].append(s.entropy())

        # Model
        state = BlockState(g, B=K)
        mcmc_equilibrate(state, callback=collect_marginals)
        #mcmc_equilibrate(state, callback=collect_marginals, force_niter=expe.iterations, mcmc_args=dict(niter=1), gibbs=True)
        #mcmc_equilibrate(state, force_niter=expe.iterations, callback=collect_marginals, wait=0, mcmc_args=dict(niter=10))

        entropy = np.array(measures['entropy'])

        #print("Change in description length:", ds)
        #print("Number of accepted vertex moves:", nmoves)

        # state.get_edges_prob(e)
        #theta = state.get_ers()

        print('entropy:')
        print(entropy)
        print(len(entropy))


class SBM_gt(ModelBase):

    def fit(self):
        self._state =


