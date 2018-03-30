from collections import defaultdict
import numpy as np
from graph_tool.all import *

from .modelbase import ModelBase



class SBM_gt(ModelBase):


    def fit(self, frontend):
        K = self.expe.K

        g = frontend.data

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

        theta = state.get_blocks()
        phi = state.get_matrix()

        print('entropy:')
        print(entropy)
        print(len(entropy))


    def likelihood(self, theta=None, phi=None):
        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi

        bilinear_form = theta.dot(phi).dot(theta.T)
        likelihood = 1 / (1 + np.exp(-bilinear_form))

        likelihood =  likelihood[:,0,:]
        return likelihood

    def generate(self, N=None, K=None, hyperparams=None, mode='predictive', symmetric=True, **kwargs):
        pass
