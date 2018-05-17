from collections import defaultdict
import numpy as np
import scipy as sp
from numpy import ma
import scipy.stats

from .modelbase import ModelBase

import graph_tool as gt
from graph_tool import inference


class SbmBase(ModelBase):
    __abstractmethods__ = 'model'

    def _init_params(self, frontend):
        frontend = self.frontend

        # Save the testdata
        self.data_test = frontend.data_test_w

        # For fast computation of bernoulli pmf.
        self._w_a = self.data_test[:,2].T.astype(int)
        self._w_a[self._w_a > 0] = 1
        self._w_a[self._w_a == 0] = -1
        self._w_b = np.zeros(self._w_a.shape, dtype=int)
        self._w_b[self._w_a == -1] = 1

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

    def _reduce_latent(self):
        if hasattr(self, 'get_overlap_blocks'):
            raise NotImplementedError
        else:
            try:
               _theta = self._state.get_blocks().a
               theta = np.zeros((len(_theta), self._K))
               theta[(np.arange(len(_theta)), _theta)] = 1
            except AttributeError as e:
                return self._theta, self._phi

        phi = self._state.get_matrix().A

        nr = self._state.get_nr()
        norm = np.outer(nr.a, nr.a)

        if not getattr(self, '_weighted', False):
            #phi = phi / phi.sum()
            phi = phi / norm

        self._theta, self._phi = theta, phi

        return theta, phi

    def likelihood(self, theta=None, phi=None):
        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi

        qijs = []
        for i,j, xij in self.data_test:
            #try:
            #   # Too long !
            #    pij = np.exp(self._state.get_edges_prob([(i,j),]))
            #except AttributeError:
            #    # If recomputed with fig :roc...
            #    pij = theta[i].dot(phi).dot(theta[j])
            pij = theta[i].dot(phi).dot(theta[j])

            qijs.append( pij )

        qijs = ma.masked_invalid(qijs)
        return qijs


    def compute_entropy(self, theta=None, phi=None, **kws):
        if 'likelihood' in kws:
            pij = kws['likelihood']
        else:
            if theta is None:
                theta, phi = self._reduce_latent()
            pij = self.likelihood(theta, phi)

        ll = pij * self._w_a + self._w_b

        ll[ll<=1e-300] = 1e-300
        # Log-likelihood
        ll = np.log(ll).sum()
        # Perplexity is 2**H(X).
        return ll
        #return self._state.entropy()

    def compute_roc(self, theta=None, phi=None, **kws):
        from sklearn.metrics import roc_curve, auc, precision_recall_curve

        if 'likelihood' in kws:
            pij = kws['likelihood']
        else:
            if theta is None:
                theta, phi = self._reduce_latent()
            pij = self.likelihood(theta, phi)

        weights = np.squeeze(self.data_test[:,2].T)

        y_true = weights.astype(bool)*1
        self._probas = pij
        self._y_true = y_true

        fpr, tpr, thresholds = roc_curve(y_true, pij)
        roc = auc(fpr, tpr)
        return roc

    def compute_pr(self, *args, **kwargs):
        from sklearn.metrics import average_precision_score
        return average_precision_score(self._y_true, self._probas)

    def compute_wsim(self, *args, **kws):
        return None


    def _spec_from_expe(self, _model):
        ''' Set Sklearn parameters. '''
        import inspect # @temp to be integrated

        model_params = list(inspect.signature(_model).parameters)
        spec = dict()
        spec_map = getattr(self, 'spec_map', {})
        default_spec = getattr(self, '_default_spec', {})
        for k in model_params:
            if k in list(self.expe)+list(spec_map):
                _k = spec_map.get(k, k)
                if _k in self.expe:
                    spec[k] = self.expe[_k]
                elif callable(_k):
                    spec[k] = _k(self)
            elif k in default_spec:
                spec[k] = default_spec[k]


        return spec

    def fit(self, frontend):
        self._init(frontend)
        g = self.frontend.data

        fit_fun = inference.minimize_blockmodel_dl
        spec = self._spec_from_expe(fit_fun)

        self.log.info("Fitting `%s' model with spec: %s" % (type(self), str(spec)))
        self._state = fit_fun(g, **spec)

        #inference.mcmc_equilibrate(self._state)

        self.compute_measures()
        if self.expe.get('_write'):
            self.write_current_state(self)


class SBM_gt(SbmBase):

    _default_spec = dict(deg_corr=False, overlap=False, layers=False)
    spec_map = dict(B_min='K', B_max='K')


class OSBM_gt(SbmBase):

    _default_spec = dict(deg_corr=False, overlap=True, layers=False)
    spec_map = dict(B_min='K', B_max='K')

class WSBM_gt(SbmBase):

    _default_spec = dict(deg_corr=False, overlap=False, layers=False)
    spec_map = dict(B_min='K', B_max='K', state_args=lambda self: {'recs':[self.frontend.data.ep.weights], 'rec_types' : ["discrete-poisson"]})
    _weighted = True

    def compute_entropy(self, theta=None, phi=None, **kws):
        if 'likelihood' in kws:
            pij = kws['likelihood']
        else:
            if theta is None:
                theta, phi = self._reduce_latent()
            pij = self.likelihood(theta, phi)

        weights = self.data_test[:,2].T
        ll = sp.stats.poisson.pmf(weights, pij)

        ll[ll<=1e-300] = 1e-300
        # Log-likelihood
        ll = np.log(ll).sum()
        # Perplexity is 2**H(X).
        return ll
        #return self._state.entropy()

    def compute_roc(self, theta=None, phi=None, treshold=1, **kws):
        from sklearn.metrics import roc_curve, auc, precision_recall_curve

        if 'likelihood' in kws:
            pij = kws['likelihood']
        else:
            if theta is None:
                theta, phi = self._reduce_latent()
            pij = self.likelihood(theta, phi)

        trsh = treshold
        weights = np.squeeze(self.data_test[:,2].T)

        nr = self._state.get_nr()
        norm = np.outer(nr.a, nr.a)
        phi = phi / norm
        #phi = phi / phi.sum()

        probas = []
        for i,j, xij in self.data_test:
            pij =  theta[i].dot(phi).dot(theta[j])
            probas.append( pij )

        probas = ma.masked_invalid(probas)
        #probas = 1 - sp.stats.poisson.pmf(0, probas)
        #probas = pij

        y_true = weights.astype(bool)*1
        self._y_true = y_true
        self._probas = probas

        fpr, tpr, thresholds = roc_curve(y_true, probas)
        roc = auc(fpr, tpr)
        return roc

    def compute_wsim(self, theta=None, phi=None, **kws):
        if 'likelihood' in kws:
            pij = kws['likelihood']
        else:
            if theta is None:
                theta, phi = self._reduce_latent()
            pij = self.likelihood(theta, phi)

        weights = self.data_test[:,2].T

        nr = self._state.get_nr()

        ws = np.array([ theta[i].dot(phi).dot(theta[j]) for i,j,w in self.data_test if w > 0])

        # l1 norm
        wd = weights[weights>0]
        nnz = len(wd)
        mean_dist = np.abs(ws - wd).sum() / nnz
        return mean_dist

class WSBM2_gt(SbmBase):
    # graph_tool work in progress for weithed networds...
    _default_spec = dict(deg_corr=False, overlap=False, layers=False)
    spec_map = dict(B_min='K', B_max='K',
                    state_args=lambda self:{'eweight': self.frontend.data.ep.weights,'recs':[self.frontend.data.ep.weights], 'rec_types': ["discrete-poisson"]})




class OWSBM_gt(SbmBase):
    pass


