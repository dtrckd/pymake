import numpy as np
import scipy as sp

from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
try:
    from rescal import rescal_als
except ImportError as e:
    print('Import Error: %s' % e)


from .modelbase import ModelBase

class Rescal_als(ModelBase):

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

        self._K = self.expe.K

    def _reduce_latent(self):
        return self._theta, self._phi

    def likelihood(self, theta=None, phi=None):
        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi

        qijs = []
        for i,j, xij in self.data_test:
            qijs.append( theta[i].dot(phi[0]).dot(theta[j]) )

        qijs = np.array(qijs)
        likelihood = 1 / (1 + np.exp(-qijs))

        return likelihood

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

    def generate(self, N=None, K=None, hyperparams=None, mode='predictive', symmetric=True, **kwargs):
        likelihood = self.likelihood()
        #likelihood[likelihood <= 0.5] = 0
        #likelihood[likelihood > 0.5] = 1
        #Y = likelihood
        Y = sp.stats.bernoulli.rvs(likelihood)
        return Y

    def fit(self, frontend):
        self._init(frontend)
        K = self.expe.K
        y = frontend.adj()
        data = [y]

        self.log.info("Fitting `%s' model" % (type(self)))
        A, R, fit, itr, exectimes = rescal_als(data, K, init='nvecs', lambda_A=10, lambda_R=10)

        self._theta = A
        self._phi = R

        self.log.info('rescal fit info: %s; itr: %s, exectimes: %s' % (fit, itr, exectimes))

        self.compute_measures()
        if self.expe.get('_write'):
            self.write_current_state(self)






def rescal(X, K):

    ## Set logging to INFO to see RESCAL information
    #logging.basicConfig(level=logging.INFO)

    ## Load Matlab data and convert it to dense tensor format
    #T = loadmat('data/alyawarra.mat')['Rs']
    #X = [lil_matrix(T[:, :, k]) for k in range(T.shape[2])]

    X = [sp.sparse.csr_matrix(X)]
    A, R, fit, itr, exectimes = rescal_als(X, K, init='nvecs', lambda_A=10, lambda_R=10)

    theta =  A.dot(R).dot(A.T)
    Y = 1 / (1 + np.exp(-theta))
    Y =  Y[:,0,:]
    Y[Y <= 0.5] = 0
    Y[Y > 0.5] = 1
    #Y = sp.stats.bernoulli.rvs(Y)
    return Y

