import numpy as np
import scipy as sp

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import mean_squared_error

from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix

from ml.model import RandomGraphModel

try:
    from rescal import rescal_als
except ImportError as e:
    print('Import Error: %s' % e)


class Rescal_als(RandomGraphModel):

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

        # Log-likelihood (Perplexity is 2**H(X).)
        ll[ll<=1e-300] = 1e-200
        ll = np.log(ll).sum()

        return ll



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

        A, R, fit, itr, exectimes = rescal_als(data, K, init='nvecs', lambda_A=10, lambda_R=10)

        self._theta = A
        self._phi = R

        self.log.info('rescal fit info: %s; itr: %s, exectimes: %s' % (fit, itr, exectimes))

        self.compute_measures()
        if self.expe.get('_write'):
            self.write_current_state(self)






def rescal(X, K):

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

