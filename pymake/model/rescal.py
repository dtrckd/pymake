import numpy as np
import scipy as sp

from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
try:
    from rescal import rescal_als
except:
    pass


from .modelbase import ModelBase

class Rescal(ModelBase):

    def __init__(self, expe, frontend):
        super().__init__(frontend, **expe)

        self.expe = expe

        #self.frontend = frontend # @debug, typo ?
        self.fr = self.frontend = frontend
        self.mask = self.fr.data_ma.mask

    def fit(self):
        data = self.frontend.data_ma
        K = self.expe.K

        data = [sp.sparse.csr_matrix(data)]
        A, R, fit, itr, exectimes = rescal_als(data, K, init='nvecs', lambda_A=10, lambda_R=10, maxIter=self.iterations)

        self.log.info('Rescal fit info : ')
        print('fit: %s; itr: %s, exectimes: %s' % (fit, itr, exectimes))

        self._theta = A
        self._phi = R

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
        likelihood = self.likelihood()
        #likelihood[likelihood <= 0.5] = 0
        #likelihood[likelihood > 0.5] = 1
        #Y = likelihood
        Y = sp.stats.bernoulli.rvs(likelihood)
        return Y


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
