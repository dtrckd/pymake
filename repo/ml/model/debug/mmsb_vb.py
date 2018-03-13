import numpy as np
from scipy.special import psi

from ml.model.modelbase import ModelBase

class MMSB(object):
    """
    Mixed-membership stochastic block models, Airoldi et al., 2008

    """

    def __init__(self, expe, Y):
        """ follows the notations in the original NIPS paper

        :param Y: node by node interaction matrix, row=sources, col=destinations
        :param K: number of mixtures
        :param rho: sparsity parameter
        """
        self.N = int(Y.shape[0])    # number of nodes
        self.K = expe['K']
        self.Y = Y

        self.alpha = np.asarray([1.0 / (i + np.sqrt(self.K)) for i in range(self.K)])
        self.optimize_rho = False
        self.max_iter = expe['iterations']

        #variational parameters
        self.phi = np.random.dirichlet([1]*self.K, size=(self.N, self.N))

        self.gamma = np.random.dirichlet([1]*self.K, size=self.N)

        self.B = np.random.random(size=(self.K,self.K))

        self.rho = (1.-np.sum(self.Y)/(self.N*self.N))  # 1 - density of matrix

    def variational_inference(self, converge_ll_fraction=1e-3):
        """ run variational inference for this model
        maximize the evidence variational lower bound

        :param converge_ll_fraction: terminate variational inference when the fractional change of the lower bound falls below this
        """
        converge = False
        old_ll = 0.
        iteration = 0

        while not converge:
            ll = self.run_e_step()
            ll += self.run_m_step()

            iteration += 1

            #if (old_ll-ll)/ll < converge_ll_fraction or iteration >= self.max_iter:
            if iteration >= self.max_iter:
                converge = True

            print('iteration %d, lower bound %.2f' %(iteration, ll))


    def run_e_step(self):
        """ compute variational expectations
        """
        ll = 0.

        for p in range(self.N):
            for q in range(self.N):
                new_phi = np.zeros(self.K)

                for g in range(self.K):
                    new_phi[g] = np.exp(psi(self.gamma[p,g])-psi(np.sum(self.gamma[p,:]))) * np.prod(( (self.B[g,:]**self.Y[p,q])
                        * ((1.-self.B[g,:])**(1.-self.Y[p,q])) )
                        ** self.phi[q,p,:] )
                self.phi[p,q,:] = new_phi/np.sum(new_phi)

                new_phi = np.zeros(self.K)
                for h in range(self.K):
                    new_phi[h] = np.exp(psi(self.gamma[q,h])-psi(np.sum(self.gamma[q,:]))) * np.prod(( (self.B[:,h]**self.Y[p,q])
                        * ((1.-self.B[:,h])**(1.-self.Y[p,q])) )
                        ** self.phi[p,q,:] )
                self.phi[q,p,:] = new_phi/np.sum(new_phi)

                for k in range(self.K):
                    self.gamma[p,k] = self.alpha[k] + np.sum(self.phi[p,:,k]) + np.sum(self.phi[:,p,k])
                    self.gamma[q,k] = self.alpha[k] + np.sum(self.phi[q,:,k]) + np.sum(self.phi[:,q,k])

        return ll

    def run_m_step(self):
        """ maximize the hyper parameters
        """
        ll = 0.

        self.optimize_alpha()
        self.optimize_B()
        if self.optimize_rho:
            self.update_rho()

        return ll

    def optimize_alpha(self):
        return

    def optimize_B(self):
        for g in range(self.K):
            for h in range(self.K):
                tmp1=0.
                tmp2=0.
                for p in range(self.N):
                    for q in range(self.N):
                        tmp = self.phi[p,q,g]*self.phi[q,p,h]
                        tmp1 += self.Y[p,q]*tmp
                        tmp2 += tmp
                self.B[g,h] = tmp1/tmp2
        return

    def update_rho(self):
        return

class mmsb_vb(ModelBase):

    def __init__(self, expe, frontend):
        super().__init__(expe, frontend)

        self.expe = expe

        self.frontend = frontend
        self.mask = self.frontend.data_ma.mask

    def fit(self):
        data = self.frontend.data_ma
        K = self.expe.K

        model = MMSB(self.expe, data)
        model.variational_inference()

        self._theta = model.gamma
        self._phi = model.B


if __name__ == '__main__':

    """ test with interaction matrix:
    1 1 0 0 0
    1 1 0 0 0
    0 0 1 1 1
    0 0 1 1 1
    0 0 1 1 1
    """
    Y = np.array([[1,1,0,0,0],[1,1,0,0,0],[0,0,1,1,1],[0,0,1,1,1],[0,0,1,1,1]])
    model = MMSB(Y, 2)
    model.variational_inference()

    print(model.gamma)
    print(model.B)



