# -*- coding: utf-8 -*-

import logging
lgg = logging.getLogger('root')

from ml.model.ibp.ilfm_gs import IBPGibbsSampling

class ilfm_cgs(IBPGibbsSampling):
    def __init__(self, expe, frontend):

        hyperparams = expe.get('hyperparams', {})
        sigma_w = 1.
        sigma_w_hyper_parameter = None #(1., 1.)
        alpha = hyperparams.get('alpha',1)

        alpha_hyper_parameter = expe.hyper
        assortativity = expe.get('homo')

        #if '_cgs' in expe.model:
        #    metropolis_hastings_k_new = False
        metropolis_hastings_k_new = True

        if assortativity == 2:
            raise NotImplementedError('Warning !: Metropolis Hasting not implemented with matrix normal. Exiting....')

        super(ilfm_cgs, self).__init__(expe, frontend,
                                       assortativity,
                                       alpha_hyper_parameter,
                                       sigma_w_hyper_parameter,
                                       metropolis_hastings_k_new)

        self._initialize(frontend, alpha, sigma_w, KK=expe.K)
        self.update_hyper(hyperparams)

        lgg.warn('Warning: K is IBP initialized...')

        self.normalization_fun = lambda x : 1/(1 + np.exp(-x))

class lfm_cgs(ilfm_cgs):
    def __init__(self, expe, frontend):
        super().__init__(expe, frontend)

        self.log.warning(''' Random initializatin intead of ibp ? ''')
        self._metropolis_hastings_k_new = False



