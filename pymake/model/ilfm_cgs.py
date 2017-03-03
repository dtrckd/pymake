# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
lgg = logging.getLogger('root')

from pymake.model.ibp.ilfm_gs import IBPGibbsSampling

class ilfm_cgs(IBPGibbsSampling):
    def __init__(self, expe, frontend):

        sigma_w = 1.
        sigma_w_hyper_parameter = None #(1., 1.)
        alpha = expe.hyperparams.get('alpha',1)

        alpha_hyper_parameter = expe.hyper
        assortativity = expe.get('homo')

        data = frontend.data
        data_t = frontend.data_t

        #if '_cgs' in expe.model:
        #    metropolis_hastings_k_new = False
        metropolis_hastings_k_new = True

        if assortativity == 2:
            raise NotImplementedError('Warning !: Metropolis Hasting not implemented with matrix normal. Exiting....')

        super(ilfm_cgs, self).__init__(assortativity,
                                       alpha_hyper_parameter,
                                       sigma_w_hyper_parameter,
                                       metropolis_hastings_k_new,
                                       iterations=expe.iterations,
                                       output_path=expe.output_path,
                                       write=expe.write)

        self._initialize(data, alpha, sigma_w, KK=expe.K)
        self.update_hyper(expe.hyperparams)

        lgg.warn('Warning: K is IBP initialized...')

        self.normalization_fun = lambda x : 1/(1 + np.exp(-x))

