# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


from pymake.model.hdp.mmsb import GibbsRun, Likelihood, CGS, ZSamplerParametric

# @idem than lda_cgs
class mmsb_cgs(GibbsRun):
    def __init__(self, expe, frontend):

        delta = expe.hyperparams.get('delta',1)
        alpha = expe.hyperparams.get('alpha',1)
        gmma = expe.hyperparams.get('gmma',1)

        hyper = expe.hyper
        assortativity = expe.get('homo')
        hyper_prior = expe.get('hyper_prior') # HDP hyper optimization
        K = expe.K

        data = frontend.data_ma #####
        data_t = frontend.data_t

        likelihood = Likelihood(delta, data, assortativity=assortativity)

        # Parametric case
        jointsampler = CGS(ZSamplerParametric(alpha, likelihood, K, data_t=data_t))

        super(mmsb_cgs, self).__init__(jointsampler,
                                    iterations=expe.iterations,
                                    output_path=expe.output_path,
                                    write=expe.write,
                                    data_t=data_t)
        self.update_hyper(expe.hyperparams)


