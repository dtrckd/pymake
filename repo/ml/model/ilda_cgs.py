# -*- coding: utf-8 -*-


from ml.model.hdp.lda import GibbsRun, Likelihood, ZSampler,  NP_CGS
from ml.model.hdp.hdp import MSampler, BetaSampler

# @idem than immsb_cgs
class ilda_cgs(GibbsRun):
    def __init__(self, expe, frontend):

        delta = expe.hyperparams.get('delta',1)
        alpha = expe.hyperparams.get('alpha',1)
        gmma = expe.hyperparams.get('gmma',1)

        hyper = expe.hyper
        assortativity = expe.get('homo')
        hyper_prior = expe.get('hyper_prior') # HDP hyper optimization
        K = expe.K

        data = frontend.data
        data_t = frontend.data_t

        likelihood = Likelihood(delta, data, assortativity=assortativity)

        # Nonparametric case
        zsampler = ZSampler(alpha, likelihood, K_init=K, data_t=data_t)
        msampler = MSampler(zsampler)
        betasampler = BetaSampler(gmma, msampler)
        jointsampler = NP_CGS(zsampler, msampler, betasampler,
                              hyper=hyper, hyper_prior=hyper_prior)

        super(ilda_cgs, self).__init__(expe, frontend,
                                       jointsampler,
                                       iterations=expe.iterations,
                                       output_path=expe._output_path,
                                       write=expe._write,
                                       data_t=data_t)
        self.update_hyper(expe.hyperparams)


