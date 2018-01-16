# -*- coding: utf-8 -*-


from ml.model.hdp.mmsb import GibbsRun, Likelihood, ZSampler, MSampler, BetaSampler, NP_CGS

from ml.model.hdp.mmsb import CGS, ZSamplerParametric

# @idem than ilda_cgs
class immsb_cgs(GibbsRun):
    def __init__(self, expe, frontend):

        delta = expe.hyperparams.get('delta',1)
        alpha = expe.hyperparams.get('alpha',1)
        gmma = expe.hyperparams.get('gmma',1)

        hyper = expe.hyper
        assortativity = expe.get('homo')
        hyper_prior = expe.get('hyper_prior') # HDP hyper optimization
        K = expe.K

        likelihood = Likelihood(delta, frontend, assortativity=assortativity)
        likelihood._symmetric = frontend.is_symmetric()

        # Nonparametric case
        zsampler = ZSampler(alpha, likelihood, K_init=K)
        msampler = MSampler(zsampler)
        betasampler = BetaSampler(gmma, msampler)
        jointsampler = NP_CGS(zsampler, msampler, betasampler,
                              hyper=hyper, hyper_prior=hyper_prior)

        self.s = jointsampler

        super().__init__(expe, frontend)
        self.update_hyper(expe.hyperparams)


class mmsb_cgs(GibbsRun):
    def __init__(self, expe, frontend):

        delta = expe.hyperparams.get('delta',1)
        alpha = expe.hyperparams.get('alpha',1)
        gmma = expe.hyperparams.get('gmma',1)

        hyper = expe.hyper
        assortativity = expe.get('homo')
        hyper_prior = expe.get('hyper_prior') # HDP hyper optimization
        K = expe.K

        likelihood = Likelihood(delta, frontend, assortativity=assortativity)
        likelihood._symmetric = frontend.is_symmetric()

        # Parametric case
        jointsampler = CGS(ZSamplerParametric(alpha, likelihood, K))

        self.s = jointsampler

        super().__init__(expe, frontend)
        self.update_hyper(expe.hyperparams)
