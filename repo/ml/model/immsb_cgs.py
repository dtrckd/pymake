# -*- coding: utf-8 -*-


from ml.model.hdp.mmsb import GibbsRun, Likelihood, ZSampler, NP_CGS
from ml.model.hdp.hdp import MSampler, BetaSampler

from ml.model.hdp.mmsb import CGS, ZSamplerParametric

# @idem than ilda_cgs
class immsb_cgs(GibbsRun):
    def __init__(self, expe, frontend):

        hyperparams = expe.get('hyperparams', {})
        delta = hyperparams.get('delta',1)
        alpha = hyperparams.get('alpha',1)
        gmma = hyperparams.get('gmma',1)

        hyper = expe.hyper
        assortativity = expe.get('homo')
        hyper_prior = expe.get('hyper_prior') # HDP hyper optimization
        K = expe.K

        likelihood = Likelihood(delta, frontend, assortativity=assortativity)

        if frontend:
            likelihood._symmetric = frontend.is_symmetric()

            # Nonparametric case
            zsampler = ZSampler(alpha, likelihood, K_init=K)
            msampler = MSampler(zsampler)
            betasampler = BetaSampler(gmma, msampler)
            jointsampler = NP_CGS(zsampler, msampler, betasampler,
                                  hyper=hyper, hyper_prior=hyper_prior)

            self.s = jointsampler
        else:
            likelihood._symmetric = None


        super().__init__(expe, frontend)
        self.update_hyper(hyperparams)


class mmsb_cgs(GibbsRun):
    def __init__(self, expe, frontend):

        hyperparams = expe.get('hyperparams', {})
        delta = hyperparams.get('delta',1)
        alpha = hyperparams.get('alpha',1)
        gmma = hyperparams.get('gmma',1)

        hyper = expe.hyper
        assortativity = expe.get('homo')
        hyper_prior = expe.get('hyper_prior') # HDP hyper optimization
        K = expe.K

        likelihood = Likelihood(delta, frontend, assortativity=assortativity)

        if frontend:
            likelihood._symmetric = frontend.is_symmetric()

            # Parametric case
            jointsampler = CGS(ZSamplerParametric(alpha, likelihood, K))

            self.s = jointsampler
        else:
            likelihood._symmetric = None

        super().__init__(expe, frontend)
        self.update_hyper(hyperparams)

