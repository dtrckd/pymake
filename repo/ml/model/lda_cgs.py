# -*- coding: utf-8 -*-


from ml.model.hdp.lda import GibbsRun, Likelihood, CGS, ZSamplerParametric

# @idem than mmsb_cgs
class lda_cgs(GibbsRun):
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

        # Parametric case
        jointsampler = CGS(ZSamplerParametric(alpha, likelihood, K, data_t=data_t))

        super(lda_cgs, self).__init__(jointsampler,
                                    iterations=expe.iterations,
                                    output_path=expe._output_path,
                                    write=expe._write,
                                    data_t=data_t)
        self.update_hyper(expe.hyperparams)


