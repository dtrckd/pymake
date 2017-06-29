#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy import ma
from pymake import ModelManager, FrontendManager, GramExp, ExpeFormat, ExpSpace

USAGE = """\
----------------
Fit the data :
----------------
"""

class FitIndex(ExpeFormat):

    _default_expe = ExpSpace(
        data_type   = 'networks',
        corpus      = 'clique2',
        model       = 'immsb_cgs',
        hyper       = 'auto',
        refdir      = 'debug',
        testset_ratio = 0.2,
        K           = 3,
        N           = 42,
        chunk       = 10000,
        iterations  = 3,
        homo        = 0, # learn W in IBP
    )


    def __call__(self):
        expe = self.expe
        frontend = FrontendManager.load(expe)

        # @Debug: Obsolete / Inside model
        alpha = expe.get('alpha', .1)
        gmma = expe.get('gmma', .5)
        delta = expe.get('delta', .5)

        hyperparams = {'alpha': alpha, 'delta': delta, 'gmma': gmma}
        expe['hyperparams'] = hyperparams

        ### Feature Porcessing
        frontend.data = frontend.data.astype(float)
        #/

        model = ModelManager(expe=expe, frontend=frontend)
        model.fit(frontend)

        # here if save -> predict
        model.predict(frontend=frontend)

if __name__ == '__main__':
    GramExp.generate().pymake(FitIndex)
