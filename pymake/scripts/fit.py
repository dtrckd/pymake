#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy import ma
from pymake import ExpTensor, ModelManager, FrontendManager, GramExp, ExpeFormat, ExpSpace

import logging
lgg = logging.getLogger('root')
_spec = GramExp.Spec()

USAGE = """\
----------------
Fit the data :
----------------
"""

class Fit(ExpeFormat):

    _default_expe = ExpSpace(
        corpus = 'clique2',
        model  = 'immsb_cgs',
        hyper       = 'auto',
        refdir      = 'debug',
        testset_ratio = 0.2,
        K           = 3,
        N           = 42,
        chunk       = 10000,
        iterations  = 3,
        homo        = 0, # learn W in IBP
    )

    @classmethod
    def preprocess(cls, gramexp):
        lgg.info(gramexp.exptable())

    def __call__(self):
        expe = self.expe
        frontend = FrontendManager.load(expe)

        # @Debug: Obsolete / Inside model
        alpha = expe.get('alpha', .1)
        gmma = expe.get('gmma', .5)
        delta = expe.get('delta', 1.)

        hyperparams = {'alpha': alpha, 'delta': delta, 'gmma': gmma}
        expe['hyperparams'] = hyperparams

        model = ModelManager(expe=expe, frontend=frontend)
        model.fit(frontend)

        # here if save -> predict
        model.predict(frontend=frontend)

