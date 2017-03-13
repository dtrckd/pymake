#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy import ma
from pymake import ModelManager, FrontendManager, GramExp, ExpeFormat, ExpSpace

import logging
lgg = logging.getLogger('root')

USAGE = """\
----------------
Fit the data :
----------------
"""

class Fit(ExpeFormat):

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
        m = model.model.get_mask()

        model.predict(frontend=frontend)

if __name__ == '__main__':
    GramExp.generate().pymake(Fit)
