#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import numpy as np
from numpy import ma
from pymake import ModelManager, FrontendManager, GramExp, ExpeFormat, ExpSpace


USAGE = """\
----------------
Fit the data :
----------------
"""

class Fit(ExpeFormat):


    def __call__(self):
        expe = self.expe

        t0 = time.time()

        frontend = FrontendManager.load(expe)

        # @Debug: Obsolete / Inside model
        alpha = expe.get('alpha', .5)
        gmma = expe.get('gmma', .5)
        delta = expe.get('delta', .5)

        hyperparams = {'alpha': alpha, 'delta': delta, 'gmma': gmma}
        expe['hyperparams'] = hyperparams

        ### Feature Porcessing
        #frontend.data = frontend.data.astype(float)
        #/

        model = ModelManager(expe=expe, frontend=frontend)
        model.fit(frontend)

        # Uhgh, trashed this !
        # here if save -> predict
        m = model.model.get_mask()

        model.predict(frontend=frontend)

        self.log.info('Expe %d finished in %.1f' % (self.pt['expe']+1, time.time()-t0))

if __name__ == '__main__':
    GramExp.generate().pymake(Fit)
