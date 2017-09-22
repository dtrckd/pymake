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

    def init_run(self):
        # setup seed ?
        # setup branch ?
        # setup description ?
        if self.expe.get('write'):
            self.init_fitfile()

    def _preprocess(self):
        pass

    def _postprocess(self):
        if self.expe.get('write'):
            if hasattr(self, 'model'):
                self.clear_fitfile()
                self.model.save()


    def __call__(self):
        self.init_run()
        expe = self.expe

        t0 = time.time()

        frontend = FrontendManager.load(expe)

        # @Debug: Obsolete / Inside model
        alpha = expe.get('alpha', .1)
        gmma = expe.get('gmma', .1)
        delta = expe.get('delta', .5)

        hyperparams = {'alpha': alpha, 'delta': delta, 'gmma': gmma}
        expe['hyperparams'] = hyperparams

        ### Feature Porcessing
        #frontend.data = frontend.data.astype(float)
        #/

        self.model = ModelManager.from_expe_frontend(expe, frontend)
        self.configure_model(self.model)

        for i in range(1):
            self.model.fit()


        #model.predict(frontend=frontend)

        self.log.info('Expe %d finished in %.1f' % (self.pt['expe']+1, time.time()-t0))

    def fit_missing(self, _type='pk'):

        is_fitted = self.gramexp.make_output_path(self.expe, _type=_type, status='f')
        if not is_fitted:
            self()
        else:
            self.log.info("Expe `%s' already fitted, passing..." % self._it)


if __name__ == '__main__':
    GramExp.generate().pymake(Fit)
