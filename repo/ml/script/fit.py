import time
import numpy as np
from numpy import ma
from pymake import GramExp, ExpeFormat
from pymake.frontend.manager import ModelManager, FrontendManager


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

        ### @Debug: Obsolete / Inside model
        alpha = expe.get('alpha', .1)
        gmma = expe.get('gmma', .1)
        delta = expe.get('delta', (0.5, 0.5))
        hyperparams = {'alpha': alpha, 'delta': delta, 'gmma': gmma}
        expe['hyperparams'] = hyperparams
        #############################################################


        self.model = ModelManager.from_expe_frontend(expe, frontend)
        self.configure_model(self.model)

        for i in range(1):
            self.model.fit()

        #model.predict(frontend=frontend)
        self.log.info('Expe %d finished in %.1f' % (self.pt['expe']+1, time.time()-t0))


    def fitw(self):
        if self._it == 0:
            self.gramexp.check_format()

        self.expe['write'] = True
        self()


    def fit_missing(self, _type='pk'):

        is_fitted = self.gramexp.make_output_path(self.expe, _type=_type, status='f')
        if not is_fitted:
            self.expe['write'] = True
            self()
        else:
            self.log.info("Expe `%s' already fitted, passing..." % self._it)


if __name__ == '__main__':
    GramExp.generate().pymake(Fit)
