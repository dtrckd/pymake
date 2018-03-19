import time
import numpy as np
from numpy import ma
from pymake import GramExp, ExpeFormat


USAGE = """\
----------------
Fit the data :
----------------
"""

class Fit(ExpeFormat):

    def __call__(self):
        expe = self.expe
        t0 = time.time()

        # Load data
        frontend = self.load_frontend()

        ### @Debug: Obsolete / Inside model
        alpha = expe.get('alpha', .1)
        gmma = expe.get('gmma', .1)
        delta = expe.get('delta', (0.5, 0.5))
        hyperparams = {'alpha': alpha, 'delta': delta, 'gmma': gmma}
        expe['hyperparams'] = hyperparams
        #############################################################

        # Load Model
        model = self.load_model(frontend)

        # Fit
        model.fit()

        self.log.info('Expe %d finished in %.1f' % (self.pt['expe']+1, time.time()-t0))


    def fitw(self):
        if self._it == 0:
            self.gramexp.check_format()

        self.expe['_write'] = True
        self()


    def fit_missing(self, _type='pk'):

        is_fitted = self.gramexp.make_output_path(self.expe, _type=_type, status='f')
        if not is_fitted:
            self.expe['_write'] = True
            self()
        else:
            self.log.info("Expe `%s' already fitted, passing..." % self._it)


if __name__ == '__main__':
    GramExp.zymake().pymake(Fit)
