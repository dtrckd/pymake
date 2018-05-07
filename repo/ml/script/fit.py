import time
import numpy as np
from numpy import ma
from pymake import GramExp, ExpeFormat



USAGE = """\
----------------
Fit the data : This script is part of the repo/ml of pymake.
----------------
"""



class Fit(ExpeFormat):

    def __call__(self):
        return self.fit()

    def fit(self):
        expe = self.expe
        t0 = time.time()

        # Load data
        frontend = self.load_frontend()

        ### @Debug: Obsolete / Inside model
        alpha = expe.get('alpha', .1)
        gmma = expe.get('gmma', .1)
        delta = expe.get('delta', (0.5, 0.5))
        hyperparams = {'alpha': alpha, 'delta': delta, 'gmma': gmma}
        expe['hyperparams'] = hyperparams
        #############################################################

        # Load Model
        model = self.load_model(frontend)

        # Fit
        model.fit(frontend)

        self.log.info('Expe %d finished in %.1f' % (self.pt['expe']+1, time.time()-t0))


    def fitw(self):
        if self._it == 0:
            self.gramexp.check_format()

        self.expe['_write'] = True
        self()


    def fit_missing(self, ext=None):

        is_fitted = self.gramexp.make_output_path(self.expe, ext=ext, status='f')
        if not is_fitted:
            self.fitw()
            #print(self.output_path)
        else:
            self.log.info("Expe `%s' already fitted, passing..." % self._it)

    def fit_incomplete(self, ext='inf'):

        is_fitted = self.gramexp.make_output_path(self.expe, ext=ext, status='f')
        _file = self.expe['_output_path'] +'.' + ext
        try:
            if not is_fitted:
                flag = False
            else:
                flag = list(filter(None,open(_file).read().split('\n')))[-1].split()[-1]

            is_incomplete = flag != 'terminated'
        except FileNotFoundError as e:
            is_incomplete = True

        if is_incomplete:
            self.fitw()
            #print(self.output_path)
        else:
            self.log.info("Expe `%s' completed, passing..." % self._it)


if __name__ == '__main__':
    GramExp.zymake().pymake(Fit)
