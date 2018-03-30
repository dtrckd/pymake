import os
import numpy as np
from numpy import ma
from pymake import GramExp, ExpeFormat
from pymake.frontend.manager import ModelManager, FrontendManager
from pymake.plot import _markers, _colors, _linestyle

import matplotlib.pyplot as plt

import logging
lgg = logging.getLogger('root')


USAGE = """\
----------------
Plot utility :
----------------
"""


class Plot(ExpeFormat):

    def _preprocess(self):
        self.model = ModelManager.from_expe(self.expe, load=True)

    @ExpeFormat.raw_plot('corpus')
    def __call__(self, frame,  attribute='_entropy'):
        ''' Plot figure group by :corpus:.
            Notes: likelihood/perplexity convergence report
        '''
        expe = self.expe

        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        values = data[attribute]
        values = self._to_masked(values)

        description = '/'.join((expe._refdir, os.path.basename(self.output_path)))

        ax = frame.ax()
        ax.plot(values, label=description, marker=frame.markers.next())
        ax.legend(loc='upper right',prop={'size':5})

    def _to_masked(self, lst, dtype=float):
        return np.ma.masked_invalid(np.array(lst, dtype=dtype))


    @ExpeFormat.raw_plot
    def plot_unique(self, attribute='_entropy'):
        ''' Plot all figure in the same window.
            Notes: likelihood/perplexity convergence report '''
        expe = self.expe

        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        values = data[attribute]
        values = self._to_masked(values)

        description = '/'.join((expe._refdir, os.path.basename(self.output_path)))

        plt.plot(values, label=description, marker=_markers.next())
        plt.legend(loc='upper right',prop={'size':1})


    #@ExpeFormat.plot(1,2) # improve ergonomy ?
    @ExpeFormat.plot()
    def fig(self, frame, attribute, *args):
        ''' Plot all figure args is  `a:b..:c' (plot c by grouping by a, b...),
            if args is given, use for filename discrimination `key1[/key2]...'.
        '''
        expe = self.expe

        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        try:
            values = data[attribute]
        except KeyError as e:
            func_name = 'get_'+attribute
            if hasattr(self, func_name):
                values = getattr(self, func_name)(data)
            else:
                self.log.error('attribute unknown: %s' % attribute)
                return

        values = self._to_masked(values)

        burnin = 0
        description = self.get_description()

        ax = frame.ax()

        if expe.get('fig_xaxis'):
            xaxis = expe['fig_xaxis']
            if isinstance(xaxis, (tuple, list)):
                xaxis_name = xaxis[0]
                xaxis_surname = xaxis[1]
            else:
                xaxis_name = xaxis_surname = xaxis
            x = np.array(data[xaxis_name], dtype=int)
            xmax = frame.get('xmax',[])
            xmin = frame.get('xmin',[])
            xmax.append(max(x))
            xmax.append(min(x))
            frame.xmax = xmax
            frame.xmin = xmin
            frame.xaxis = xaxis_surname
        else:
            x = range(len(values))

        ax.plot(x[burnin:], values, label=description, marker=frame.markers.next())
        ax.legend(loc=expe.get('fig_legend',1), prop={'size':5})

        #if self.is_last_expe() and expe.get('fig_xaxis'):
        #    for frame in self.get_figs():
        #        xmax = max(frame.xmax)
        #        xmin = min(frame.xmin)
        #        xx = np.linspace(0,xmax, 10).astype(int)
        #        ax = frame.ax()
        #        ax.set_xticks(x)


    #@ExpeFormat.table(1,2) # improve ergonomy ?
    @ExpeFormat.table()
    def table(self, array, floc, x, y, z, *args):
        ''' Plot table according to parameter `x:y:z[-z2](param)'
            if args is given, use for filename discrimination `key1[/key2]...'
        '''
        expe = self.expe
        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)

        if not data or z not in data:
            if not self.model:
                self.log.warning('No model for expe : %s' % self.output_path)
                return

            func_name = 'get_'+z
            if hasattr(self, func_name):
                data = getattr(self, func_name)()
            else:
                self.log.error('attribute unknown: %s' % z)
                return
        else:
            data = data[z][-1]
            #data = np.ma.masked_invalid(np.array(data, dtype='float'))

        loc = floc(expe[x], expe[y], z)
        array[loc] = data


    #
    # ml/model specific measure.
    #


    def get_roc(self, _ratio=100):
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        expe = self.expe
        model = self.model

        frontend = FrontendManager.load(expe)
        data = frontend.data

        _ratio = int(_ratio)
        _predictall = (_ratio >= 100) or (_ratio < 0)
        if not hasattr(expe, 'testset_ratio'):
            setattr(expe, 'testset_ratio', 20)

        y_true, probas = model.mask_probas(data)
        theta, phi = model.get_params()

        try:
            fpr, tpr, thresholds = roc_curve(y_true, probas)
        except Exception as e:
            print(e)
            self.log.error('cant format expe : %s' % (self.output_path))
            return

        roc_auc = auc(fpr, tpr)
        return roc_auc

    def get_perplexity(self, data):
        entropy = self._to_masked(data['_entropy'])
        testset = self.model._data_test
        nnz = testset.shape[0]
        return 2 ** (-entropy / nnz)


    def get_wsim(self):
        expe = self.expe
        #frontend = self.frontend
        model = self.model

        #y = model.generate(**expe)
        theta, phi = model.get_params()
        # NB mean/var
        mean, var = model.get_nb_ss()

        phi = mean
        nnz = self.model._data_test.shape[0]

        sim = []
        for i,j,_w in model._data_test:
            w = np.random.poisson(theta[i].dot(phi).dot(theta[j].T))
            sim.append(w)

        # l1 norm
        mean_dist = np.abs(np.array(w) - model._data_test[:,2].T).sum() / nnz
        return mean_dist





if __name__ == '__main__':
    GramExp.generate().pymake(Plot)
