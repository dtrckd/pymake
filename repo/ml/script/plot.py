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
        self.model = ModelManager.from_expe(self.expe)

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

        data = data[attribute]
        data = np.ma.masked_invalid(np.array(data, dtype='float'))

        burnin = 5
        description = '/'.join((expe._refdir, os.path.basename(self.output_path)))

        ax = frame.ax()
        ax.plot(data, label=description, marker=frame.markers.next())
        ax.legend(loc='upper right',prop={'size':5})


    @ExpeFormat.raw_plot
    def plot_unique(self, attribute='_entropy'):
        ''' Plot all figure in the same window.
            Notes: likelihood/perplexity convergence report '''
        expe = self.expe

        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        data = data[attribute]
        data = np.ma.masked_invalid(np.array(data, dtype='float'))

        burnin = 5
        description = '/'.join((expe._refdir, os.path.basename(self.output_path)))

        plt.plot(data, label=description, marker=_markers.next())
        plt.legend(loc='upper right',prop={'size':1})


    #@ExpeFormat.plot(1,2) # improve ergonomy ?
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

        data = data[attribute]
        data = np.ma.masked_invalid(np.array(data, dtype='float'))

        burnin = 5
        description = '/'.join((expe._refdir, os.path.basename(self.output_path)))

        ax = frame.ax()
        ax.plot(data, label=description, marker=frame.markers.next())
        ax.legend(loc='upper right',prop={'size':5})


    #@ExpeFormat.table(1,2) # improve ergonomy ?
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
                print('attribute unknown: %s' % z)
                return
        else:
            data = data[z][-1]
            #data = np.ma.masked_invalid(np.array(data, dtype='float'))

        loc = floc(expe[x], expe[y], z)
        array[loc] = data



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


if __name__ == '__main__':
    GramExp.generate().pymake(Plot)
