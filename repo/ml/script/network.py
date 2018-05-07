import numpy as np

from pymake import ExpeFormat
from pymake.manager import ModelManager, FrontendManager

from pymake.util.utils import colored



''' Build and Stats for Networds data '''



class Net(ExpeFormat):

    _default_expe = dict(
        _spec = 'data_net_all',
        _data_type = 'networks'
    )

    def __call__(self):
        return self.stats()

    def stats(self, fmt='simple'):
        ''' Show data stats '''
        expe = self.expe
        frontend = FrontendManager.load(expe)

        if self.is_first_expe():
            # Warning order sensitive @deprecated Table.
            #corpuses = self.specname(self.gramexp.get_set('corpus'))
            corpuses = self.specname(self.gramexp.get_list('corpus'))
            Meas = ['num_nodes', 'num_edges', 'density',
                    'is_directed', 'modularity', 'diameter', 'clustering_coefficient', 'net_type', 'feat_len']
            Meas_ = ['nodes', 'edges', 'density',
                     'directed', 'modularity', 'diameter', 'cluster-coef', 'weights', 'feat-len']
            Table = np.zeros((len(corpuses), len(Meas))) * np.nan
            Table = np.column_stack((corpuses, Table))
            self.D.Table = Table
            self.D.Meas = Meas
            self.D.Meas_ = Meas_

        Table = self.D.Table
        Meas = self.D.Meas

        for i, v in enumerate(Meas):
            if frontend.data is None:
                Table[self.corpus_pos, 1:] = np.nan
                break
            value = getattr(frontend, v)()
            value = value if value is not None else np.nan
            Table[self.corpus_pos, i+1] = value

        if hasattr(frontend, '_check'):
            frontend._check()


        if self.is_last_expe():
            tablefmt = 'latex' if fmt == 'tex' else fmt
            precision = '.5f'
            print(colored('\nStats Table :', 'green'))
            Meas_ = self.D.Meas_
            print(self.tabulate(Table, headers=Meas_, tablefmt=tablefmt, floatfmt=precision))


