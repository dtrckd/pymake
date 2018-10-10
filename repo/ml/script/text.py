import numpy as np
#np.set_printoptions(threshold='nan')

from pymake import ExpeFormat
from pymake.manager import ModelManager, FrontendManager

from pymake.util.utils import colored



''' Build and Stats for Text data '''


class Text(ExpeFormat):

    _default_expe = dict(
        _spec = 'data_text_all',
        _data_type = 'text'
    )

    def __call__(self):
        return self.stats()

    def stats(self):
        ''' Show data stats '''
        raise NotImplementedError
