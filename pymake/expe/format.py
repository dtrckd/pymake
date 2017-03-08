# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import traceback
import numpy as np
from collections import OrderedDict, defaultdict
from pymake.plot import colored, display, tabulate
from decorator import decorator
from functools import wraps
from pymake import basestring
import pymake as pmk

import matplotlib.pyplot as plt

import logging
lgg = logging.getLogger('root')

# Not sure this one is necessary, or not here
class BaseObject(object):
    ''' Notes : Avoid method conflict by ALWAYS settings this class in last
                at class definitions.
    '''
    def __init__(self, name):
        # Le ruban est infini...
        #if name is None:
        #    print(traceback.extract_stack()[-2])
        #    fn,ln,func,text = traceback.extract_stack()[-2]
        #    name = text[:text.find('=')].strip()
        self.__name__ = name

    def name(self):
        return self.__name__
    def items(self):
        return [(str(i), j) for i,j in enumerate(self)]
    def table(self):
        return tabulate(self.items())

class ExpSpace(dict):
    """ A dictionnary with dot notation access.
        Used for the **expe** settings stream.
    """
    def __init__(self, *args, **kwargs):
        super(ExpSpace, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    # For Piclking
    def __getstate__(self):
        return self
    #def __setstate__(self, state):
    #    self.update(state)
    #    self.__dict__ = self

class ExpVector(list, BaseObject):
    ''' A List of elements of an ExpTensor. '''
    def __add__(self, other):
        return self.__class__(list.__add__(self, other))
    def __sub__(self, other):
        return self.__class__([item for item in self if item not in other])

class Corpus(ExpVector):
    @staticmethod
    def get_atoms(spec):
        # get some information about what package to use in _spec ...
        atoms = pmk.frontend.get_packages('pymake.data')
        return atoms

class Model(ExpVector):
    @staticmethod
    def get_atoms(spec, _type='short'):
        if _type == 'short':
            shrink_module_name = True
        elif _type == 'topos':
            shrink_module_name = False

        from pymake.util.loader import ModelsLoader
        packages = spec._package.get('model',[])
        if 'pymake.model' in packages:
            atoms = ModelsLoader.get_packages(packages.pop(packages.index('pymake.model')), prefix='pmk')
        else:
            atoms = OrderedDict
        for pkg in packages:
            if len(pkg) > 8:
                prefix = pkg[:3]
                if '.' in pkg:
                    prefix  += ''.join(map(lambda x:x[0], pkg.split('.')[1:]))
            else:
                prefix = True
            atoms.update(ModelsLoader.get_packages(pkg,  prefix=prefix, max_depth=3, shrink_module_name=shrink_module_name))
        return atoms


class ExpTensor(OrderedDict, BaseObject):
    ''' Represent a set of Experiences (**expe**). '''
    def __init__(self,  *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        name = self.pop('_name', 'expTensor')
        BaseObject.__init__(self, name)

    @classmethod
    def from_expe(cls, expe):
        ''' Return the tensor who is an Orderedict of iterable.
            Assume conf is an exp. Non list value will be listified.

            Parameters
            ----------
            expe : (ExpDesign, ExpSpace or dict)
                A design of experiment.
        '''
        _conf = None
        if 'spec' in expe:
            _conf = expe.copy()
            expe = _conf.pop('spec')

        if not isinstance(expe, (cls, ExpSpace, dict)):
            raise ValueError('Expe not understood: %s' % type(expe))

        if issubclass(type(expe), Corpus):
            tensor = cls(corpus=expe)
        elif issubclass(type(expe), Model):
            tensor = cls(model=expe)
        elif not isinstance(expe, ExpTensor):
            tensor = cls()
            tensor.update_from_dict(expe)
        else:
            # ExpTensor or not implemented expVector
            tensor = expe.copy()

        for k, v in tensor.items():
            if not issubclass(type(v), (list, set, tuple)):
                tensor[k] = [v]

        if _conf:
            tensor.update_from_dict(_conf)

        return tensor

    def update_from_dict(self, d):
        for k, v in d.items():
            if issubclass(type(v), ExpVector):
                self[k] = v
            else:
                self[k] = [v]

    def table(self, extra=[]):
        return tabulate(extra+sorted(self.items(), key=lambda x:x[0]),
                               headers=['Params','Values'])

class ExpDesign(dict, BaseObject):
    ''' An Ensemble composed of ExpTensors and ExpVectors.

        NOTES
        -----
        Special attribute meaning:
            _mapname : dict
                use when self.name is called to translate keywords
            _alias : dict
                command line alias
            _model_package : list of str
                where to get models
            _corpus_package : list of str
                where to get corpus
    '''
    def __init__(self,  *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

        # Not a Ultimate solution to keep a flexibility when defining Exp Design
        for k in dir(self):
            #_spec = ExpDesign((k, getattr(Netw, k)) for k in dir(Netw) if not k.startswith('__') )
            if not k.startswith('_'):
                self[k] = getattr(self, k)
        self._reserved_keywords = list(set([w for w in dir(self) if w.startswith('_')] + ['_reserved_keywords']+dir(dict)+dir(BaseObject)))
        name = self.pop('_name', 'expDesign')

        BaseObject.__init__(self, name)


    # no more complex.
    # @sortbytype
    def _table(self):
        glob_table = sorted([ (k, v) for k, v in self.items() if k not in self._reserved_keywords ], key=lambda x:x[0])

        Headers = OrderedDict((('Corpuses',Corpus),
                               ('Exp',(ExpSpace, ExpTensor)),
                               ('Unknown', str)))
        tables = [ [] for i in range(len(Headers))]
        for name, _type in glob_table:
            try:
                pos = [isinstance(_type, v) for v in Headers.values()].index(True)
            except ValueError:
                pos = len(Headers) - 1
            tables[pos].append(name)

        return self._table_(tables, headers=list(Headers.keys()))

    def _table_atoms(self, _type='short'):

        Headers = OrderedDict((('Corpuses',Corpus),
                               ('Models',(ExpSpace, ExpTensor)),
                               ('Unknown', str)))

        tables = [[], # corpus atoms...
                  list(Model.get_atoms(self, _type).keys()),
        ]

        return self._table_(tables, headers=list(Headers.keys()))

    def _table_(self, tables, headers=[], max_line=10, max_row=30):
        raw = []
        for sec, table in enumerate(tables):
            table = sorted(table, key=lambda x:x[0])
            size = len(table)
            if size == 0:
                continue
            col = int((size-0.1) // max_line)
            junk = max_line % size
            table += ['-']*junk
            table = [table[j:max_line*(i+1)] for i,j in enumerate(range(0, size, max_line))]
            table = np.char.array(table).astype('|S'+str(max_row))
            fmt = 'simple'
            raw.append(tabulate(table.T,
                                headers=[headers[sec]]+['']*(col),
                                tablefmt=fmt))
        sep = '\n'+'='*20+'\n'
        return '#'+self.__name__ +sep+sep.join(raw)

    def name(self, l):
        if '_mapname' in self:
            mapname =  self['_mapname']
        else:
            return l

        if isinstance(l, (set, list, tuple)):
            return [ mapname.get(i, i) for i in l ]
        elif isinstance(l, (dict, ExpSpace)):
            d = dict(l)
            for k, v in d.items():
                if isinstance(v, basestring) and v in mapname:
                    d[k] = mapname[v].replace(' ', '').lower()
            return d
        else :
            return mapname.get(l, l)


### Todo:
#  try to read the decorator that were called for _[post|pre]process
# * if @plot then display
# * if @tabulate then ...
class ExpeFormat(object):
    ''' A Base class for processing individuals experiments (**expe**).

        Notes
        -----
        The following attribute have a special meaning when subclassing:
            * _default_expe : is updated by each single expe.

    '''
    def __init__(self, pt, expe, gramexp):
        # Global
        self.expe_size = len(gramexp)
        self.gramexp = gramexp
        self.specname = gramexp.getSpec().name
        # Local
        self.pt = pt
        self.expe = expe

        # to exploit / Vizu
        self._it = pt['expe']
        self.corpus_pos = pt['corpus']
        self.model_pos = pt['model']

        lgg.info('---')
        lgg.info(''.join([colored('Expe %d/%d', 'red'),
                          ' : %s -- %s -- N=%s -- K=%s']) % (
                              self._it+1, self.expe_size,
                              self.specname(expe.corpus),
                              self.specname(expe.model),
                              expe.N, expe.K,))
        lgg.info('---')

    @classmethod
    def display(cls, conf):
        block = not conf.get('save_plot', False)
        display(block=block)

    @staticmethod
    @decorator
    def plot_simple(fun, *args, **kwargs):
        self = args[0]
        expe = self.expe
        kernel = fun(*args, **kwargs)
        if hasattr(expe, 'block_plot') and getattr(self, 'noplot', False) is not True:
            display(block=expe.block_plot)
        return kernel

    @staticmethod
    def plot(*groups):
        ''' If no argument, simple plot.
            If arguments :
                * [0] : group figure by this
                * [1] : key for id (title and filename)
        '''
        if len(groups[1:]) == 0 and callable(groups[0]):
            # decorator whithout arguments
            return ExpeFormat.plot_simple(groups[0])
        else:
            def decorator(fun):
                @wraps(fun)
                def wrapper(*args, **kwargs):
                    group = groups[0]
                    self = args[0]
                    expe = self.expe

                    # Init Figs Sink
                    if not hasattr(self.gramexp, 'figs'):
                        figs = dict()
                        for c in self.gramexp.get(group, []):
                            figs[c] = ExpSpace()
                            figs[c].fig = plt.figure()
                        self.gramexp.figs = figs

                    kernel = fun(*args, **kwargs)

                    # Set title and filename
                    title = ' '.join('{{{0}}}'.format(w) for w in groups).format(**self.specname(expe))
                    expfig = self.gramexp.figs[expe[group]]
                    expfig.fn = '%s_%s' % (fun.__name__, title.replace(' ', '_'))
                    expfig.fig.gca().set_title(title)

                    # Save on last call
                    if self._it == self.expe_size -1:
                        if expe.write:
                            from private import out
                            out.write_figs(expe, self.gramexp.figs)

                    return kernel
                return wrapper
        return decorator

    @staticmethod
    #tensor in decorator to build SINK thiis th new tensor !!!!
    def tabulate(*groups):
        ''' TODO
        '''
        if len(groups[1:]) == 0 and callable(groups[0]):
            # decorator whithout arguments
            return groups[0]
        else:
            row = groups[0]
            column = groups[1]
            def decorator(fun):
                @wraps(fun)
                def wrapper(*args, **kwargs):
                    kernel = fun(*args, **kwargs)
                    return kernel
                return wrapper
        return decorator

    @classmethod
    def _preprocess(cls, gramexp):
        ''' This method has **write** access to Gramexp '''

        # Put a valid expe a the end.
        gramexp.reorder_lastvalid()

        # update exp_tensor in gramexp
        if hasattr(cls, '_default_expe'):
            _exp = ExpTensor.from_expe(cls._default_expe)
            _exp.update(gramexp.exp_tensor)
            gramexp.exp_setup(_exp)

        return cls.preprocess(gramexp)


    @classmethod
    def _postprocess(cls, gramexp):
        cls.display(gramexp.exp_tensor)

        return cls.postprocess(gramexp)

    @classmethod
    def preprocess(cls, gramexp):
        # heere, do a wrapper ?
        pass
    @classmethod
    def postprocess(cls, gramexp):
        # heere, do a wrapper ?
        pass

    def __call__(self):
        raise NotImplementedError

