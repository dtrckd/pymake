# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import traceback
import numpy as np
from collections import OrderedDict, defaultdict
from pymake.plot import colored, display, tabulate
from decorator import decorator
from functools import wraps
from pymake import basestring

import matplotlib.pyplot as plt

import logging
lgg = logging.getLogger('root')


class ExpSpace(dict):
    def __init__(self, *args, **kwargs):
        super(ExpSpace, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    """dot.notation access to dictionary attributes"""
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
    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


### Base Object
#
# @todo integrate Frontend and Model to that workflow
#


class BaseObject(object):
    '''' Notes : Avoid method conflict by ALWAYS settings this class in last
                 at class definitions.
     '''
    def __init__(self, name):
        # Le ruban est infini...
        #if name is None:
        #    print(traceback.extract_stack()[-2])
        #    fn,ln,func,text = traceback.extract_stack()[-2]
        #    name = text[:text.find('=')].strip()
        #else:
        #    name = '<undefined>'
        self.__name__ = name

    def name(self):
        return self.__name__
    def items(self):
        return [(str(i), j) for i,j in enumerate(self)]
    def table(self):
        return tabulate(self.items())

class ExpDesign(dict, BaseObject):
    _reserved_keywords = ['_mapname', '_name', '_reserved_keywords']+dir(dict)+dir(BaseObject) # _* ?
    def __init__(self,  *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        name = self.pop('_name', 'expDesign')
        BaseObject.__init__(self, name)

        for k in dir(self):
            #_spec = ExpDesign((k, getattr(Netw, k)) for k in dir(Netw) if not k.startswith('__') )
            if not k.startswith('__'):
                self[k] = getattr(self, k)

    # no more complex.
    # @sortbytype
    def table(self, line=10):
        glob_table = sorted([ (k, v) for k, v in self.items() if k not in self._reserved_keywords ], key=lambda x:x[0])

        Headers = OrderedDict((('Corpuses',Corpus),
                               ('Exp',(Expe, ExpTensor)),
                               ('Unknown', str)))
        tables = [ [] for i in range(len(Headers))]
        for name, _type in glob_table:
            try:
                pos = [isinstance(_type, v) for v in Headers.values()].index(True)
            except ValueError:
                pos = len(Headers) - 1
            tables[pos].append(name)

        raw = []
        for sec, table in enumerate(tables):
            size = len(table)
            if size == 0:
                continue
            col = int((size-0.1) // line)
            junk = line % size
            table += ['-']*junk
            table = [table[j:line*(i+1)] for i,j in enumerate(range(0, size, line))]
            table = np.char.array(table).astype("|S20")
            fmt = 'simple'
            raw.append(tabulate(table.T,
                                headers=[list(Headers.keys())[sec]]+['']*(col),
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

class ExpVector(list, BaseObject):
    def __add__(self, other):
        return self.__class__(list.__add__(self, other))
    def __sub__(self, other):
        return self.__class__([item for item in self if item not in other])

class Corpus(ExpVector):
    pass

class Model(ExpVector):
    pass

class Expe(dict, BaseObject):
    pass

class ExpTensor(OrderedDict, BaseObject):
    def __init__(self,  *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        name = self.pop('_name', 'expTensor')
        BaseObject.__init__(self, name)

    def update_from_dict(self, d):
        for k, v in d.items():
            if issubclass(type(v), ExpVector):
                self[k] = v
            else:
                self[k] = [v]

    def table(self, extra=[]):
        return tabulate(extra+sorted(self.items(), key=lambda x:x[0]),
                               headers=['Params','Values'])
#\


class ExpeFormat(object):
    def __init__(self, pt, expe, gramexp):
        from pymake.expe.spec import _spec
        self.specname = _spec.name
        # Global
        self.expe_size = len(gramexp)
        self.gramexp = gramexp
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
    def preprocess(cls, gramexp):
        # @here try to read the decorator that were called
        # * if @plot then dicplay
        # * if @tabulate then ...
        #   etc..
        if 'save_plot' in gramexp.expe:
            #import matplotlib; matplotlib.use('Agg')
            # killu
            pass

        # Put a valid expe a the end.
        gramexp.reorder_lastvalid()

        return

    @classmethod
    def postprocess(cls, gramexp):
        # @here try to read the decorator that were called
        # * if @plot then dicplay
        # * if @tabulate then ...
        #   etc..
        cls.display(gramexp.expe)

    def __call__(self):
        raise NotImplementedError

