# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import traceback
import numpy as np
from collections import OrderedDict
from pymake.plot import colored, display, tabulate

import logging
lgg = logging.getLogger('root')


def homo(self, _type='pearson', _sim='latent'):
    """ Hmophily test -- table output
        Parameters
        ==========
        _type: similarity type in (contengency, pearson)
        _sim: similarity metric in (natural, latent)
    """
    expe = self.expe
    Y = self._Y
    N = Y[0].shape[0]
    model = self.model

    lgg.info('using `%s\' type' % _type)
    lgg.info('using `%s\' similarity' % _sim)
    force_table_print = False

    # class / self !
    global Table

    if _type == 'pearson':
        # No variance for link expecation !!!
        Y = [Y[0]]

        Meas = [ 'pearson coeff', '2-tailed pvalue' ]; headers = Meas
        row_headers = _spec.name(self.gramexp.getCorpuses())
        Table = globals().get('Table', np.empty((len(row_headers), len(Meas), len(Y))))

        ### Global degree
        d, dc, yerr = random_degree(Y)
        sim = model.similarity_matrix(sim=_sim)
        #plot(sim, title='Similarity', sort=True)
        #plot_degree(sim)
        for it_dat, data in enumerate(Y):

            #homo_object = data
            homo_object = model.likelihood()

            Table[corpus_pos, :,  it_dat] = sp.stats.pearsonr(homo_object.flatten(), sim.flatten())

    elif _type == 'contingency':
        force_table_print = True
        Meas = [ 'esij', 'vsij' ]; headers = Meas
        row_headers = ['Non Links', 'Links']
        Table = globals().get('Table', np.empty((len(row_headers), len(Meas), len(Y))))

        ### Global degree
        d, dc, yerr = random_degree(Y)
        sim = model.similarity_matrix(sim=_sim)
        for it_dat, data in enumerate(Y):

            #homo_object = data
            homo_object = model.likelihood()

            Table[0, 0,  it_dat] = sim[data == 0].mean()
            Table[1, 0,  it_dat] = sim[data == 1].mean()
            Table[0, 1,  it_dat] = sim[data == 0].var()
            Table[1, 1,  it_dat] = sim[data == 1].var()

    if self._it == self.expe_size -1:
        # Function in (utils. ?)
        # Mean and standard deviation
        table_mean = np.char.array(np.around(Table.mean(2), decimals=3)).astype("|S20")
        table_std = np.char.array(np.around(Table.std(2), decimals=3)).astype("|S20")
        Table = table_mean + b' p2m ' + table_std

        # Table formatting
        Table = np.column_stack((row_headers, Table))
        tablefmt = 'latex' # 'latex'
        print()
        print( tabulate(Table, headers=headers, tablefmt=tablefmt, floatfmt='.3f'))
        del Table


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
        return enumerate(self)
    def table(self):
        return tabulate(self.items())

class ExpDesign(dict, BaseObject):
    _reserved_keywords = ['_mapname']

    # no more complex.
    # @sortbytype
    def table(self, line=10):
        glob_table = sorted([ (k, v) for k, v in self.items() if k not in self._reserved_keywords ], key=lambda x:x[0])

        Headers = OrderedDict((('Corpuses',Corpus),
                               ('Exp',(Expe, ExpTensor)),
                               ('Unknown',str)))
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
        return sep[1:]+sep.join(raw)

    def name(self, l):
        if not hasattr(self, '_mapname'):
            return l

        if isinstance(l, (set, list, tuple)):
            return [ self._mapname[i] for i in l ]
        else :
            try:
                return self._mapname[l]
            except:
                return l

class Corpus(list, BaseObject):
    def __add__(self, other):
        return Corpus(list.__add__(self, other))

class Expe(dict, BaseObject):
    pass

class ExpTensor(OrderedDict, BaseObject):
    def __init__(self,  *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        name = self.pop('_name', 'expTensor')
        BaseObject.__init__(self, name)

    def update_from_dict(self, d):
        for k, v in d.items():
            self[k] = [v]

    def table(self, extra=[]):
        return tabulate(extra+sorted(self.items(), key=lambda x:x[0]),
                               headers=['Params','Values'])
#\

class ExpeFormat(object):
    def __init__(self, pt, expe, gramexp):
        from pymake.expe.spec import _spec
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
                              _spec.name(expe.corpus),
                              _spec.name(expe.model),
                              expe.N, expe.K,))
        lgg.info('---')

    @classmethod
    def display(cls, conf):
        block = not conf.get('save_plot', False)
        display(block=block)

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
        return

    @classmethod
    def postprocess(cls, gramexp):
        # @here try to read the decorator that were called
        # * if @plot then dicplay
        # * if @tabulate then ...
        #   etc..
        cls.display(gramexp.expe)

    @staticmethod
    def plot(fun):
        def wrapper(*args, **kwargs):
            expe = args[0].expe
            f = fun(*args, **kwargs)
            if hasattr(expe, 'block_plot'):
                display(block=expe.block_plot)
            return f
        return wrapper


    def __call__(self):
        raise NotImplementedError

