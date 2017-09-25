# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import logging
from copy import copy
import traceback,  importlib
import numpy as np
from collections import OrderedDict, defaultdict
from decorator import decorator
from functools import wraps
from itertools import product

from pymake.util.utils import colored, basestring, get_dest_opt_filled, make_path, hash_objects, ask_sure_exit
from pymake.index.indexmanager import IndexManager as IX

lgg = logging.getLogger('root')




''' Structure of Pymake Objects.

This is what Pandas does ? So here is a Lama.
'''


from tabulate import tabulate

# Ugly, integrate.
def _table_(tables, headers=[], max_line=10, max_row=30, name=''):

    if isinstance(headers, str):
        sep = '# %s'%name +  '\n'+'='*20
        print(sep)
        return tabulate(tables, headers=headers)


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
    return sep[1:] + sep.join(raw)




# Not sure this one is necessary, or not here
class BaseObject(object):
    ''' Notes : Avoid method conflict by ALWAYS settings this class in last
                at class definitions.
    '''
    def __init__(self, name='BaseObject'):
        # Le ruban est infini...
        #if name is None:
        #    print(traceback.extract_stack()[-2])
        #    fn,ln,func,text = traceback.extract_stack()[-2]
        #    name = text[:text.find('=')].strip()
        pass

    #def _name(self):
    #    return self.__name__
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

    def __copy__(self):
        return self.__class__(**self)

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

class ExpGroup(list, BaseObject):
    ''' A List of elements of an ExpTensor. '''
    def __add__(self, other):
        return self.__class__(list.__add__(self, other))
    def __sub__(self, other):
        return self.__class__([item for item in self if item not in other])

class Spec(BaseObject):
    @staticmethod
    def get(scriptname, *expe):
        ix = IX(default_index='spec')
        topmethod = ix.getfirst(scriptname, field='scriptsurname')
        if not topmethod:
            topmethod = ix.getfirst(scriptname, field='method')
            if not topmethod:
                raise ValueError('error: Unknown script: %s' % (scriptname))
            arguments = [scriptname] + arguments
            #script_name = topmethod['scriptsurname']

        module = importlib.import_module(topmethod['module'])
        script = getattr(module, topmethod['scriptname'])
        return script, arguments

    @staticmethod
    def get_all():
        ix = IX(default_index='spec')
        _res = ix.query(field='expe_name', terms='module_name')
        return _res

    @staticmethod
    def load(expe_name, expe_module):
        # debug to load from module or expe_name !

        p =  expe_module.split('.')
        modula, modulb = '.'.join(p[:-1]), p[-1]
        try:
            expdesign = getattr(importlib.import_module(modula), modulb)
            exp = getattr(expdesign, expe_name)
        except AttributeError as e:
            lgg.error("Seems that a spec `%s' has been removed : %s" % (expe_name, e))
            lgg.critical("Fatal Error: unable to load spec:  try `pymake update' or try again.")
            exit(2)

        return exp, expdesign


    @classmethod
    def table(cls):
        ix = IX(default_index='spec')
        t = OrderedDict()
        for elt in  ix.query(index='spec', terms=True):
            name = elt['module_name'].split('.')[-1]
            expes = t.get(name, []) + [ elt['expe_name'] ]
            t[name] = sorted(expes)
        return _table_(t, headers='keys', name=cls.__name__)

    # no more complex.
    # @sortbytype
    @classmethod
    def table_topos(cls, _spec):

        Headers = OrderedDict((('Corpuses',Corpus),
                               ('Exp',(ExpSpace, ExpTensor)),
                               ('Unknown', str)))

        tables = [ [] for i in range(len(Headers))]

        for expe_name, expe_module in _spec.items():
            expe, _ = cls.load(expe_name, expe_module)
            try:
                pos = [isinstance(expe, T) for T in Headers.values()].index(True)
            except ValueError:
                pos = len(Headers) - 1
            tables[pos].append(expe_name)


        return _table_(tables, headers=list(Headers.keys()))



class Script(BaseObject):
    @staticmethod
    def get(scriptname, arguments):

        ix = IX(default_index='script')
        topmethod = ix.getfirst(scriptname, field='scriptsurname')
        if not topmethod:
            # get the first method that have this name
            topmethod = ix.getfirst(scriptname, field='method')
            if not topmethod:
                raise ValueError('error: Unknown script: %s' % (scriptname))
            arguments = [scriptname] + arguments
            #script_name = topmethod['scriptsurname']

        module = importlib.import_module(topmethod['module'])
        script = getattr(module, topmethod['scriptname'])
        return script, arguments

    @classmethod
    def table(cls):
        ix = IX(default_index='script')
        t = OrderedDict()
        for elt in  ix.query(index='script', terms=True):
            name = elt['scriptname']
            methods = t.get(name, []) + [ elt['method'] ]
            t[name] = sorted(methods)
        return _table_(t, headers='keys', name=cls.__name__)


class Corpus(ExpVector):
    @staticmethod
    def get():
        pass

class Model(ExpVector):

    @staticmethod
    def get(model_name):
        ix = IX(default_index='model')

        _model =  None
        docir = ix.getfirst(model_name, field='surname')
        if docir:
            mn = importlib.import_module(docir['module'])
            _model = getattr(mn, docir['name'], None)
        return _model

    @staticmethod
    def get_all(_type='short'):
        ix = IX(default_index='model')
        if _type == 'short':
            res = ix.query(field='surname')
        elif _type == 'topos':
            _res = ix.query(field='surname', terms=True)
            res = []
            for elt in _res:
                # beurk
                if len(elt['category']) > 0:
                    # means that len(surname.split('.')) > 1
                    names = elt['surname'].split('.')
                    topos = '.'.join(elt['category'].split())
                    surname = '.'.join((names[0],  topos , names[1]))
                else:
                    surname = elt['surname']
                res.append(surname)
        return res

    @classmethod
    def table(cls, _type='short'):
        Headers = OrderedDict((('Corpuses',Corpus),
                               ('Models',(ExpSpace, ExpTensor)),
                               ('Unknown', str)))

        tables = [[], # corpus atoms...
                  cls.get_all(_type),
        ]

        return _table_(tables, headers=list(Headers.keys()))


class ExpTensor(OrderedDict, BaseObject):
    ''' Represent a set of Experiences (**expe**). '''
    def __init__(self,  *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        BaseObject.__init__(self)

    @classmethod
    def from_expe(cls, conf=None, expe=None, parser=None):
        ''' Return the tensor who is an Orderedict of iterable.
            Assume conf is an exp. Non list value will be listified.

            Parameters
            ----------
            expe : (ExpDesign, ExpSpace or dict)
                A design of experiment.
        '''
        _conf = conf.copy()
        if expe is None:
            expe = conf

        if not issubclass(type(expe), (cls, ExpSpace, dict, ExpVector)):
            raise ValueError('Expe not understood: %s' % type(expe))

        if issubclass(type(expe), Corpus):
            tensor = cls(corpus=expe)
        elif issubclass(type(expe), Model):
            tensor = cls(model=expe)
        elif isinstance(expe, ExpTensor):
            tensor = expe.copy()
        elif isinstance(expe, (dict, ExpSpace)):
            tensor = cls()
            tensor.update_from_dict(expe)
        else:
            raise NotImplementedError('input type of ExpVector unknow %s' % (expe))

        for k, v in tensor.items():
            if not issubclass(type(v), (list, set, tuple)):
                tensor[k] = [v]

        if _conf:
            tensor.update_from_dict(_conf, parser=parser)

        return tensor

    def update_from_dict(self, d, parser=None):
        ''' Update a tensor from a dict

            Parameters
            ----------
            d : dict
                the dict that uptate the tensor
            from_argv : bool
                if True, the is assumed to come from an CLI argparser. if the following conds are true :
                    * the settings in {d} are specified in the CLI (@check already filtererd in GramExp.parseargs)
                    * the settings in {d} is not in the CLI, and not in self.

            Notes
            -----
            SHould inherit _reserved keyword to prevent
        '''

        if parser is not None:
            dests_filled = get_dest_opt_filled(parser)

        for k, v in d.items():
            if k in ['_id_expe']:
                continue

            if parser is not None:
                if not k in dests_filled and k in self :
                    continue

            if issubclass(type(v), ExpVector):
                self[k] = v
            else:
                self[k] = [v]

    def get_size(self):
        return  np.prod([len(x) for x in self.values()])

    def push_dict(self, d):
        ''' push one dict inside a exptensor.
            It extend _bind rule to filter the tensor.
        '''
        tensor_len = np.prod([len(x) for x in self.values()])
        if len(self) == 0:
            self.update_from_dict(d)
            return True

        _need_bind = False
        _up_dict = {}
        for k, v in d.items():
            if k in ['_id_expe']:
                continue

            vector = self.get(k, []).copy()
            if v not in vector:
                if len(vector) == 0:
                    _need_bind = True
                    #lgg.debug('setting to bind: (%s : %s)' % (k, v))
                    break
                vector.append(v)
            _up_dict[k] = vector

        if _need_bind:
            #raise NotImplementedError('Need to push bind value to build a tensor from non-overlaping settings.')
            return False
        else:
            self.update(_up_dict)
            return True


    def table(self, extra=[]):
        return tabulate(extra+sorted(self.items(), key=lambda x:x[0]),
                               headers=['Params','Values'])

# @debug : Rename this class to ?
class ExpTensorV2(BaseObject):
    ''' Represent a set of Experiences (**expe**). '''
    def __init__(self, private_keywords=[]):
        BaseObject.__init__(self)
        self._private_keywords = private_keywords

        # --- Those are aligned ---
        self._tensors = [] # list of ExpTensor
        self._bind = []
        self._null = []
        self._hash = []
        #--------------------------
        self._lod = [] # list of dict
        self._conf = {}
        self._size = None

    @classmethod
    def from_conf(cls, conf, _max_expe=150000, private_keywords=[]):
        gt = cls(private_keywords=private_keywords)
        _spec = conf.pop('_spec', None)
        if not _spec:
            gt._tensors.append(ExpTensor.from_expe(conf))
            return gt

        exp = []
        max_expe = len(_spec)
        consume_expe = 0
        while consume_expe < max_expe:
            o = _spec[consume_expe]
            if isinstance(o, tuple):
                name, o = o

            if isinstance(o, ExpGroup):
                max_expe += len(o) -1
                _spec = _spec[:consume_expe] + o + _spec[consume_expe+1:]
            else:
                o['_name_expe'] = name
                exp.append(o)
                consume_expe += 1

            if max_expe > _max_expe:
                lgg.warning('Number of experiences exceeds the hard limit of %d.' % _max_expe)

        gt._tensors.extend([ExpTensor.from_expe(conf, spec) for spec in exp])
        return gt

    def __iter__(self):
        for tensor in self._tensors:
            yield tensor

    def remove_all(self, key):
        if key in self._conf:
            self._conf.pop(key)

        for tensor in self._tensors:
            if key in tensor:
                tensor.pop(key)

        # @Debug self._lod is left untouched...
        # Really ?
        for d in self._lod:
            if key in d:
                d.pop(key)

    def update_all(self, **kwargs):
        self._conf.update(kwargs)

        for tensor in self._tensors:
            tensor.update_from_dict(kwargs)

        for d in self._lod:
            d.update(kwargs)

    def set_default_all(self, defconf):
        ''' set default value in exp '''
        for k, v in defconf.items():

            for tensor in self._tensors:
                if not k in tensor:
                    tensor[k] = [v]
            for expe in self._lod:
                if not k in expe:
                    expe[k] = v
            if k in self._conf:
                # @debug: dont test if all the group have this unique value.
                self._conf[k] = v

    def get_all(self, key, default=[]):
        vec = []
        for tensor in self._tensors:
            vec.extend(tensor.get(key, []))

        if not vec:
            return default
        else:
            return vec

    def get_conf(self):
        _conf = {}
        for tensor in self._tensors:
            for k, v in tensor.items():
                if len(v) != 1:
                    if k in _conf:
                        _conf.pop(k)
                    continue

                if k in _conf and v[0] != _conf[k]:
                    _conf.pop(k)
                    continue
                else:
                    _conf[k] = v[0]

            #_confs.append(_conf)

        self._conf = _conf
        return self._conf

    def get_size(self):
        size = 0
        for tensor in self._tensors:
            size += tensor.get_size()
        self._size = size
        return self._size

    def check_bind(self):
        ''' Rules Filter '''

        for tensor in self._tensors:

            if '_bind' in tensor:
                _bind = tensor.pop('_bind')
                if not isinstance(_bind, list):
                    _bind = [_bind]
            else:
                #_bind = getattr(self, '_bind', [])
                _bind = []

            self._bind.append(_bind)

    def check_model_typo(self):
        ''' Assume default module is pymake '''
        for tensor in self._tensors:
            models = tensor.get('model', [])
            for i, m in enumerate(models):
                if not '.' in m:
                    models[i] = 'pmk.%s'%(m)

    def check_null(self):
        ''' Filter _null '''
        for tensor in self._tensors:
            _null = []
            for k in list(tensor.keys()):
                if '_null' in tensor.get(k, []):
                    tensor.pop(k)
                    _null.append(k)
            self._null.append(_null)

    def make_lod(self):
        ''' Make a list of Expe from tensor, with filtering '''

        self._lod = []
        for _id, tensor in enumerate(self._tensors):
            self._lod.extend(self._make_lod(tensor, _id))

        self._make_hash()
        return self._lod

    def _make_lod(self, tensor, _id):
        ''' 1. make dol to lod
            2. filter _bind rule
            3. add special parameter (expe_id)
        '''
        if len(tensor) == 0:
            lod =  []
        else:
            len_l = [len(l) for l in tensor.values()]
            keys = sorted(tensor)
            lod = [dict(zip(keys, prod)) for prod in product(*(tensor[key] for key in keys))]

        # POSTFILTERING
        # Bind Rules
        idtoremove = []
        for expe_id, d in enumerate(lod):
            for rule in self._bind[_id]:
                _bind = rule.split('.')
                values = list(d.values())

                # This is only for  last dot separator process
                for j, e in enumerate(values):
                    if type(e) is str:
                        values[j] = e.split('.')[-1]


                if len(_bind) == 2:
                    # remove all occurence if this bind don't occur
                    # simltaneous in each expe.
                    a, b = _bind
                    if b.startswith('!'):
                        # Exclusif Rule
                        b = b[1:]
                        if a in values and b in values:
                            idtoremove.append(expe_id)
                    else:
                        # Inclusif Rule
                        if a in values and not b in values:
                            idtoremove.append(expe_id)

                elif len(_bind) == 3:
                    # remove occurence of this specific key:value if
                    # it does not comply with this bind.
                    a, b, c = _bind
                    # Get the type of this key:value.
                    _type = type(d[b])
                    if _type is bool:
                        _type = lambda x: True if x in ['True', 'true', '1'] else False

                    if c.startswith('!'):
                        # Exclusif Rule
                        c = c[1:]
                        if a in values and _type(c) == d[b]:
                            idtoremove.append(expe_id)
                    else:
                        # Inclusif Rule
                        if a in values and _type(c) != d[b]:
                            idtoremove.append(expe_id)

        lod = [d for i,d in enumerate(lod) if i not in idtoremove]
        # Save true size of tensor (_bind remove)
        self._tensors[_id]._size = len(lod)

        # Add extra information in lod expes
        n_last_expe = sum([t._size for t in self._tensors[:_id]])
        for _id, expe in enumerate(lod):
            expe['_id_expe'] = _id + n_last_expe

        return lod

    # @todo; lhs for clustering expe applications.
    def _make_hash(self):
        _hash = []
        n_duplicate = 0
        for _id, _d in enumerate(self._lod):
            d = _d.copy()
            [ d.pop(k) for k in self._private_keywords if k in d and k != '_repeat']
            o = hash_objects(d)
            if o in _hash:
                n_duplicate += 1
            _hash.append(o)


        if n_duplicate > 0:
            lgg.warning('Duplicate experience: %d' % (n_duplicate))
            ask_sure_exit('Continue [y/n]?')
        self._hash = _hash

    def remake(self, indexs):
        ''' Update the curent tensors by selecting the ${indexs} '''

        self._lod = [self._lod[i] for i in indexs]
        self._tensors = []

        new_tensor = ExpTensor()
        consume_expe = 0
        self._tensors.append(new_tensor)
        while consume_expe < len(self._lod):
            d = self._lod[consume_expe]
            res = new_tensor.push_dict(d)
            if res is False:
                new_tensor = ExpTensor()
                self._tensors.append(new_tensor)
            else:
                consume_expe += 1

    def get_gt(self):
        ''' get Global Tensors.
            No _binding here...
        '''
        gt = {}
        for tensor in self._tensors:
            for k, v in tensor.items():
                _v = gt.get(k,[])
                gt[k] = _v + v
        return gt

    def table(self):
        tables = []
        for id, group in enumerate(self._tensors):
            if self._bind:
                extra = [('_bind', self._bind[id])]
            if id == 0:
                headers = ['Params','Values']
            else:
                headers = ''
            tables.append(tabulate(extra+sorted(group.items(), key=lambda x:x[0]), headers=headers))

        return '\n'.join(tables)

class ExpDesign(dict, BaseObject):
    ''' An Ensemble composed of ExpTensors and ExpVectors.

        NOTES
        -----
        Special attribute meaning:
            _alias : dict
                use when self._name is called to translate keywords
    '''
    def __init__(self,  *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

        # Not a Ultimate solution to keep a flexibility when defining Exp Design
        for k in dir(self):
            #_spec = ExpDesign((k, getattr(Netw, k)) for k in dir(Netw) if not k.startswith('__') )
            if not k.startswith('_'):
                v = getattr(self, k)
                #if not callable(v): #  python >3.2
                if not hasattr(v, '__call__'):
                    self[k] = v
        # @debug: add callable in reserved keyword
        self._reserved_keywords = list(set([w for w in dir(self) if w.startswith('_')] + ['_reserved_keywords']+dir(dict)+dir(BaseObject)))

        BaseObject.__init__(self)

    def _specs(self):
        return [ k for k  in self.keys() if k not in self._reserved_keywords ]


    @classmethod
    def _name(cls, l):
        if getattr(cls, '_alias', None):
            _alias =  cls._alias
        else:
            return l


        if isinstance(l, (set, list, tuple)):
            return [ _alias.get(i, i) for i in l ]
        elif isinstance(l, (dict, ExpSpace)):
            d = dict(l)
            for k, v in d.items():
                if isinstance(v, basestring) and v in _alias:
                    d[k] = _alias[v]
            return d
        else :
            return _alias.get(l, l)


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

    log = logging.getLogger('root')
    _logfile = False # for external integration @deprcated ?

    def __init__(self, pt, expe, gramexp):
        # @debug this, I dont know whyiam in lib/package sometimes, annoying !
        #os.chdir(os.getenv('PWD'))

        # Global
        self.expe_size = len(gramexp)
        self.gramexp = gramexp
        # Local
        self.pt = pt
        self.expe = expe

        # to exploit / Vizu
        self._it = pt['expe']
        self.corpus_pos = pt.get('corpus')
        self.model_pos = pt.get('model')
        self.output_path = self.expe['_output_path']

        if expe.get('_expe_silent'):
            self.log_silent()
        else:
            self.log.info('-'*10)
            self.log_expe()
            self.log.info('-'*10)

    def log_expe(self):
        expe = self.expe
        self.log.info(''.join([colored('Expe %d/%d', 'red'),
                          ' : %s -- %s -- N=%s -- K=%s']) % (
                              self._it+1, self.expe_size,
                              self.specname(expe.get('corpus')),
                              self.specname(expe.get('model')),
                              expe.get('N'), expe.get('K'),))

    def log_silent(self):
        if self.is_first_expe():
            print()

        prefix = 'Computing'
        n_it = self._it
        n_total = self.expe_size
        # Normalize
        n_it_norm = 2*42 * n_it // n_total

        progress= n_it_norm * '='  + (2*42-n_it_norm) * ' '
        print('\r%s: [%s>] %s/%s' % (prefix, progress, n_it+1, n_total), end = '\r')

        if self.is_last_expe():
            print()
            print()


    def is_first_expe(self):
        if 0 == self.pt['expe']:
            return True
        else:
            return False

    def is_last_expe(self):
        if self.expe_size - 1 == self.pt['expe']:
            return True
        else:
            return False

    @classmethod
    def display(cls, conf):
        import matplotlib.pyplot as plt
        block = not conf.get('save_plot', False)
        plt.show(block=block)

    @staticmethod
    @decorator
    def plot_simple(fun, *args, **kwargs):
        import matplotlib.pyplot as plt
        self = args[0]
        expe = self.expe
        kernel = fun(*args, **kwargs)
        if hasattr(expe, 'block_plot') and getattr(self, 'noplot', False) is not True:
            plt.show(block=expe.block_plot)
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
                    import matplotlib.pyplot as plt
                    from pymake.plot import _linestyle, _markers
                    group = groups[0]
                    self = args[0]
                    expe = self.expe

                    # Init Figs Sink
                    if not hasattr(self.gramexp, 'figs'):
                        figs = dict()
                        for c in self.gramexp.get_all(group):
                            figs[c] = ExpSpace()
                            figs[c].fig = plt.figure()
                            figs[c].linestyle = _linestyle.copy()
                            figs[c].markers = _markers.copy()

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
                            self.write_figs(expe, self.gramexp.figs)

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
    def _preprocess_(cls, gramexp):
        ''' This method has **write** access to Gramexp '''

        # update exp_tensor in gramexp
        if hasattr(cls, '_default_expe'):
            gramexp.exp_tensor.set_default_all(cls._default_expe)

        # Put a valid expe a the end.
        gramexp.reorder_firstnonvalid()

        if not gramexp._conf.get('simulate'):
            print(gramexp.exptable())

        return

    @classmethod
    def _postprocess_(cls, gramexp):
        cls.display(gramexp._conf)
        return


    def _preprocess(self):
        # heere, do a wrapper ?
        pass

    def _postprocess(self):
        # heere, do a wrapper ?
        pass

    def __call__(self):
        raise NotImplementedError

    def specname(self, n):
        #return self.gramexp._expdesign._name(n).lower().replace(' ', '')
        return self.gramexp._expdesign._name(n)

    def full_fig_path(self, fn):
        from pymake.frontend.frontend_io import _FIGS_PATH
        path = os.path.join(_FIGS_PATH, self.specname(fn))
        make_path(path)
        return path

    def formatName(fun):
        def wrapper(self, *args, **kwargs):
            args = list(args)
            expe = copy(args[0])
            for k, v in expe.items():
                if isinstance(v, basestring):
                    nn = self.specname(v)
                else:
                    nn = v
                setattr(expe, k, nn)
            args[0] = expe
            f = fun(self, *args, **kwargs)
            return f
        return wrapper


    @formatName
    def write_figs(self, expe, figs, _suffix=None, _fn=None, ext='.pdf'):
        if type(figs) is list:
            _fn = '' if _fn is None else self.specname(_fn)+'_'
            for i, f in enumerate(figs):
                suffix = '_'+ _suffix if _suffix and len(figs)>1 else ''
                fn = ''.join([_fn, '%s_%s', ext]) % (expe.corpus, str(i) + suffix)
                print('Writings figs: %s' % fn)
                f.savefig(self.full_fig_path(fn));
        elif issubclass(type(figs), dict):
            for c, f in figs.items():
                fn = ''.join([f.fn , ext])
                print('Writings figs: %s' % fn)
                f.fig.savefig(self.full_fig_path(fn));
        else:
            print('ERROR : type of Figure unknow, passing')

    def write_table(self, table, _fn=None, ext='.txt'):
        if isinstance(table, (list, np.ndarray)):
            _fn = '' if _fn is None else self.specname(_fn)+'_'
            fn = ''.join([_fn, 'table', ext])
            fn = self.full_fig_path(fn)
            with open(fn, 'w') as _f:
                _f.write(table)
        elif isinstance(table, dict):
            for c, t in table.items():
                fn = ''.join([t.fn ,'_', self.specname(c), '_table', ext])
                fn = self.full_fig_path(fn)
                with open(fn, 'w') as _f:
                    _f.write(t.table)
        else:
            print('ERROR : type `%s\' of Table unknow, passing' % (type(table)))




    def _format_line_out(self, model, comments=('#','%')):
        ''' extract data in model from variable name declared in {self.scv_typo}.
            The iterable build constitute the csv-like line, for **one iteration**, to be written in the outputfile.

            Spec : (Make a grammar dude !)
                * Each name in the **typo** should be accesible (gettable) in the model/module class, at fit time,
                * a '{}' symbol means that this is a list.
                * a 'x[y]' symbole that the dict value is requested,
                * if there is a list '{}', the size of the list should be **put just before it**,
                * if there is several list next each other, they should have the same size.

                @debug: raise a sprcialerror from this class,
                    instead of duplicate the exception print in _fotmat_line_out.
        '''
        line = []
        for o in self.expe._csv_typo.split():
            if o in comments:
                continue
            elif o.startswith('{'): # is a list
                obj = o[1:-1]
                brak_pt = obj.find('[')
                if brak_pt != -1: # assume  nested dictionary
                    obj, key = obj[:brak_pt], obj[brak_pt+1:-1]
                    try:
                        values = [str(elt[key]) for elt in getattr(model, obj).values()]
                    except (KeyError, AttributeError) as e:
                        values = self.format_error(model, o)

                else : # assume list
                    try:
                        values = [str(elt) for elt in getattr(model, obj)]
                    except (KeyError, AttributeError) as e:
                        values = self.format_error(model, o)
            else: # is atomic ie can be converted to string.
                try: values = str(getattr(model, o))
                except (KeyError, AttributeError) as e: values = self.format_error(model, o)


            if isinstance(values, list):
                line.extend(values)
            else:
                line.append(values)

        return line

    def format_error(self, model, o):
        traceback.print_exc()
        print('\n')
        self.log.critical("expe setting ${_format} is probably wrong !")
        self.log.error("model `%s' do not contains one of object: %s" % (str(model), o))
        print('Continue...')
        #os._exit(2)
        return 'None'


    def write_some(self, _f,  samples, buff=20):
        ''' Write data with buffer manager
            * lines are formatted as {self.csv_typo}
            * output file is {self._f}
        '''
        #fmt = self.fmt

        if samples is None:
            buff=1
        else:
            self._samples.append(samples)

        if len(self._samples) >= buff:
            #samples = np.array(self._samples)
            samples = self._samples
            #np.savetxt(f, samples, fmt=str(fmt))
            for line in samples:
                # @debug manage float .4f !
                line = ' '.join(line)+'\n'
                _f.write(line.encode('utf8'))

            _f.flush()
            self._samples = []


    def load_some(self, filename, iter_max=None, comments=('#','%')):
        ''' Load data from file according that each line
            respect the format in {self._csv_typo}.
        '''
        with open(filename) as f:
            data = f.read()

        data = filter(None, data.split('\n'))
        if iter_max:
            data = data[:iter_max]
        # Ignore Comments
        data = [re.sub("\s\s+" , " ", x.strip()).split() for l,x in enumerate(data) if not x.startswith(comments)]

        # Grammar dude ?
        col_typo = self.expe._csv_typo.split()[1:]
        array = []
        last_list_size = None
        # Format the data in a list of dict by entry
        for data_line in data:
            line = {}
            offset = 0
            for true_pos in range(len(data_line)):
                pos = true_pos + offset
                if pos >= len(data_line):
                    break
                o = col_typo[true_pos]
                if data_line[0] in comments:
                    break
                elif o.startswith('{'): # is a list
                    obj = o[1:-1]
                    brak_pt = obj.find('[')
                    if brak_pt != -1: # assume  nested dictionary
                        obj, key = obj[:brak_pt], obj[brak_pt+1:-1]
                        newkey = '.'.join((obj, key))
                        values = data_line[pos:pos+last_elt_size]
                        line[newkey] = values
                    else : # assume list
                        values = data_line[pos:pos+last_elt_size]
                        line[obj] = values
                    offset += last_elt_size-1
                else:
                    line[o] = data_line[pos]
                    if str.isdigit(line[o]):
                        last_elt_size = int(line[o])

            array.append(line)


        # Format the array of dict to a dict by entry function of iterations
        data = {}
        for line in array:
            for k, v in line.items():
                l = data.get(k, [])
                l.append(v)
                data[k] = l

        return data

    def init_fitfile(self):
        ''' Create the file to save the iterations state of a model.'''
        self._samples = []
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.fname_i = self.output_path + '.inf'
        if not '_csv_typo' in self.expe:
            self.log.warning('No csv_typo, for this model %s, no inference file...')
        else:
            self._fitit_f = open(self.fname_i, 'wb')
            self._fitit_f.write((self.expe._csv_typo + '\n').encode('utf8'))


    def clear_fitfile(self):
        ''' Write remaining data and close the file. '''
        if self._samples:
            self.write_some(self._fitit_f, None)
            self._fitit_f.close()


    def write_current_state(self, model):
        ''' push the current state of a model in the output file. '''
        self.write_some(self._fitit_f, self._format_line_out(model))


    def write_it_step(self, model):
        if not self.expe.get('write'):
            return
        self.write_current_state(model)

    def configure_model(self, model):
        # Inject the writing some method
        setattr(model, 'write_it_step', self.write_it_step)

        # meta model ? ugly hach
        if hasattr(model, 'model') and hasattr(model.model, 'fit'):
            setattr(model.model, 'write_it_step', self.write_it_step)

        # Configure csv_typo if present in model.
        if getattr(model, '_csv_typo', False):
            self.expe._csv_typo = model._csv_typo



