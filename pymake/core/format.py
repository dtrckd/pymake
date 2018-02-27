import os
import re
from copy import copy, deepcopy
import traceback,  importlib
import numpy as np
from collections import OrderedDict, defaultdict
from decorator import decorator
from functools import wraps
from itertools import product

from pymake.util.utils import colored, basestring, get_dest_opt_filled, make_path, hash_objects, ask_sure_exit, get_pymake_settings
from pymake.index.indexmanager import IndexManager as IX

import logging
lgg = logging.getLogger('root')


''' Structure of Pymake Objects.
    This is what Pandas does ? no, but we provide a semantics structure, like a Lama does.
'''


from tabulate import tabulate

# Ugly, integrate.
def _table_(tables, headers=[], max_line=10, max_row=30, name=''):


    # tables is dict
    if isinstance(headers, str):
        # Sort the dict
        ordered_keys = sorted(tables.keys())
        tables = OrderedDict([(k,tables[k]) for k in ordered_keys ])

        sep = '# %s'%name +  '\n'+'='*20
        print(sep)
        return tabulate(tables, headers=headers)


    # tables is list
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

    def __init__(self, *args, **kwargs):
    #def __init__(self, name='BaseObject'):
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

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    #__builtins__.hasattr = hasattr

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
    def __deepcopy__(self, memo):
        return self.copy()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            lgg.warning('an ExpSpace request exceptions occured for key :%s ' % (key))
            raise AttributeError(key)

    # Scratch method because __hasattr__ catch an error in getattr.
    def hasattr(self, key):
        return key in self

    # For Piclking
    def __getstate__(self):
        return self
    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

class ExpVector(list, BaseObject):
    ''' A List of elements of an ExpTensor. '''
    def __add__(self, other):
        return self.__class__(list.__add__(self, other))
    def __sub__(self, other):
        return self.__class__([item for item in self if item not in other])

class ExpGroup(list, BaseObject):
    ''' A List of elements of an ExpTensor. '''

    def __init__(self, args, **kwargs):
        if kwargs:
            args = deepcopy(args)

        if isinstance(args, dict):
            args = [args]

        # Don't work well, why ?
        #for i, o in enumerate(args):
        #    if isinstance(o, (dict, ExpGroup)):
        #        args[i] = deepcopy(o)

        list.__init__(self, args)
        BaseObject.__init__(args, **kwargs)

        # Recursively update value if kwargs found.
        if len(kwargs) > 0:
            self.update_all(self, kwargs)

    def update_all(self, l, d):
        for o in l:
            if isinstance(o, list):
                self.update_all(o, d)
            elif isinstance(o, dict):
                for k, v in d.items():
                    o[k] = v
        return

    def __add__(self, other):
        return self.__class__(list.__add__(self, other))
    def __sub__(self, other):
        return self.__class__([item for item in self if item not in other])

class Spec(BaseObject):
    @staticmethod
    def get(scriptname, *expe):
        ix = IX(default_index='spec')
        raise NotImplementedError

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
            lgg.error("Fatal Error: unable to load spec:  try `pymake update' or try again.")
            exit(2)

        return exp, expdesign


    @classmethod
    def table(cls):
        ix = IX(default_index='spec')
        t = OrderedDict()
        for elt in ix.query(index='spec', terms=True):
            name = elt['module_name'].split('.')[-1]
            obj, _ = cls.load(elt['expe_name'], elt['module_name'])
            if isinstance(obj, (ExpSpace, ExpTensor, ExpGroup)):
                expes = t.get(name, []) + [ elt['expe_name'] ]
                t[name] = sorted(expes)
        return _table_(t, headers='keys', name=cls.__name__)

    # no more complex.
    # @sortbytype
    @classmethod
    def table_topos(cls, _spec):

        Headers = OrderedDict((('Corpuses', Corpus),
                               ('Models', Model),
                               ('Vector', ExpVector),
                               ('Exp', (ExpSpace, ExpTensor, ExpGroup)),
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
    def get_all(_type='flat'):
        ix = IX(default_index='script')

        if _type == 'flat':
            _res = ix.query(field='method')
        elif _type == 'hierarchical':
            _res = ix.query(field='scriptsurname', terms=True)
        return _res

    @staticmethod
    def get(scriptname, arguments):

        ix = IX(default_index='script')
        topmethod = ix.getfirst(scriptname, field='scriptsurname')
        if not topmethod:
            # get the first method that have this name
            topmethod = ix.getfirst(scriptname, field='method')
            if not topmethod:
                try:
                    raise ValueError('error: Unknown script: %s' % (scriptname))
                except:
                    # Exception from pyclbr
                    # index commit race condition I guess.
                    print('error: Unknown script: %s' % (scriptname))
                    exit(42)

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

    # Meta-grammar / Ontology :
    #   Corpus := {
    #               structure : text | network | image | ...,
    #               type : "specific to a structure"
    #               name : "object identifier"
    #               path : "disk repo"
    #               source : url | random
    #             }

    # IX integration needed..
    _corpus_typo = {
        'network': [
            'clique', 'generator', 'graph', 'alternate', 'BA', # random
            'facebook',
            'fb_uc',
            'manufacturing',
            'propro',
            'blogs',
            'euroroad',
            'emaileu'
        ],
        'text': ['reuter50',
                 'nips12',
                 'nips',
                 'enron',
                 'kos',
                 'nytimes',
                 'pubmed',
                 '20ngroups',
                 'odp',
                 'wikipedia',
                 'lucene']}
    @classmethod
    def get(cls, corpus_name):
        if not corpus_name:
            return None

        corpus = False
        for key, cps in cls._corpus_typo.items():
            if corpus_name.startswith(tuple(cps)):
                corpus = {'name': corpus_name, 'structure':key}
                break
        return corpus

    @classmethod
    def get_all(cls):
        return cls._corpus_typo

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
        tables = cls.get_all(_type),
        return _table_(tables, headers=['Models'])


class ExpTensor(OrderedDict, BaseObject):
    ''' Represent a set of Experiences (**expe**). '''
    def __init__(self,  *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        BaseObject.__init__(self)

        self._size = 0

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
        elif issubclass(type(expe), ExpVector):
            tensor = cls((str(i),j) for i,j in enumerate(expe))
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

    def get_size(self, virtual=False):
        if virtual:
            return  np.prod([len(x) for x in self.values()])
        else:
            return self._size


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
    ''' Represent a set of Experiences (**expe**) of type ExpTensor... '''
    def __init__(self, private_keywords=[]):
        BaseObject.__init__(self)
        self._private_keywords = private_keywords

        # --- Those are aligned ---
        self._tensors = [] # list of ExpTensor
        self._bind = []
        self._null = defaultdict(list)
        self._hash = []
        self._ds_ = [] # ExpDesign class per tensor
        #
        self._lod = [] # list of dict
        self._ds = [] # ExpDesign class per expe
        # --- meta ---
        self._conf = {}
        self._size = None

    @classmethod
    def from_conf(cls, conf, _max_expe=2e6, private_keywords=[], expdesign=None):
        gt = cls(private_keywords=private_keywords)
        _spec = conf.pop('_spec', None)
        if not _spec:
            if not expdesign:
                expdesign = ExpDesign
            conf['_name_expe'] = '_default_expe'
            gt._tensors.append(ExpTensor.from_expe(conf))
            gt._ds_.append(expdesign)
            return gt

        exp = []
        size_expe = len(_spec)
        consume_expe = 0
        while consume_expe < size_expe:
            o = _spec[consume_expe]
            if isinstance(o, tuple):
                # _type => expdesign
                name, o, _type = o

            if isinstance(o, ExpGroup):
                size_expe += len(o) -1
                _spec = _spec[:consume_expe] + o + _spec[consume_expe+1:]
            elif isinstance(o, list):
                exp.append(o)
                gt._ds_.append(_type)
                consume_expe += 1
            else:
                o['_name_expe'] = name
                exp.append(o)
                gt._ds_.append(_type)
                consume_expe += 1

            if size_expe > _max_expe:
                lgg.warning('Number of experiences exceeds the hard limit of %d (please review ExpTensor).' % _max_expe)

        gt._tensors.extend([ExpTensor.from_expe(conf, spec) for spec in exp])
        return gt

    def __iter__(self):
        for tensor in self._tensors:
            yield tensor

    def __len__(self):
        return self.get_size()

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
        ''' Get all values associated to a given key. '''
        vec = []
        for tensor in self._tensors:
            vec.extend(tensor.get(key, []))

        if not vec:
            return default
        else:
            return vec

    def get_nounique_keys(self):
        ''' Return key that has gridded (different value occurence in the set of tensor). '''
        keys = defaultdict(set)
        for tensor in self._tensors:
            for k in tensor:
                o = tensor.get(k, [])
                for v in o:
                    if isinstance(v, str) and not v.startswith('_'):
                        keys[k].add(v)

        nounique_keys = []
        for k, _set in keys.items():
            if len(_set) > 1:
                nounique_keys.append(k)

        return nounique_keys


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
                    # Set the model ref name
                    pkg = get_pymake_settings('default_model')
                    if len(pkg) > 8:
                        prefix = pkg[:3]
                        if '.' in pkg:
                            prefix  += ''.join(map(lambda x:x[0], pkg.split('.')[1:]))
                    else:
                        prefix = pkg.split('.')[0]

                    models[i] = '%s.%s'%(prefix, m)

    def check_null(self):
        ''' Filter _null '''
        for tensor in self._tensors:
            for k in list(tensor.keys()):
                if '_null' in tensor.get(k, []):
                    v = tensor.pop(k)
                    self._null[k].append(v)

    def make_lod(self, skip_check=False):
        ''' Make a list of Expe from tensor, with filtering '''

        self._lod = []
        for _id, tensor in enumerate(self._tensors):
            lods = self._make_lod(tensor, _id)
            tensor._size = len(lods)
            self._lod.extend(lods)
            self._ds.extend([self._ds_[_id]]*len(lods))

        self._make_hash(skip_check)
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
    def _make_hash(self, skip_check=False):
        _hash = []
        n_duplicate = 0
        for _id, _d in enumerate(self._lod):
            d = _d.copy()
            [ d.pop(k) for k in self._private_keywords if k in d and k != '_repeat']
            o = hash_objects(d)
            if o in _hash:
                n_duplicate += 1
            _hash.append(o)


        if n_duplicate > 0 and not skip_check:
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
                new_tensor._size += 1

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

    def get_keys(self):
       return list(self.get_gt())

    def table(self):
        tables = []
        for id, group in enumerate(self._tensors):
            src = self._ds[id].__name__
            spec = group.get('_name_expe', ['void'])[0]
            h = '=== %s > %s > %s expe ===' % (src, spec, group.get_size())
            tables.append(h)
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
                #if not hasattr(v, '__call__'): # print a waring because hasattr call getattr in expSpace.
                if not callable(v): #  python >3.2
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

    def __init__(self, pt, expe, expdesign, gramexp):
        ''' Sandbox class for scripts/actions.

            Parameters
            ----------
            pt: int
                Positional indicator for current run
            expe: ExpSpace
                Current spec
            expdesign: ExpDesign
                Current design class
            gramexp: Global object
        '''

        self._expdesign = expdesign

        # @debug this, I dont know whyiam in lib/package sometimes, annoying !
        os.chdir(os.getenv('PWD'))

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

    def get_data_path(self):
        path = get_pymake_settings('project_data')
        path = os.path.join(path, self.expe.get('_data_type', ''))
        return path

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
        n_it = self._it +1
        n_total = self.expe_size
        # Normalize
        n_it_norm = 2*42 * n_it // n_total

        progress= n_it_norm * '='  + (2*42-n_it_norm) * ' '
        print('\r%s: [%s>] %s/%s' % (prefix, progress, n_it, n_total), end = '\r')

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
    def tabulate(cls, *args, **kwargs):
        return tabulate(*args, **kwargs)

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
    def raw_plot(*groups, **_kwargs):
        ''' If no argument, simple plot.
            If arguments :
                * [0] : group figure by this
                * [1] : key for id (title and filename)
        '''
        if groups and len(groups[1:]) == 0 and callable(groups[0]):
            # decorator whithout arguments
            return ExpeFormat.plot_simple(groups[0])

        def decorator(fun):
            @wraps(fun)
            def wrapper(*args, **kwargs):
                import matplotlib.pyplot as plt
                from pymake.plot import _linestyle, _markers
                group = groups[0]
                self = args[0]
                expe = self.expe
                discr_args = []
                if len(args) > 1:
                    discr_args = args[2].split('/')

                # Init Figs Sink
                if not hasattr(self.gramexp, '_figs'):
                    figs = dict()
                    for c in self.gramexp.get_set(group):
                        figs[c] = ExpSpace()
                        figs[c].fig = plt.figure()
                        figs[c].linestyle = _linestyle.copy()
                        figs[c].markers = _markers.copy()

                    self.gramexp._figs = figs
                kernel = fun(*args, **kwargs)

                # Set title and filename
                title = ' '.join('{{{0}}}'.format(w) for w in groups).format(**self.specname(expe))
                expfig = self.gramexp._figs[expe[group]]
                expfig.base = '%s_%s' % (fun.__name__, title.replace(' ', '_'))
                expfig.args = discr_args
                expfig.fig.gca().set_title(title)

                # Save on last call
                if self._it == self.expe_size -1:
                    if expe.write:
                        self.write_frames(self.gramexp._figs)

                return kernel
            return wrapper

        return decorator

    @staticmethod
    def plot(*a, **b):
        ''' Doc todo (plot alignement...) '''

        def decorator(fun):
            @wraps(fun)
            def wrapper(*args, **kwargs):
                import matplotlib.pyplot as plt
                from pymake.plot import _linestyle, _markers

                self = args[0]
                discr_args = [] # discriminant keys (to distinguish filename)
                if len(args) == 1:
                    groups = ['corpus']
                    attribute = '_entropy'
                else:
                    groups = args[1].split(':')
                    if len(groups) == 1:
                        attribute = groups[0]
                        groups = None
                    else:
                        attribute = groups.pop(-1)
                        groups = groups[0]
                    if len(args) > 2:
                        discr_args = args[2].split('/')

                if groups:
                    groups = groups.split('/')
                    ggroup = self._file_part([self.expe.get(g) for g in groups], sep='-')
                else:
                    ggroup = None

                # Init Figs Sink
                if not hasattr(self.gramexp, '_figs'):
                    figs = dict()
                    if groups:
                        gset = product(*filter(None, [self.gramexp.get_set(g) for g in groups]))
                    else:
                        gset = [None]

                    for g in gset:
                        gg = '-'.join(map(str,g)) if g else None
                        figs[gg] = ExpSpace()
                        figs[gg].fig = plt.figure()
                        figs[gg].linestyle = _linestyle.copy()
                        figs[gg].markers = _markers.copy()

                    self.gramexp._figs = figs

                frame = self.gramexp._figs[ggroup]
                kernel = fun(self, frame, attribute)

                # Set title and filename
                if self.expe.get(groups[0]):
                    #title = ' '.join('{{{0}}}'.format(w) for w in groups).format(**self.specname(self.expe))
                    ctitle = tuple(filter(None,map(lambda x:self.specname(self.expe.get(x, x)), groups)))
                    s = '_'.join(['%s'] * len(ctitle))
                    title = s % ctitle
                else:
                    title = ' '.join(self.gramexp.get_nounique_keys())
                    if not title:
                        title = '%s %s' % tuple(map(lambda x:self.expe.get(x, x), ['corpus', 'model']))

                frame.base = '%s_%s' % (fun.__name__, attribute)
                frame.args = discr_args
                frame.fig.gca().set_title(title)
                frame.fig.gca().set_xlabel('iterations')
                frame.fig.gca().set_ylabel(attribute)

                # Save on last call
                if self._it == self.expe_size -1:
                    if self.expe.write:
                        self.write_frames(self.gramexp._figs)

                return kernel
            return wrapper

        return decorator

    @staticmethod
    def table(*a, **b):
        ''' Doc todo (plot alignement...) '''

        def decorator(fun):
            @wraps(fun)
            def wrapper(*args, **kwargs):
                self = args[0]
                discr_args = [] # discriminant keys (to distinguish filename)
                if len(args) == 1:
                    x, y, z = 'corpus', 'model', '_entropy'
                else:
                    x, y, z = args[1].split(':')
                    if len(args) > 2:
                        discr_args = args[2].split('/')

                if discr_args:
                    groups = discr_args
                    # or None if args not in expe (tex option...)
                    ggroup = self._file_part([self.expe.get(g) for g in groups], sep='-') or None
                else:
                    groups = None
                    ggroup = None

                _z = z.split('-')

                if not hasattr(self.gramexp, '_tables'):
                    tables = dict()
                    if groups:
                        gset = product(*filter(None, [self.gramexp.get_set(g) for g in groups]))
                    else:
                        gset = [None]

                    for g in gset:
                        gg = '-'.join(map(str,g)) if g else None
                        tables[gg] = ExpSpace()
                        array, floc = self.gramexp.get_array_loc(x, y, _z)
                        tables[gg].array = array
                        tables[gg].floc = floc

                    self.gramexp._tables = tables

                frame = self.gramexp._tables[ggroup]
                array = frame.array
                floc = frame.floc

                for z in _z:
                    kernel = fun(self, array, floc, x, y, z, **kwargs)

                if self._it == self.expe_size -1:
                    for ggroup in list(self.gramexp._tables):
                        _table = self.gramexp._tables.pop(ggroup)
                        array = _table.array
                        for zpos, z in enumerate(_z):
                            # Format table
                            #tablefmt = 'latex' # 'simple'
                            tablefmt = 'latex' if 'tex' in discr_args else 'simple'
                            Meas = self.specname(self.gramexp.get_set(y))
                            arr = self.highlight_table(array[:,:,zpos])
                            table = np.column_stack((self.specname(self.gramexp.get_set(x)), arr))
                            Table = tabulate(table, headers=Meas, tablefmt=tablefmt, floatfmt='.3f')

                            gg = z +'-'+ ggroup if ggroup else z
                            self.gramexp._tables[gg] = ExpSpace({'table': Table,
                                                                 'base':'_'.join((fun.__name__,
                                                                                  str(self.expe[x]),
                                                                                  str(self.expe[y]))),
                                                                 'args':discr_args,
                                                                 #'args':self.gramexp.get_nounique_keys(x, y),
                                                                })

                            print(colored('\n%s Table:'%(gg), 'green'))
                            print(Table)


                    if self.expe.write:
                        tablefmt_ext = dict(simple='md', latex='tex')
                        self.write_frames(self.gramexp._tables, ext=tablefmt_ext[tablefmt])

                return kernel
            return wrapper


        return decorator


    @staticmethod
    def _file_part(group, sep='_'):
        part = sep.join(map(str, filter(None, group)))
        return part

    def highlight_table(self, array, highlight_dim=1):
        hack_float = np.vectorize(lambda x : '{:.3f}'.format(float(x)))
        table = np.char.array(hack_float(array), itemsize=42)
        # vectorize
        for i, col in enumerate(array.argmax(1)):
            table[i, col] = colored(table[i, col], 'bold')
        for i, col in enumerate(array.argmin(1)):
            table[i, col] = colored(table[i, col], 'magenta')

        return table

    def specname(self, n):
        #return self._expdesign._name(n).lower().replace(' ', '')
        return self._expdesign._name(n)

    def full_fig_path(self, fn):
        figs_path = get_pymake_settings('project_figs')
        path = os.path.join(figs_path, self.expe.get('_refdir',''),  self.specname(fn))
        make_path(path)
        return path

    def formatName(self, expe):
        expe = copy(expe)
        for k, v in expe.items():
            if isinstance(v, basestring):
                nn = self.specname(v)
            else:
                nn = v
            setattr(expe, k, nn)
        return expe


    def write_frames(self, frames, base='', suffix='', ext=None, args=''):
        expe = self.formatName(self.expe)

        if isinstance(frames, str):
            frames = [frames]

        if type(frames) is list:
            if base:
                base = self.specname(base)
            if args:
                s = '_'.join(['%s'] * len(args))
                args = s % tuple(map(lambda x:expe.get(x, x), args))
            for i, f in enumerate(frames):
                idi = i if len(frames) > 1 else None
                fn = self._file_part([base, args, suffix, idi])
                fn = self.full_fig_path(fn)
                self._kernel_write(f, fn, ext=ext)
        elif issubclass(type(frames), dict):
            for c, f in frames.items():
                base = f.get('base') or base
                args = f.get('args') or args
                if base:
                    base = self.specname(base)
                if args:
                    s = '_'.join(['%s'] * len(args))
                    args = s % tuple(map(lambda x:expe.get(x, x), args))
                fn = self._file_part([self.specname(c), base, args, suffix])
                fn = self.full_fig_path(fn)
                self._kernel_write(f, fn, ext=ext, title=c)
        else:
            print('Error : type of Frame unknow, passing: %s' % type(frame))


    def _kernel_write(self, frame, fn, title=None, ext=None):
        if isinstance(frame, dict):
            if 'fig' in frame:
                ext = ext or 'pdf'
                fn = fn +'.'+ ext
                print('Writing frame: %s' % fn)
                frame.fig.savefig(fn)
            elif 'table' in frame:
                ext = ext or 'md'
                fn = fn +'.'+ ext
                print('Writing frame: %s' % fn)
                caption = '\caption{{{title}}}\n'
                with open(fn, 'w') as _f:
                    if  title:
                        _f.write(caption.format(title=title))
                    _f.write(frame.table+'\n')

        elif isinstance(frame, str):
            ext = ext or 'md'
            fn = fn +'.'+ ext
            print('Writing frame: %s' % fn)
            with open(fn, 'w') as _f:
                _f.write(frame)
        else:
            # assume figure
            ext = ext or 'pdf'
            fn = fn +'.'+ ext
            print('Writing frame: %s' % fn)
            frame.savefig(fn)



    @classmethod
    def _preprocess_(cls, gramexp):
        ''' This method has **write** access to Gramexp '''

        # update exp_tensor in gramexp
        if hasattr(cls, '_default_expe'):
            gramexp._tensors.set_default_all(cls._default_expe)

        # Put a valid expe a the end.
        gramexp.reorder_firstnonvalid()

        if not gramexp._conf.get('simulate'):
            cls.log.info(gramexp.exptable())

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


    def _format_line_out(self, model):
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
            if o.startswith('{'): # is a list
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


    def load_some(self, filename=None, iter_max=None, comments=('#','%')):
        ''' Load data from file according that each line
            respect the format in {self._csv_typo}.
        '''
        if filename is None:
            filename = self.output_path + '.inf'

        if not os.path.exists(filename):
            return None

        with open(filename) as f:
            data = f.read()

        data = filter(None, data.split('\n'))
        if iter_max:
            data = data[:iter_max]
        # Ignore Comments
        data = [re.sub("\s\s+" , " ", x.strip()).split() for l,x in enumerate(data) if not x.startswith(comments)]

        # Grammar dude ?
        col_typo = self.expe._csv_typo.split()
        array = []
        last_elt_size = None
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
            self.log.warning('No csv_typo, for this model %s, no inference file...'%(self.expe.get('model')))
        else:
            self._fitit_f = open(self.fname_i, 'wb')
            self._fitit_f.write(('#' + self.expe._csv_typo + '\n').encode('utf8'))


    def clear_fitfile(self):
        ''' Write remaining data and close the file. '''
        if self._samples:
            self.write_some(self._fitit_f, None)
            self._fitit_f.close()


    def write_current_state(self, model):
        ''' push the current state of a model in the output file. '''
        self.write_some(self._fitit_f, self._format_line_out(model))


    def write_it_step(self, model):
        if self.expe.get('write'):
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



