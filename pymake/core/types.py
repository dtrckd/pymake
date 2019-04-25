from copy import copy, deepcopy
import traceback,  importlib
import numpy as np
from collections import OrderedDict, defaultdict
from itertools import product

from pymake import get_pymake_settings
from pymake.util.utils import get_dest_opt_filled, hash_objects, ask_sure_exit, basestring
from pymake.index.indexmanager import IndexManager as IX

import logging
lgg = logging.getLogger('root')


''' Structure of Pymake Objects.
'''


from tabulate import tabulate

# Ugly, integrate.
def _table_(tables, headers=[], max_line=10, max_row=30, name=''):

    if isinstance(headers, str):
        # tables is dict
        # Sort the dict
        ordered_keys = sorted(tables.keys())
        tables = OrderedDict([(k,tables[k]) for k in ordered_keys ])

        _tables = []
        cpt = 0
        max_row = 10
        for k, v in tables.items():
            if cpt % max_row == 0:
                t = OrderedDict()
                _tables.append(t)
            t[k] = v
            cpt += 1

        sep = '# %s'%name +  '\n'+'='*20
        print(sep)
        tables = '\n\n'.join([str(tabulate(t, headers=headers)) for t in _tables])
        return tables
    else:
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




# Not sure this one is necessary, or not here
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

    def copy(self):
        return type(self)(self)
    def __copy__(self):
        return self.__class__(**self)
    def __deepcopy__(self, memo):
        return self.copy()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            # Default pmk settings
            if key == '_write':
                return False

            lgg.debug('an ExpSpace request exceptions occured for key: %s ' % (key))
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

        # Don't work well, why ?
        #for i, o in enumerate(args):
        #    if isinstance(o, (dict, ExpGroup)):
        #        args[i] = deepcopy(o)

        list.__init__(self, args)
        BaseObject.__init__(args, **kwargs)

        # Recursively update value if kwargs found.
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
    def find(spec, field='expe_name'):
        ix = IX(default_index='spec')
        spec = ix.getfirst(spec, field=field)
        return spec

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
        # debug to load from module or expe_name !

        p =  expe_module.split('.')
        modula, modulb = '.'.join(p[:-1]), p[-1]
        try:
            expdesign = getattr(importlib.import_module(modula), modulb)
            exp = getattr(expdesign, expe_name)
        except AttributeError as e:
            lgg.error("Seems that a spec `%s' has been removed : %s" % (expe_name, e))
            lgg.error("Fatal Error: unable to load spec:  try `pmk update' or try again.")
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
    def get(scriptname, arguments, field='scriptsurname'):

        ix = IX(default_index='script')
        topmethod = ix.getfirst(scriptname, field=field)
        if not topmethod:
            # get the first method that have this name
            topmethod = ix.getfirst(scriptname, field='method')
            if not topmethod:
                return None
                #try:
                #    raise ValueError('error: Unknown script: %s' % (scriptname))
                #except:
                #    # Exception from pyclbr
                #    # index commit race condition I guess.
                #    print('error: Unknown script: %s' % (scriptname))
                #    exit(42)

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
    #               data_type : text | network | image | ...,
    #               dtype : "specific to a data_type"
    #               name : "object identifier"
    #               path : "disk repo"
    #               source : url | random
    #             }

    # IX integration needed..

    _corpus_data = [
        dict(name='clique'        , data_type='network', data_source='random', directed=False),
        dict(name='generator'     , data_type='network', data_source='random', directed=False, nodes=1000),
        dict(name='graph'         , data_type='network', data_source='random'),
        dict(name='alternate'     , data_type='network', data_source='random', directed=False),
        dict(name='BA'            , data_type='network', data_source='random'),
        dict(name='manufacturing' , data_type='network', data_source='web', directed=True, nodes=167, edges=5784, density=0.209, weighted=True),
        dict(name='fb_uc'         , data_type='network', data_source='web', directed=True, nodes=1899, edges=22195, density=0.006, weighted=True),
        dict(name='blogs'         , data_type='network', data_source='web', directed=True, nodes=1490, edges=19025, density=0.009, weighted=False),
        dict(name='emaileu'       , data_type='network', data_source='web', directed=True, nodes=1005, edges=25571, density=0.025, weighted=False),
        dict(name='propro'        , data_type='network', data_source='web', directed=False, nodes=2113, edges=1432, density=0.001, weighted=False),
        dict(name='euroroad'      , data_type='network', data_source='web', directed=True, nodes=1177, edges=1432, density=0.001, weighted=False),

        # gt
        dict(name='astro-ph',    data_type='network', data_source='gt', directed=False, nodes=16706, edges=121251, weighted=True),
        dict(name='cond-mat',    data_type='network', data_source='gt', directed=False, nodes=16726, edges=47594 , weighted=True),
        dict(name='hep-th',      data_type='network', data_source='gt', directed=False, nodes=8361,  edges=15751 , weighted=True),
        dict(name='netscience',  data_type='network', data_source='gt', directed=False, nodes=1589,  edges=2742  , weighted=True),
        dict(name='email-Enron', data_type='network', data_source='gt', directed=False, nodes=36692, edges=367662, weighted=False), # time weighted

        #dict(name='facebook'     ,  data_type='network', data_source='web', directed=True, nodes=None, edges=None, density=None, wheigted=None),

        #dict(name='reuter50'  , data_type='text', data_source='web'),
        #dict(name='nips12'    , data_type='text', data_source='web'),
        #dict(name='nips'      , data_type='text', data_source='web'),
        #dict(name='enron'     , data_type='text', data_source='web'),
        #dict(name='kos'       , data_type='text', data_source='web'),
        #dict(name='nytimes'   , data_type='text', data_source='web'),
        #dict(name='pubmed'    , data_type='text', data_source='web'),
        #dict(name='20ngroups' , data_type='text', data_source='web'),
        #dict(name='odp'       , data_type='text', data_source='web'),
        #dict(name='wikipedia' , data_type='text', data_source='web'),
        #dict(name='lucene', data_type='text', data_source='lucene'), # needs field spec
        #dict(name='mongo', data_type='text', data_source='mongo'), # needs field spec
    ]

    @classmethod
    def get(cls, corpus_name):
        if not corpus_name:
            return None

        corpus = False

        # index/mongo...
        for data in cls._corpus_data:
            if corpus_name.startswith(data['name']):
                corpus = data.copy()
                break

        return corpus

    @classmethod
    def get_all(cls):
        return cls._corpus_data

class Model(ExpVector):

    @staticmethod
    def get(model_name):
        ix = IX(default_index='model')

        _model =  None
        docir = ix.getfirst(model_name, field='surname')
        if docir:
            mn = importlib.import_module(docir['module'])
            #mn = importlib.import_module(docir['module'], package=local_package)
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
                # beurk
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
        ''' Return the tensor who is an OrderedDict of iterable.
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
        self._ds_ = [] # ExpDesign class per tensor
        #
        self._lod = [] # list of dict
        self._ds = [] # ExpDesign class per expe
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
            conf['_expe_hash'] = hash_objects(dict((k,v) for k,v in conf.items() if k not in private_keywords))
            gt._tensors.append(ExpTensor.from_expe(conf))
            gt._ds_.append(expdesign)
            return gt

        exp = []
        size_expe = len(_spec)
        consume_expe = 0
        while consume_expe < size_expe:
            o = _spec[consume_expe]
            if isinstance(o, tuple):
                #_type => expdesign
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
                o['_expe_hash'] = hash_objects(dict((k,v) for k,v in o.items() if k not in private_keywords))
                if hasattr(_type, '_alias'):
                    o['_alias'] = getattr(_type, '_alias')

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

        # Update current spec with _default_expe
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

        if hasattr(self, '_lod'):
            for d in self._lod:
                if key in d:
                    vec.append(d[key])
        else:
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
                elif len(_bind) ==1 and isinstance(_bind[0], list):
                    _bind = _bind[0]
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
                    # Set the model ref name
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
                        # Exclusif Rule
                        b = b[1:]
                        if a in values and b in values:
                            idtoremove.append(expe_id)
                    else:
                        # Inclusif Rule
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
                        # Exclusif Rule
                        c = c[1:]
                        if a in values and _type(c) == d[b]:
                            idtoremove.append(expe_id)
                    else:
                        # Inclusif Rule
                        if a in values and _type(c) != d[b]:
                            idtoremove.append(expe_id)


        lod = [d for i,d in enumerate(lod) if i not in idtoremove]
        # Save true size of tensor (_bind remove)
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
        It is the base class to write group of specification.

        NOTES
        -----
        Special attribute meaning:
            _alias : dict
                use when self._name is called to translate keywords
    '''

    def __init__(self,  *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

        # Not a Ultimate solution to keep a flexibility when defining Exp Design
        for k in dir(self):
            #_spec = ExpDesign((k, getattr(Netw, k)) for k in dir(Netw) if not k.startswith('__') )
            if not k.startswith('_'):
                v = getattr(self, k)
                #if not hasattr(v, '__call__'): # print a warning because hasattr call getattr in expSpace.
                if not callable(v): #  python >3.2
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

