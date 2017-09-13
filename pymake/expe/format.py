# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import logging
import traceback,  importlib
import numpy as np
from collections import OrderedDict, defaultdict
from decorator import decorator
from functools import wraps

from pymake.util.utils import colored, basestring, get_dest_opt_filled
from pymake.index.indexmanager import IndexManager as IX

from tabulate import tabulate


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


class Script(BaseObject):
    @staticmethod
    def get(scriptname, arguments):

        ix = IX(default_index='script')
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
    def table():
        ix = IX()
        t = {}
        for elt in  ix.query(index='script', terms=True):
            name = elt['scriptname']
            methods = t.get(name, []) + [ elt['method'] ]
            t[name] = methods
        return tabulate(t, headers='keys')

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
    def list_all(_type='short'):
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


class ExpTensor(OrderedDict, BaseObject):
    ''' Represent a set of Experiences (**expe**). '''
    def __init__(self,  *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        name = self.pop('_name', 'expTensor')
        BaseObject.__init__(self, name)

    @classmethod
    def from_expe(cls, expe, parser=None):
        ''' Return the tensor who is an Orderedict of iterable.
            Assume conf is an exp. Non list value will be listified.

            Parameters
            ----------
            expe : (ExpDesign, ExpSpace or dict)
                A design of experiment.
        '''
        _conf = None
        if '_spec' in expe:
            _conf = expe.copy()
            expe = _conf.pop('_spec')

        if not issubclass(type(expe), (cls, ExpSpace, dict, ExpVector)):
            raise ValueError('Expe not understood: %s' % type(expe))

        if issubclass(type(expe), Corpus):
            tensor = cls(corpus=expe)
        elif issubclass(type(expe), Model):
            tensor = cls(model=expe)
        elif isinstance(expe, ExpTensor):
            # ExpSpace or dict or  not implemented ExpVector
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

    def push_dict(self, d):
        ''' push one dict inside a exptensor.
            It extend _bind rule to filter the tensor.
        '''
        tensor_len = np.prod([len(x) for x in self.values()])
        if len(self) == 0:
            self.update_from_dict(d)
            return

        _need_bind = False
        for k, v in d.items():
            if k in ['_id_expe']:
                continue

            vector = self.get(k, [])
            if v not in vector:
                if len(vector) == 0:
                    _need_bind = True
                    lgg.debug('setting to bind: (%s : %s)' % (k, v))
                vector.append(v)
            self[k] = vector

        if _need_bind:
            raise NotImplementedError('Need to push bind value to build a tensor from non-overlaping settings.')



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
                v = getattr(self, k)
                #if not callable(v): #  python >3.2
                if not hasattr(v, '__call__'):
                    self[k] = v
        # @debug: add callable in reserved keyword
        self._reserved_keywords = list(set([w for w in dir(self) if w.startswith('_')] + ['_reserved_keywords']+dir(dict)+dir(BaseObject)))
        name = self.pop('_name', 'expDesign')

        BaseObject.__init__(self, name)

    def _specs(self):
        return [ k for k  in self.keys() if k not in self._reserved_keywords ]

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
                  Model.list_all(_type),
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
        if getattr(self, '_mapname', None):
            mapname =  self._mapname
        else:
            return l


        if isinstance(l, (set, list, tuple)):
            return [ mapname.get(i, i) for i in l ]
        elif isinstance(l, (dict, ExpSpace)):
            d = dict(l)
            for k, v in d.items():
                if isinstance(v, basestring) and v in mapname:
                    d[k] = mapname[v]
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

    log = logging.getLogger('root')
    _logfile = False # for external integration

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
        self.corpus_pos = pt.get('corpus')
        self.model_pos = pt.get('model')
        self.output_path = self.gramexp.make_output_path(self.expe)

        self.log.info('-'*10)
        self.log.info(''.join([colored('Expe %d/%d', 'red'),
                          ' : %s -- %s -- N=%s -- K=%s']) % (
                              self._it+1, self.expe_size,
                              self.specname(expe.get('corpus')),
                              self.specname(expe.get('model')),
                              expe.get('N'), expe.get('K'),))
        self.log.info('-'*10)

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
                    from pymake.plot import _linestyle
                    group = groups[0]
                    self = args[0]
                    expe = self.expe

                    # Init Figs Sink
                    if not hasattr(self.gramexp, 'figs'):
                        figs = dict()
                        for c in self.gramexp.get(group, []):
                            figs[c] = ExpSpace()
                            figs[c].fig = plt.figure()
                            figs[c].linestyle = _linestyle.copy()

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
                            from pymake.util import out
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
    def _preprocess_(cls, gramexp):
        ''' This method has **write** access to Gramexp '''

        # update exp_tensor in gramexp
        if hasattr(cls, '_default_expe'):
            gramexp.set_default_expe(cls._default_expe)

        # Put a valid expe a the end.
        gramexp.reorder_lastvalid()

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
                        self.format_error(model, o)

                else : # assume list
                    try:
                        values = [str(elt) for elt in getattr(model, obj)]
                    except (KeyError, AttributeError) as e:
                        self.format_error(model, o)

                line.extend(values)
            else: # is atomic ie can be converted to string.
                try: value = str(getattr(model, o))
                except (KeyError, AttributeError) as e: line.append(value)

        return line

    def format_error(self, model, o):
        traceback.print_exc()
        print('\n')
        self.log.critical("expe setting ${_format} is probably wrong !")
        self.log.error("model `%s' do not contains one of object: %s" % (str(model), o))
        print('Continue...')
        #os._exit(2)


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



