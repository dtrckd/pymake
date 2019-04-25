import os
import re
from copy import copy
import traceback
from decorator import decorator
from functools import wraps
from itertools import product
import logging
from collections import OrderedDict
import numpy as np

from pymake import ExpSpace, get_pymake_settings
from pymake.util.utils import colored, basestring, make_path, hash_objects

from tabulate import tabulate



''' Sanbox Base class for expe execution.
'''



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
    _logfile = False # for external integration @deprcated ?

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
            gramexp: Global pmk object

            Methods
            -------
            log_expe:
                overwrite the header message before each run.
            is_first_expe:
                return true if the current run is the first.
            is_last_expe:
                return true if the current run is the last.
            load_frontend:
                load the data frontend for the current expe.
            load_model:
                the model for the current expe.

            Decorator
            --------
            plot:
                plot control on expe
            table:
                table control on expe

        '''

        self._expdesign = expdesign

        # Global
        self.expe_size = len(gramexp)
        self.gramexp = gramexp
        self.D = self.gramexp.D # Global conteneur
        # Local
        self.pt = pt
        self.expe = expe

        # Plot utils
        from pymake.plot import _linestyle, _markers, _colors
        self.linestyles = _linestyle.copy()
        self.markers = _markers.copy()
        self.colors = _colors.copy()

        # to exploit / Vizu
        self._it = pt['expe']
        self.corpus_pos = pt.get('corpus')
        self.model_pos = pt.get('model')
        self.output_path = self.expe['_output_path']
        self.input_path = self.expe['_input_path']

        if expe.get('_expe_silent'):
            self.log_silent()
        else:
            self.log.info('-'*10)
            self.log.info(''.join([colored('Expe %d/%d', 'red'), ' : ', self.log_expe()]) % (
                self._it+1, self.expe_size,
            ))
            self.log.info('-'*10)

        self._set_csv_typo()

    def _set_csv_typo(self):
        csv_typo = self.expe.get('_csv_typo')
        if csv_typo is None:
            if 'model' in self.expe:
                from pymake.frontend.manager import ModelManager
                model = ModelManager.from_expe(self.expe)
                csv_typo = getattr(model, '_csv_typo', None)
                try:
                    from pymake.frontend.manager import ModelManager
                    model = ModelManager.from_expe(self.expe)
                    csv_typo = getattr(model, '_csv_typo', None)
                except Exception as e:
                    self.log.debug('No csv_typo spec found...: %s' % e)
                    csv_typo = None

        self._csv_typo = csv_typo

    def log_expe(self):
        expe = self.expe
        msg = '%s -- %s -- N=%s -- K=%s' % (self.specname(expe.get('corpus')),
                                            self.specname(expe.get('model')),
                                            expe.get('N'), expe.get('K'))
        return msg

    def log_silent(self):
        if self.is_first_expe():
            print()

        prefix = 'Computing'
        n_it = self._it +1
        n_total = self.expe_size
        # Normalize
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

    def spec_from_expe(self, spec_map=None):
        ''' Return a sub dict from expe, from spec_map. '''
        if spec_map is None:
            spec_map = {}

        spec = {}
        for k, v in spec_map.items():
            if v in self.expe:
                spec[k] = self.expe[v]

        return spec

    def get_data_path(self):
        path = get_pymake_settings('project_data')
        path = os.path.join(path, '')
        return path

    def getenv(self, a=None):
        return self.gramexp.getenv(a)

    def get_expset(self, param):
        return self.gramexp.get_set(param)

    def get_figs(self):
        return list(self.gramexp._figs.values())

    def get_tables(self):
        return list(self.gramexp._tables.values())

    def get_current_group(self):
        return self._file_part(sorted([str(self.expe.get(g)) for g in self._groups]), sep='-')

    def get_current_frame(self):
        group = self.get_current_group()
        frame = self.gramexp._figs[group]
        return frame


    def get_description(self, full=False):
        if full:
            return '/'.join((self.expe._refdir, os.path.basename(self.output_path)))
        else:
            return os.path.basename(self.output_path)

    @classmethod
    def tabulate(cls, *args, **kwargs):
        return tabulate(*args, **kwargs)

    @classmethod
    def display(cls, conf):
        import matplotlib.pyplot as plt
        block = not conf.get('_no_block_plot', False)
        plt.show(block=block)

    @staticmethod
    @decorator
    def plot_simple(fun, *args, **kwargs):
        # @obsolete ? (no block in save, this is it?
        import matplotlib.pyplot as plt
        self = args[0]
        expe = self.expe
        kernel = fun(*args, **kwargs)
        if '_no_block_plot' in expe and getattr(self, 'noplot', False) is not True:
            plt.show(block=expe._no_block_plot)
        return kernel

    @staticmethod
    def expe_repeat(fun):

        def wrapper(self, *args, **kwargs):

            repeats = self.get_expset('_repeat')
            if len(repeats)>1:
                kwargs['repeat'] = True

            res = fun(self, *args, **kwargs)

            return res
        return wrapper


    @staticmethod
    def raw_plot(*groups, **_kwargs):
        ''' If no argument, simple plot => @obsolete?
            If arguments (in decorator) :
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
                if len(groups) == 0:
                    group = '_id_expe'
                else:
                    group = groups[0]
                self = args[0]
                expe = self.expe
                discr_args = []
                if len(args) > 1:
                    discr_args = args[1].split('/')

                # Init Figs Sink
                if not hasattr(self.gramexp, '_figs'):
                    figs = dict()
                    for c in self.gramexp.get_set(group):
                        figs[c] = ExpSpace()
                        figs[c].group = c
                        figs[c].fig = plt.figure()
                        figs[c].linestyle = _linestyle.copy()
                        figs[c].markers = _markers.copy()

                    self.gramexp._figs = figs

                frame = self.gramexp._figs[expe[group]]
                frame.ax = lambda: frame.fig.gca()

                kernel = fun(self, frame, *args[1:])

                # Set title and filename
                title = ' '.join('{{{0}}}'.format(w) for w in groups).format(**self.specname(expe))
                frame.base = '%s_%s' % (fun.__name__, title.replace(' ', '_'))
                frame.args = discr_args
                if 'title' in frame:
                    # if multipolot
                    #plt.suptitle(frame.title)
                    frame.ax().set_title(frame.title)
                else:
                    frame.ax().set_title(title)

                # Save on last call
                if self._it == self.expe_size -1:
                    if expe._write:
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
                discr_args = [] # discriminant keys (to distinguish filename)
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
                    self._groups = groups
                    ggroup = self.get_current_group()
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
                        gg = '-'.join(sorted(map(str,g))) if g else None
                        figs[gg] = ExpSpace()
                        figs[gg].group = g
                        figs[gg].fig = plt.figure()
                        figs[gg].linestyle = _linestyle.copy()
                        figs[gg].markers = _markers.copy()

                    self.gramexp._figs = figs

                frame = self.gramexp._figs[ggroup]
                frame.ax = lambda: frame.fig.gca()

                kernel = fun(self, frame, attribute)

                # Set title and filename
                if groups and self.expe.get(groups[0]):
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

                if 'fig_xaxis' in frame or self.expe.get('fig_xaxis'):
                    xaxis_name = frame.get('xaxis', self.expe.get('fig_xaxis'))
                    frame.ax().set_xlabel(xaxis_name)
                else:
                    frame.ax().set_xlabel('iterations')

                if 'fig_yaxis' in frame or self.expe.get('fig_yaxis'):
                    yaxis_name = frame.get('yaxis', expe.get('fig_yaxis'))
                    frame.ax().set_ylabel(yaxis_name)
                else:
                    frame.ax().set_ylabel(attribute)


                if 'title_size' in self.expe:
                    ts =  float(self.expe['title_size'])
                else:
                    ts = 15

                if 'title' in frame:
                    plt.suptitle(frame.title, fontsize=ts)
                else:
                    frame.ax().set_title(title, fontsize=ts)

                if 'ticks_size' in self.expe:
                     plt.xticks(fontsize=float((self.expe['ticks_size'])))

                # Save on last call
                if self._it == self.expe_size -1:
                    if self.expe._write:
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
                discr_args = [] # discriminant keys (to distinguish filename)
                if len(args) == 1:
                    x, y, z = 'corpus', 'model', '_entropy'
                else:
                    x, y, z = args[1].split(':')
                    if len(args) > 2:
                        if ',' in args[2]:
                            split_sep = '|'
                            discr_args = args[2].split(',')
                        else:
                            split_sep = '/'
                            discr_args = args[2].split('/')

                        kwargs['split_sep'] = split_sep

                if discr_args:
                    groups = discr_args
                    self._groups = groups
                    # or None if args not in expe (tex option...)
                    ggroup = self.get_current_group() or None
                else:
                    groups = None
                    ggroup = None

                if not hasattr(self.gramexp, '_tables'):
                # self.is_first_expe

                    tablefmt_ext = dict(simple='md', latex='tex')
                    tablefmt = 'latex' if 'tex' in discr_args else 'simple'

                    tables = OrderedDict()
                    if groups:
                        gset = product(*filter(None, [self.gramexp.get_set(g) for g in groups]))
                    else:
                        gset = [None]

                    for g in sorted(gset):
                        gg = '-'.join(sorted(map(str,g))) if g else None
                        tables[gg] = ExpSpace()
                        _z = z.split('-')
                        array, floc = self.gramexp.get_array_loc(x, y, _z, repeat=kwargs.get('repeat'))
                        tables[gg].name = gg
                        tables[gg].array = array
                        tables[gg].floc = floc
                        tables[gg].x = x
                        tables[gg].y = y
                        tables[gg].z = _z
                        tables[gg].headers = self.specname(self.gramexp.get_set(y))
                        tables[gg].column = self.specname(self.gramexp.get_set(x))
                        tables[gg].floatfmt = '.3f'
                        tables[gg].tablefmt = tablefmt
                        tables[gg].ext = tablefmt_ext[tablefmt]
                        tables[gg].args = discr_args
                        tables[gg].kwargs = kwargs
                        tables[gg].base = '_'.join((fun.__name__,
                                                str(self.expe[x]),
                                                str(self.expe[y]))),

                    self.gramexp._tables = tables

                frame = self.gramexp._tables[ggroup]
                floc = frame.floc
                if kwargs.get('repeat'):
                    repeat_pos = self.pt['_repeat']
                    array = frame.array[repeat_pos]
                else:
                    array = frame.array

                for z in frame.z:
                    kernel = fun(self, array, floc, x, y, z, *discr_args)

                if self._it == self.expe_size -1:
                    tables = []

                    self.aggregate_tables(**kwargs)
                    self.decompose_tables(**kwargs)

                    self.log_tables(**kwargs)

                    if self.expe._write:
                        self.write_frames(self.gramexp._tables)

                return kernel
            return wrapper

        return decorator


    def aggregate_tables(self, **kwargs):
        ''' Group table if separator is "|" '''

        split_sep = kwargs.get('split_sep')

        if split_sep == '|':
            titles = list(self.gramexp._tables.keys())
            tables = list(self.gramexp._tables.values())
            title = '|'.join(titles)

            # Assume all columns are the same.
            t0 = tables[0]
            headers = len(titles) * t0.headers
            rp = 1 if 'repeat' in kwargs else 0
            new_array = np.concatenate([tt.array for tt in tables], 1+rp)
            t0.array = new_array
            t0.headers = headers

            self.gramexp._tables.clear()
            self.gramexp._tables[title] = t0


    def decompose_tables(self, **kwargs):

        if kwargs.get('repeat'):
            for ggroup in list(self.gramexp._tables):
                _table = self.gramexp._tables.pop(ggroup)
                array = _table.array
                for zpos, z in enumerate(_table.z):

                    fop = np.max if 'rmax' in _table.args else np.mean
                    fop = np.min if 'rmin' in _table.args else fop

                    # Mean and standard deviation
                    table = array[:,:,:,zpos]
                    mtable = self.highlight_table(np.around(fop(table, 0), decimals=3))
                    vtable = np.around(table.std(0), decimals=3)
                    table_mean = np.char.array(mtable).astype("|S42")
                    table_std = np.char.array(vtable).astype("|S42")
                    arr = table_mean + b' pm ' + table_std

                    new_table = _table.copy()
                    new_table.array = arr
                    ngg = z +'-'+ ggroup if ggroup else z
                    self.gramexp._tables[ngg] = new_table

        else:
            for ggroup in list(self.gramexp._tables):
                _table = self.gramexp._tables.pop(ggroup)
                array = _table.array
                for zpos, z in enumerate(_table.z):
                    arr = self.highlight_table(array[:,:,zpos])

                    new_table = _table.copy()
                    new_table.array = arr
                    ngg = z +'-'+ ggroup if ggroup else z
                    self.gramexp._tables[ngg] = new_table

    def log_tables(self, **kwargs):

        for _title, _table in self.gramexp._tables.items():

            arr = _table.array.astype(str)
            table = np.column_stack((_table.column, arr))
            Table = tabulate(table, headers=_table.headers, tablefmt=_table.tablefmt, floatfmt=_table.floatfmt)

            print(colored('\n%s Table:'%(_title), 'green'))
            print(Table)

    @staticmethod
    def _file_part(group, sep='_'):
        part = [str(e) for e in group if (e != None and e != 'None')]
        part = sep.join(part)
        return part

    def highlight_table(self, array, highlight_dim=1):
        hack_float = np.vectorize(lambda x : '{:.3f}'.format(float(x)))
        table = np.char.array(hack_float(array), itemsize=42)
        # vectorize
        for i, col in enumerate(array.argmax(1)):
            table[i, col] = colored(table[i, col], 'bold')
        for i, col in enumerate(array.argmin(1)):
            table[i, col] = colored(table[i, col], 'magenta')

        return table


    def write_frames(self, frames, base='', suffix='', args=''):
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
                idi = str(i) if len(frames) > 1 else None
                fn = self._file_part([base, args, suffix, idi])
                fn = self.full_fig_path(fn)
                self._kernel_write(f, fn)
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
                self._kernel_write(f, fn, title=c)
        else:
            print('Error : type of Frame unknow, passing: %s' % type(frame))


    def _kernel_write(self, frame, fn, title=None):

        if hasattr(frame, 'get'):
            ext = frame.get('ext')
        else:
            ext = None

        if isinstance(frame, dict):
            if 'fig' in frame:
                ext = ext or 'pdf'
                fn = fn +'.'+ ext
                print('Writing frame: %s' % fn)
                #frame.fig.tight_layout() # works better in parameter
                frame['fig'].savefig(fn, bbox_inches='tight') # pad_inches=-1
            elif '_table' in frame:
                ext = ext or 'md'
                fn = fn +'.'+ ext
                print('Writing frame: %s' % fn)
                caption = '\caption{{{title}}}\n'

                arr = frame['array'].astype(str)
                table = np.dstack((frame['column'], arr))
                Table = tabulate(table, headers=frame['headers'], tablefmt=frame['tablefmt'], floatfmt=frame['floatfmt'])

                with open(fn, 'w') as _f:
                    if  title:
                        _f.write(caption.format(title=title))
                    _f.write(Table+'\n')

        elif isinstance(frame, str):
            ext = ext or 'md'
            fn = fn +'.'+ ext
            print('Writing frame: %s' % fn)
            with open(fn, 'w') as _f:
                _f.write(frame)
        else:
            # assume figure
            ext = ext or 'pdf'
            fn = fn +'.'+ ext
            print('Writing frame: %s' % fn)
            frame.savefig(fn, bbox_inches='tight')

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




    @classmethod
    def _preprocess_(cls, gramexp):
        ''' This method has **write** access to Gramexp

            Notes
            -----
            Called once before running expe.
        '''

        # Update defautl settings of  gramexp
        # given _default_expe
        gramexp.update_default_expe(cls)

        # Put a valid expe a the end.
        gramexp.reorder_firstnonvalid()

        if not gramexp._conf.get('simulate'):
            cls.log.info(gramexp.exptable())

        # Global container shared by all expe
        # running in the sandbox.
        gramexp.D = ExpSpace()

        return

    @classmethod
    def _postprocess_(cls, gramexp):
        '''
            Notes
            -----
            Called once after all expe are finished.
        '''
        cls.display(gramexp._conf)
        return

    def _expe_preprocess(self):
        ''' system preprocess '''
        # setup seed ?
        # setup branch ?
        # setup description ?

        if hasattr(self, '_preprocess'):
            self._preprocess()

    def _expe_postprocess(self):
        ''' system postprocess '''

        if self.expe.get('_write'):

            # @improve: In ModelManager ?
            self.clear_fitfile()
            if hasattr(self, 'model') and hasattr(self.model, 'save'):
                if hasattr(self.model, 'write_current_state'):
                    self.model.save()

        if hasattr(self, '_postprocess'):
            self._postprocess()

    def _preprocess(self):
        ''' user defined preprocess '''
        # heere, do a decorator ?
        pass

    def _postprocess(self):
        ''' user defined postprocess '''
        # heere, do a decorator ?
        pass

    def __call__(self):
        raise NotImplementedError


    def _extract_csv_sample(self, model):
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

                @todo use _fmt if given.
        '''
        line = []
        for o in self._csv_typo.split():
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
            else: # is atomic ie can be converted to string.
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

    def init_fitfile(self):
        ''' Create the file to save the iterations state of a model. '''
        self._samples = []
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.fname_i = self.output_path + '.inf'
        if not '_csv_typo' in self.expe:
            self.log.debug('No _csv_typo, for this model %s, no inference file...'%(self.expe.get('model')))
        else:
            self._fitit_f = open(self.fname_i, 'wb')
            self._fitit_f.write(('#' + self._csv_typo + '\n').encode('utf8'))


    def clear_fitfile(self):
        ''' Write remaining data and close the file. '''

        if not hasattr(self, '_fitit_f'):
            return

        if hasattr(self, '_samples') and self._samples:
            self._write_some(self._fitit_f, None)

        end_line = '# terminated\n'
        self._fitit_f.write(end_line.encode('utf8'))

        self._fitit_f.close()


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
        col_typo = self._csv_typo.split()
        array = []
        last_elt_size = None
        _pos_failed = set()
        # Format the data in a list of dict by entry
        for data_line in data:
            line = {}
            offset = 0
            for true_pos in range(len(data_line)):
                pos = true_pos + offset
                if pos >= len(data_line):
                    break

                try:
                    o = col_typo[true_pos]
                except:
                    #print(col_typo)
                    #print(data_line)
                    #print(true_pos)
                    if true_pos not in _pos_failed:
                        self.log.warning('No value found for pos: %d' % true_pos)
                        _pos_failed.add(true_pos)
                    continue

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
                        # size of the list behind.
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


    def _write_some(self, _f, samples, buff=20):
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
                # @debug manage float .4f !
                line = ' '.join(line)+'\n'
                _f.write(line.encode('utf8'))

            _f.flush()
            self._samples = []


    def write_current_state(self, model):
        ''' push the current state of a model in the output file. '''
        samples = self._extract_csv_sample(model)
        self._write_some(self._fitit_f, samples)


    def configure_model(self, model):
        ''' Configure Model:
            * [warning] it removes existing [expe_path].inf file
        '''

        self.model = model

        # Inject the inline-writing method
        setattr(model, 'write_current_state', self.write_current_state)

        if self.expe.get('_write'):
            self.init_fitfile()

        # Could configure frontend/data path or more also here ?
        return

    # frontend params is deprecated and will be removed soon...
    def load_model(self, frontend=None, load=False):
        ''' :load: boolean. Load from **preprocess** file is true else
                            it is a raw loading.
        '''
        from pymake.frontend.manager import ModelManager

        self.model = ModelManager.from_expe(self.expe, frontend=frontend, load=load)
        if load is False:
            self.configure_model(self.model)

        return self.model

    def load_frontend(self):
        ''' See -nld and -sld option for control over load/save status
            of frontend data.
        '''
        from pymake.frontend.manager import FrontendManager

        frontend = FrontendManager.load(self.expe)
        return frontend



