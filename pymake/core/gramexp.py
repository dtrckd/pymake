import random
import sys, os, re, uuid
import argparse
from datetime import datetime
import logging
import operator
import fnmatch
import subprocess
import shlex
import inspect, traceback, importlib
import pkg_resources
from collections import defaultdict, OrderedDict
from functools import reduce, wraps
from copy import deepcopy
import shelve
import numpy as np

from pymake.core import get_pymake_settings, reset_pymake_settings
from pymake.core.types import ExpDesign, ExpTensor, ExpSpace, Model, Corpus, Script, Spec, ExpVector
from pymake.core.types import ExpTensorV2 # @debug name integration
from pymake.core.format import ExpeFormat

from pymake.io import is_empty_file
from pymake.util.utils import colored, basestring, hash_objects, PmkTemplate



''' Grammar Expe '''



# Custom formatter
# From : https://stackoverflow.com/questions/14844970/modifying-logging-message-format-based-on-message-logging-level-in-python3
class MyLogFormatter(logging.Formatter):

    critical_fmt  = "====>>> CRITICAL: %(msg)s"
    err_fmt  = "===>> ERROR: %(msg)s"
    warn_fmt  = "==> Warning: %(msg)s"
    #info_fmt = '%(module): %(msg)s'
    #info_fmt = '%(levelno)d: %(msg)s'
    default_fmt = '%(msg)s'

    # Custom Log
    VDEBUG_LEVEL_NUM = 9
    logging.addLevelName(VDEBUG_LEVEL_NUM, "VDEBUG")
    logging.VDEBUG = 9

    def __init__(self, fmt=default_fmt):
        super().__init__(fmt=fmt, datefmt=None, style='%')

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.WARNING:
            self._style._fmt = MyLogFormatter.warn_fmt
        elif record.levelno == logging.ERROR:
            self._style._fmt = MyLogFormatter.err_fmt
        elif record.levelno == logging.CRITICAL:
            self._style._fmt = MyLogFormatter.critical_fmt
        else:
            self._style._fmt = MyLogFormatter.default_fmt


        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result


def setup_logger(level=None, name='root'):
    #formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    if level is None:
        level = 0

    if level == 1:
        level = logging.DEBUG
    elif level >= 2:
        level = logging.VDEBUG
    elif level == -1:
        level = logging.WARNING
    else:
        # make a --silent option for juste error and critial log ?
        level = logging.INFO

    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Format handler
    handler = logging.StreamHandler(sys.stdout)
    #handler.setFormatter(logging.Formatter(fmt=fmt))
    handler.setFormatter(MyLogFormatter())

    # Set Logger
    logger.addHandler(handler)
    # Prevent logging propagation of handler,
    # who reults in logging things multiple times
    logger.propagate = False


    def vdebug(self, message, *args, **kws):
        # Yes, logger takes its '*args' as 'args'.
        if self.isEnabledFor(logging.VDEBUG):
            self._log(logging.VDEBUG, message, args, **kws)

    #logger.Logger.vdebug = vdebug
    logging.Logger.vdebug = vdebug

    return logger


class GramExp(object):
    ''' Create a mapping between different format of design of experiments.

        Attribute
        ---------
        _tensors :ExpTensor
            tensor of experiments

        Methods
        -------
        __init__ : conf choice in :
           * Expe -> One experiments
           * ExpTensor -> Design of experiments
           * Expe & spec in Expe -> Design of experiments with global value,
                either from conf or command-line args.
                conf value will udpdate all experiments first from command-line
                argument then from defaut values.

        Notes
        -----

        Design of Experiments can take three forms :
        1. command-line arguments (see self.parsearg).
            * use for parralization with Gnu parallel.
        2. Expe : Uniq experiments specificattion,
            * @Todo list of Expe
        3. ExpTensor : Represente a tensor of experiments.
           Each entry of the dict contains an iterable over specifications
           for this entry. (see frontend.io)
           * @Todo pass rule to filter experiments...

        ### Expe Filtering
        Expe can contains **special** keywords and value :
        * _bind rules : List of rules of constraintsi on the design.
            * Inclusif rules
                + [a.b]  --- a(value) shoudld occur only with b(value), (ie 'lda.fast_init')
                + [a.b.c] --- with a(value), key b(key) take only c (value), (ie 'lda.iterations.100')
            * Exclusif Rules
                + [a.!b]  --- a(value) shoudld not only with b(value),
                + [a.b.!c] --- with a(value), key b(key) don't take c (value),

            Warning : it does only check the last words when parameter are
                      separated by a dot (.) as for model module (name pmk.lda !) for example.

        * if an attribute  take the value "_null",
          then it will be removed from the Exp. This can be usefull when piping
          ouptuf of pymake to some script to parallelize.

        ### I/O Concerns
        An experiment typically write three kind of files :
        Corpus are load/saved using Pickle format in:
        * bdir/corpus_name.pk
        Models are load/saved using pickle/json/cvs in :
        * bdir/refdir/rep/model_name_parameters.pk   <--> ModelManager
        * bdir/refir/rep/model_name_parameters.json <--> DataBase
        * bdir/refdir/rep/model_name_parameters.inf  <--> ModelBase

    '''

    log = logging.getLogger('root')

    _version = pkg_resources.get_distribution('pmk').version

    _examples = '''Examples:
        >> Text fitting
        fit.py -k 6 -m ldafullbaye -p
        >> Network fitting :
        fit.py -m immsb -c alternate -n 100 -i 10'''

    _frontend_ext = ['gt', # graph-tool
                     'pk', # pickle
                    ]
    #_model_ext = @Todo: dense(numpy/pk.gz) or sparse => gt...?

    # has special semantics on **output_path**.
    _special_keywords = [ '_refdir', '_repeat',
                         '_format', '_csv_typo',
                        ] # output_path => pmk-basedire/{base_type}/{refdir}/{repeat}/${format}:${csv_typo}

    # Reserved by GramExp for expe identification
    _reserved_keywords = ['_spec', # for nested specification.
                          '_id_expe', # unique (locally) expe identifier.
                          '_name_expe', # exp name identifier, set before _default_expe
                          '_expe_hash', # Unique spec identifier.
                          '_pmk', # force to set some settings out of grammarg.
                         ]
    _private_keywords = _reserved_keywords + _special_keywords

    _default_expe = {
        #'_verbose'   : 0,
        '_write'     : False,

        '_ignore_format_unique' : False,

        '_force_load_data' : True, # if True, it will force the raw data parsing.
        '_force_save_data' : True, # if False, dont save corpus as pk/gt ont the filesystem.
        '_no_block_plot' : False,
    }


    _env = None # should be set during bootstrap
    _base_name = '.pmk'
    _cfg_name = 'pmk.cfg'
    _db_name = '.pmk-db'
    _pmk_error_file = '.pymake-errors'


    def __init__(self, conf, usage=None, parser=None, parseargs=True, expdesign=None):
        # @logger One logger by Expe ! # in preinit
        setup_logger(level=conf.get('_verbose'))

        if conf is None:
            conf = {}

        if parseargs is True:
            kwargs, self.argparser, _ = self.parseargsexpe(usage)
            conf.update(kwargs)
        if parser is not None:
            # merge parser an parsargs ?
            self.argparser = parser

        # It pollute the spec output...
        #[conf.update({k:v}) for k,v in self._default_expe.items() if k not in conf]

        self._base_conf = conf
        #self._do = conf.get('_do')

        self.exp_setup(conf, expdesign)

    @classmethod
    def getenv(cls, a=None):
        if a is None:
            return cls._env

        _env = cls._env
        return _env.get(a)

    @classmethod
    def setenv(cls, env):

        cls._env = env

        if not cls.is_pymake_dir():
            print('fatal: Not a pymake directory: %s not found.' % (cls._cfg_name))
            exit(10)

        db = shelve.open(os.path.join(os.getcwd(), cls._db_name))
        db.update(env)
        db.close()

        cls._data_path = get_pymake_settings('project_data')

        if os.path.isfile(os.path.join(env.get('PWD'), cls._cfg_name)):
            cls._spec = Spec.get_all()
        else:
            cls._spec = {}


    def _update(self):
        # Seems obsolete, _conf is not writtent in _tensors, right ?
        self._conf = self._tensors._conf
        #
        self.lod = self._tensors._lod

    def update_default_expe(self, expformat):

        if not hasattr(expformat, '_default_expe'):
            return
        else:
            default_expe = expformat._default_expe

        # Use default _spec if no spec given
        _names = self.get_list('_name_expe')
        if len(_names) == 1 and '_default_expe' in _names and '_spec' in default_expe:
            names = default_expe['_spec']
            names = [names] if isinstance(names, str) else names
            group = []

            # Sensitive
            conf = self._base_conf
            #conf['_do'] = [self._do]

            for name in names:
                if isinstance(name, str):
                    d, expdesign = Spec.load(name, self._spec[name])
                    group.append((name, d, expdesign))
                else:
                    # Assume dict/expspace
                    group.append(('_expe', name, ExpDesign))
                    #raise NotImplementedError('use a defined spec in _default_expe')

            conf['_spec'] = group

            if 'script' in conf:
                script = conf.pop('script')
                _script, script_args = Script.get(script[0], script[1:])
                if script_args:
                    conf['_do'] = script_args
                    #self._tensors.update_all(_do=script_args)
                else:
                    conf['_do'] = ''
                    #self._tensors.update_all(_do=[''])

            self.exp_setup(conf)

        self._tensors.set_default_all(default_expe)

    def exp_setup(self, conf, expdesign=None):
        ''' work in self._tensors '''

        # Make main data structure
        self._tensors = ExpTensorV2.from_conf(conf, private_keywords=self._private_keywords,
                                              expdesign=expdesign)

        # Global settings (unique argument)
        self._conf = self._tensors.get_conf()

        # Set expTensor init.
        self._user_spec = conf.get('_spec', {})

        # makes it contextual.
        self._preprocess_exp()

        # Make lod
        skip_check = self._conf['_do'] in ['diff']
        self.lod = self._tensors.make_lod(skip_check)

        indexs = self._conf.get('_run_indexs')
        if indexs:
            self._tensors.remake(indexs)
            self._update()

    def _preprocess_exp(self):
        #self._check_exp(self._tensors)
        self._tensors.check_bind()
        self._tensors.check_model_typo()
        self._tensors.check_null()


    def io_format_check(self):
        if len(self) > 1 and '_write' in self._conf:
            self.check_format()

        # Clean pymake extra args:
        extra_args = ['_ignore_format_unique', ('_net', False)]
        keys_to_remove = []
        for _key in extra_args:
            if type(_key) is tuple:
                if self._conf.get(_key[0]) is _key[1]:
                    keys_to_remove.append(_key[0])
            elif _key in self._conf:
                    keys_to_remove.append(_key)
        # |
        for key in keys_to_remove:
            self._tensors.remove_all(key)


    def check_format(self):
        ''' Check if all expVector are distinguishable in _format.

            @debug: not valid check accros tensors !!!
            @debug: integration in ExpTensorV2 ?
        '''

        for tensor in self._tensors:
            hidden_key = []
            _format = tensor.get('_format', [''])
            if len(_format) > 1:
                raise NotImplementedError('multiple _format not implemented')
            else:
                _format = _format[0]

            format_settings = re.findall(r'{([^{}]*)}', _format)
            for setting, values in tensor.items():
                if setting in self._special_keywords:
                    continue
                if isinstance(values, list) and len(values) > 1 and setting not in format_settings:
                    hidden_key.append(setting)

            if _format and hidden_key and self._conf.get('_ignore_format_unique') is not True and not '_id' in format_settings:
                self.log.error('The following settings are not set in _format:')
                print(' '+ '  '.join(hidden_key))
                print('Possible conflicts in experience results outputs.')
                print('Please correct {_format} key to fit the experience settings.')
                print('To force the runs, use:  --ignore-format-unique')
                print('Exiting...')
                exit(2)

        if self._conf.get('_ignore_format_unique') is True:
            _hash = int((hash_objects(tensor)), 16) % 10**8
            _format = '{_name}-expe' + str(_hash) +'h'
            self._tensors.update_all(_format=_format)



    @classmethod
    def _check_exp(cls, tensor):
        ''' check format and rules of _tensors. '''
        for exp in tensor:
            # check reserved keyword are absent from exp
            for m in cls._reserved_keywords:
                if m in exp and m != '_name_expe':
                    raise ValueError('%s is a reserved keyword of gramExp.' % m)

            # Format
            assert(isinstance(exp, ExpTensor))
            for k, v in exp.items():
                if not issubclass(type(v), (list, tuple, set)):
                    raise ValueError('error, exp value should be iterable: %s' % k, v)

    def get_set(self, key, default=[]):
        ''' Return the (ordered) set of values of expVector of that {key}. '''

        # Manage special Values
        if key == '_spec':
            raise NotImplementedError
            ss =  sorted(self.get_nounique_keys())
            list_of_identifier = []
            for tensor in self._tensors:
                local_identifier = []
                for k in ss:
                    local_identifier.append(k)

                list_of_identifier.append('-'.join(sorted(filter(None,local_identifier))))
                _set = list_of_identifier

        else:
            _set = set()
            for v in self._tensors.get_all(key, default):
                if isinstance(v, (list, set, dict)):
                    self.log.debug('Unshashable value in tensor for key: %s' % key)
                    _set.add(str(v))
                else:
                    _set.add(v)

        return sorted(_set, key=lambda x: (x is None, x))


    def get_list(self, key, default=[]):
        ''' Return the list of values of expVector of that {key}. '''
        return self._tensors.get_all(key, default)

    def get_nounique_keys(self, *args):
        ''' return list of keys that are non unique in expe_tensor
            except if present in :args:.
        '''
        nk = self._tensors.get_nounique_keys()
        for o in args:
            if o in nk:
                nk.remove(o)

        return nk

    def get_array_loc(self, key1, key2, params, repeat=False):
        ''' Construct an 2d sink array.
            Return the zeros valued array of dimensions given by x and y {keys} dimension
        '''

        loc = []

        d1 = dict()
        for i, k in enumerate(self.get_set(key1)):
            d1[k] = i
        loc.append(d1)

        d2 = dict()
        for i, k in enumerate(self.get_set(key2)):
            d2[k] = i
        loc.append(d2)

        d3 = dict()
        for i, k in enumerate(params):
            d3[k] = i
        loc.append(d3)

        tr = lambda d,k:d[str(k)] if isinstance(k, (list, set, dict)) else d[k]

        floc = lambda k1, k2, z:(tr(d1,k1), tr(d2,k2), d3[z])
        shape = list(map(len, loc))
        if repeat:
            shape = [len(self.get_set('_repeat'))] + shape
        array = np.ma.array(np.empty(shape)*np.nan, mask=True)
        return array, floc


    def get_array_loc_n(self, keys, params):
        ''' Construct an nd sink array.
            Return the zeros valued array of dimensions given by x and y {keys} dimension
        '''

        loc = OrderedDict()

        for key in keys:

            d1 = dict()
            for i, v in enumerate(self.get_set(key)):
                d1[v] = i
            loc[key] = d1

        d2 = dict()
        for i, v in enumerate(params):
            d2[v] = i
        loc['_param_'] = d2

        def floc(expe, z):
            _special = ['_param_']
            pos = []
            for k in loc:
                if k in expe:
                    v = expe[k]
                elif k in _special:
                    continue
                else:
                    self.log.warning('indice unknown in floc.')
                    v = None

                pos.append(loc[k][v])

            pos.append(loc['_param_'][z])
            return tuple(pos)

        shape = [len(loc[d]) for d in loc]


        array = np.ma.array(np.empty(shape)*np.nan, mask=True)
        return array, floc

    def __len__(self):
        #return reduce(operator.mul, [len(v) for v in self._tensors.values()], 1)
        return len(self.lod)

    def make_forest_path(self, lod, ext, status='f', full_path=False):
        """ Make a list of path from a spec/dict, the filename are
            oredered need to be align with the get_from_conf_file.

            *args -> make_output_path
        """
        targets = []
        for spec in lod:
            filen = self.make_output_path(spec, ext, status=status, _null=self._tensors._null)
            if filen:
                s = filen.find(self._data_path)
                pt = 0
                if not full_path and s >= 0:
                    pt = s + len(self._data_path)
                targets.append(filen[pt:])
        return targets

    @classmethod
    def make_output_path(cls, expe, ext=None, status=None, _null=None, _nonunique=None):
        """ Make a single output path from a expe/dict
            @status: f finished
            @type: pk, json or inference.
        """
        expe = defaultdict(lambda: None, expe)

        dict_format = cls.transcript_expe(expe)
        if _null:
            dict_format.update(dict((k,None) for k in _null))

        basedir = os.path.join(cls._data_path, cls._base_name, 'results')

        hook = expe.get('_refdir', 'default')
        hook = hook.format(**dict_format)

        rep = ''
        if '_repeat' in expe and (expe['_repeat'] is not None and expe['_repeat'] is not False):
            rep = str(expe['_repeat'])
            if rep == '-1':
                rep = ''
            else:
                rep = rep.format(**dict_format)


        p = os.path.join(hook, rep)

        if not expe['_format']:
            # or give a hash if write is False ?
            cls.log.debug('No _format given, please set _format for output_path settings.')
            nonunique = ['_name_expe', '_do']
            if _nonunique:
                nonunique.extend(_nonunique)

            argnunique = []
            for i in set(nonunique):
                v = expe.get(i, 'oops')
                if isinstance(v, list):
                    v = v[0]
                argnunique.append(v)

            t = '-'.join(argnunique)
        else:
            t = expe['_format'].format(**dict_format)

        filen = os.path.join(basedir, p, t)

        if ext:
            filen = filen +'.'+ ext
        elif status and not ext:
            # Assume pickle driver for models.
            filen = filen +'.'+ 'pk'

        if status is 'f' and is_empty_file(filen):
            return  None
        else:
            return filen

    @classmethod
    def make_input_path(cls, expe, status=None):
        """ Make a single input path from a expe/dict """
        expe = defaultdict(lambda: None, expe)
        filen = None

        # Corpus is an historical exception and has its own subfolder.
        c = expe.get('corpus')
        if not c:
            c = ''
            cls.log.debug('No Corpus is given.')
        if c.lower().startswith(('clique', 'graph', 'generator')):
            c = c.replace('generator', 'Graph')
            c = c.replace('graph', 'Graph')
            c = 'generator/' + c
        if c.endswith(tuple('.'+ext for ext in cls._frontend_ext)):
            c = c[:-(1+len(c.split('.')[-1]))]

        input_dir = os.path.join(cls._data_path, cls._base_name, 'training', c)

        return input_dir

    @staticmethod
    def transcript_expe(expe):
        ''' Transcipt value to be used in spec's path formating '''

        fmt_expe = expe.copy()
        # iteration over {expe} trow a 'RuntimeError: dictionary changed size during iteration',
        # maybe due to **expe pass in argument ?

        id_str = 'expe' + str(expe['_id_expe'])
        id_name = expe['_name_expe']
        id_hash = expe['_expe_hash']

        # Special aliase for _format
        fmt_expe['_id'] = id_str
        fmt_expe['_name'] = id_name
        fmt_expe['_hash'] = id_hash

        for k, v in fmt_expe.items():
            if isinstance(v, (list, dict)):
                #fmt_expe[k] = id_str # don't do that, no robust get back expe.
                _hash = int((hash_objects(v)), 16) % 10**8
                fmt_expe[k] = k + str(_hash) +'h'
            elif isinstance(v, float):
                if 'e' in str(v):
                    fmt_expe[k] = str(v)
                elif len(str(v).split('.')[1]) > 2:
                    fmt_expe[k] = '%.2f' % v
                elif str(v).split('.')[1] == '0':
                    fmt_expe[k] = '%d' % v
                else:
                    fmt_expe[k] = v

        # Special rule to rewirte the output_path
        if '_model' in expe and 'model' in expe:
            fmt_expe['model'] = expe['_model']

        return fmt_expe



    @classmethod
    def get_parser(cls, description=None, usage=None):
        import pymake.core.gram as _gram
        parser = _gram.ExpArgumentParser(description=description, epilog=usage,
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('-V', '--version', action='version', version='%(prog)s '+ cls._version)

        return parser

    @classmethod
    def push_gramarg(cls, parser, gram=None):
        import pymake.core.gram as _gram
        if gram is None:
            gram = _gram._Gram
        else:
            try:
                gram = importlib.import_module(gram)
            except ModuleNotFoundError as e:
                pr = get_pymake_settings('gramarg').split('.')[0]
                #@imporve priority shoiuld ne on : local project !
                cls.log.warning("Project name `%s' seems to already exists in your PYTHONPATH. Project's name should not conflict with existing ones." % pr)
                cls.log.warning("Please change the name of your project/repo")
                cls.log.warning('exiting.')
                exit(38)
            gram = next( (getattr(gram, _list) for _list in dir(gram) if isinstance(getattr(gram, _list), list)), None)

        grammar = []
        args_ = []
        for e in gram:
            if not isinstance(e, dict):
                args_.append(e)
                continue
            grammar.append((args_, e))
            args_ = []

        #[parser.add_argument(*r[0], **r[1]) for r in grammar]
        # check for duplicate
        for r in grammar:
            try:
                parser.add_argument(*r[0], **r[1])
            except argparse.ArgumentError as e:
                err_mesg = " (it'll be overwritten)"
                if r[0][0] in ('-v', '-s', '-w', '--seed'):
                    # @debug, level here is not set yet, so his not active by default.
                    # parsing difficulet becaus context sensitive (option (-v) impact how the options are interpreted)
                    cls.log.debug(str(e)+err_mesg)
                else:
                    cls.log.error(str(e)+err_mesg)
                #exit(3)


    @classmethod
    def parseargsexpe(cls, usage=None, args=None, parser=None):
        description = 'Launch and Specify Simulations.'
        if not usage:
            usage = GramExp._examples

        if parser is None:
            parser = cls.get_parser(description, usage)
        #s, remaining = parser.parse_known_args(args=args)

        # Push pymake grmarg.
        cls.push_gramarg(parser)
        # third-party
        gramarg = get_pymake_settings('gramarg')
        if gramarg:
            cls.push_gramarg(parser, gramarg)

        s = parser.parse_args(args=args)

        # Assume None value are non-filled options
        settings = dict((key,value) for key, value in vars(s).items() if value is not None)

        # Purge default/unchanged settings
        for k in list(settings):
            v = settings[k]
            if cls._default_expe.get(k) == v:
                settings.pop(k)

        pmk_opts = cls.pmk_extra_opts(settings)
        settings.update(pmk_opts)
        expdesign = GramExp.expVectorLookup(settings)

        return settings, parser, expdesign

    @staticmethod
    def pmk_extra_opts(settings):
        opts = {}
        pmk = settings.get('_pmk')
        if not pmk:
            return opts

        for s in pmk:
            k, v = s.split('=')
            try:
                v = int(v)
            except:
                try: v = float(v)
                except: pass
            opts[k] = v

        return opts

    @classmethod
    def zymake(cls, request={}, usage='', firsttime=True, expdesign=None):
        usage ='''\
        ----------------
        Communicate with the data :
        ----------------
         |   pmk update                                       : update the pymake index
         |   pmk -l [spec(default)|model|script|topo]         : list available component
         |   pmk show SPEC                                    : show one spec details (default if no arguments)
         |   pmk run SPEC [--script [fun][*args]] ...         : execute tasks (default if -x is given)
         |   pmk runpara SPEC [--script [fun][*args]] ...     : parallelize tasks (implicit if --cores is given)
         |   pmk hist [-n n_lines]                            : show command history
         |   pmk cmd SPEC ...                                 : Try to generate the command-line for each expe.
         |   pmk path SPEC Filetype(pk|json|inf) [status] ... : show output_path of each expe.
        ''' + '\n' + usage

        s, parser, expdesign_lkp = GramExp.parseargsexpe(usage)
        request.update(s)

        ontology = dict(
            _do    = ['cmd', 'show', 'path', 'run', 'update', 'init', 'runpara', 'hist', 'diff'],
            _spec   = list(cls._spec),
            _ext = ['json', 'pk', 'inf']
        )
        ont_values = sum([w for k, w in ontology.items() if k != '_spec'] , [])

        # Init _do value.
        if not request.get('_do'):
            request['_do'] = []

        # Special Case for CLI.
        if 'script' in request:
            # check if no command is specified, and
            # if 'script" is there, set 'run' command as default.
            do = request.get('_do', [])
            no_do = len(do) == 0
            no_do_command = len(request['_do']) > 0 and not request['_do'][0] in ontology['_do']
            if no_do or no_do_command:
                if '_cores' in request or '_net' in request:
                    runcmd = 'runpara'
                else:
                    runcmd = 'run'
                request['_do'] = [runcmd] + do


        do = request.get('_do', [])
        checksum = len(do)
        # No more Complex !
        run_indexs = []
        expgroup = []
        for i, v in enumerate(do.copy()):
            if str.isdigit(v):
                run_indexs.append(int(v))
                do.remove(v)
                checksum -= 1
            else:
                for ont, words in ontology.items():
                    # Walktrough the ontology to find arg semantics
                    if v in words:
                        if ont == '_spec':
                            if v in ont_values:
                                cls.log.critical('=> Warning: conflict between name of ExpDesign and Pymake commands')
                            do.remove(v)
                            d, expdesign = Spec.load(v, cls._spec[v])
                            expgroup.append((v,d, expdesign))
                        else:
                            request[ont] = v

                        checksum -= 1
                        break

        # Check erros in the command line
        if checksum != 0:

            if  firsttime == True:
                cls.log.warning('Spec not found, re-building Spec indexes...')
                cls.update_index('spec')
                cls._spec = Spec.get_all()
                return cls.zymake(firsttime=False)
            else:
                cls.log.error('==> Error : unknown argument: %s\n\nAvailable Exp : %s' % (do, list(cls._spec)))
                exit(10)


        # Setup the exp inputs
        if run_indexs:
            request['_run_indexs'] = run_indexs

        if len(expgroup) >= 1:
            request['_spec'] = expgroup

        if len(expgroup) > 0 and len(do) == 0:
            request['_do'] = 'show'


        expdesign = expdesign or expdesign_lkp
        return cls(request, usage=usage, parser=parser, parseargs=False, expdesign=expdesign)


    @classmethod
    def expVectorLookup(cls, request):
        ''' set exp from spec if presents '''

        expdesign = None
        for k, v in request.items():
            if not isinstance(v, ExpVector):
                continue

            sub_request = ExpVector()
            for vv in v:
                if isinstance(vv, list):
                    continue
                if vv in cls._spec:
                    loaded_v, expdesign = Spec.load(vv, cls._spec[vv])
                    if isinstance(loaded_v, ExpVector):
                        sub_request.extend(loaded_v)
                else:
                    # Multiple Flags
                    sub_request.append(vv)

            if sub_request:
                request[k] = sub_request

        return expdesign


    @classmethod
    def exp_tabulate(cls, conf={}, usage=''):

        gargs = clargs.grouped['_'].all
        for arg in gargs:
            try:
                conf['K'] = int(arg)
            except:
                conf['model'] = arg
        return cls(conf, usage=usage)

    def make_path(self, ext=None, status=None, fullpath=None):
        return self.make_forest_path(self.lod, ext, status, fullpath)

    #@deprecated
    def make_commandline(self):
        commands = self.make_commands()
        commands = [' '.join(c) for c in commands]
        return commands

    #@deprecated
    def make_commands(self):
        lod = self.lod
        commands = []
        for e in lod:
            argback = self.argparser._actions
            command = []
            args_seen = []
            for a in argback:
                if not hasattr(a, 'option_strings') or len(a.option_strings) == 0:
                    # Assume every options of expe has a flags
                    continue
                if a.dest in e and a.dest not in args_seen:
                    #if isinstance(a, (gram.unaggregate_append,)):
                    #    # Assume except_append are removed from expe,
                    #    # special traitment
                    #    continue
                    if a.nargs == 0:
                        # standalone flags
                        store = e[a.dest]
                        if 'StoreTrue' in str(type(a)):
                            storetrue = True
                        elif 'StoreFalse' in str(type(a)):
                            storetrue = False
                        else:
                            raise ValueError('Check the Type of argparse for : %s'%e)
                        if storetrue and store:
                            command += [a.option_strings[0]]
                    else:
                        _arg = e[a.dest]
                        if isinstance(_arg, (list, set, tuple)):
                            _arg = ' '.join(str(_arg))
                        command += [a.option_strings[0]] + [str(_arg)]
                        args_seen.append(a.dest)

            commands.append(command)
        return commands

    def reorder_firstnonvalid(self, ext='pk'):
        for i, e in enumerate(self.lod):
            if not self.make_output_path(e, ext, status='f', _null=self._tensors._null, _nonunique=self.get_nounique_keys()):
                self.lod[0], self.lod[i] = self.lod[i], self.lod[0]
                break
        return

    def exptable(self):
        return self._tensors.table()

    def spectable(self):
        return Spec.table()

    def scripttable(self):
        return Script.table()

    def modeltable(self, _type='short'):
        return Model.table(_type)

    def spectable_topo(self):
        return Spec.table_topos(self._spec)

    def alltable_topo(self):
        from pymake.core.types import _table_
        specs = self._spec
        scripts = Script.get_all()
        models = Model.get_all()
        table = [models, specs, scripts]
        headers = ['Models', 'Specs', 'Actions']
        return _table_(table, headers)

    def help_short(self):
        shelp = self.argparser.format_usage() + self.argparser.epilog
        return shelp

    def simulate_short(self):
        ''' Simulation Output '''
        print('''
              Nb of experiments : %s
              Corpuses : %s
              Models : %s
              ''' % (len(self), self.getCorpuses(), self.getModels()), file=sys.stdout)
        exit()

    def simulate(self, halt=True, file=sys.stdout):

        print('PYMAKE Exp: %d' % (len(self) ), file=file)
        print('-'*30, file=file)
        print(self.exptable(), file=file)
        if halt:
            exit()
        else:
            return

    @staticmethod
    def sign_nargs(fun):
        return sum([y.default is inspect._empty for x,y in inspect.signature(fun).parameters.items() if x != 'self'])

    @staticmethod
    def tb_expeformat(sandbox):
        signatures = []
        for m in dir(sandbox):
            if not callable(getattr(sandbox, m)) or m.startswith('__') or hasattr(ExpeFormat, m):
                continue
            sgn = inspect.signature(getattr(sandbox, m))
            d = [v for k,v in sgn.parameters.items() if k != 'self'] or []
            signatures.append((m, d))
        return signatures

    @staticmethod
    def functable(obj):
        ''' show method/doc associated to one class (in /scripts) '''
        lines = []
        for m in  GramExp.tb_expeformat(obj):
            name = m[0]
            opts = []
            for o in m[1]:
                if not o:
                    continue
                if o.default is inspect._empty:
                    # args
                    opts.append(o.name)
                else:
                    # kwargs
                    opts.append('%s [%s]'% (o.name, o.default))
            opts = ', '.join(opts)
            line = '  ' + name + ': ' + opts
            lines.append(line)
        return lines

    @staticmethod
    def load(*args, **kwargs):
        from pymake.io import load
        return load(*args, **kwargs)

    @staticmethod
    def save(*args, **kwargs):
        from pymake.io import save
        return save(*args, **kwargs)


    @staticmethod
    def model_walker(bdir, fmt='list'):
        models_files = []
        if fmt == 'list':
            ### Easy formating
            for root, dirnames, filenames in os.walk(bdir):
                for filename in fnmatch.filter(filenames, '*.pk'):
                    models_files.append(os.path.join(root, filename))
            return models_files
        else:
            ### More Complex formating
            tree = { 'json': [],
                    'pk': [],
                    'inference': [] }
            for filename in fnmatch.filter(filenames, '*.pk'):
                if filename.startswith(('dico.','vocab.')):
                    dico_files.append(os.path.join(root, filename))
                else:
                    corpus_files.append(os.path.join(root, filename))
            raise NotImplementedError()
        return tree


    @staticmethod
    def get_cls_name(cls):
        clss = str(cls).split()[1]
        return clss.split('.')[-1].replace("'>", '')


    def expe_init(self, expe, _seed_path='/tmp/pmk.seed'):
        ''' Intialize an expe:
            * Set seed
            * set in/out filesystem path
        '''

        _seed = expe.get('_seed')

        if _seed is None:
            seed0 = random.randint(0, 2**128)
            seed1 = np.random.randint(0, 2**32, 10)
            seed = [seed0, seed1]
        elif type(_seed) is str and str.isdigit(_seed):
            _seed = int(_seed)
            seed = [_seed, _seed]
        elif type(_seed) is str:
            if _seed in expe:
                _seed = expe[_seed]

            seed = []
            for c in list(_seed):
                seed.append(str(ord(c)))

            if '_repeat' in expe:
                if type(expe['_repeat']) is int:
                    seed.append(str(expe['_repeat']))
                else:
                    for c in list(expe['_repeat']):
                        seed.append(str(ord(c)))

            seed = ''.join( [chr(int(i)) for i in list(''.join(seed))] )
            seed = int((hash_objects(seed)), 32) % 2**32
            seed = [seed, seed]
            # Makes it on 32 bit...

        elif _seed is True:
            # Load state
            seed = None
            try:
                self._seed = self.load(_seed_path)
                random.seed(self._seed[0])
                np.random.seed(self._seed[1])
                #np.random.set_state(seed_state)
            except FileNotFoundError as e:
                self.log.error("Cannot initialize seed, %s file does not exist." % _seed_path)
                sid = [np.random.randint(0, 2**63), np.random.randint(0, 2**32)]
                self.save(_seed_path, sid, silent=True)
                #self.save(_seed_path, np.random.get_state() silent=True)
                raise FileNotFoundError('%s file just created, try again !')

        if seed:
            # if no seed is given, it's impossible to get a seed from numpy
            # https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
            random.seed(seed[0])
            np.random.seed(seed[1])

            # Save state
            self.save(_seed_path, seed, silent=True)
            #self.save(_seed_path, [seed, np.random.get_state()], silent=True)
            self._seed = seed

        # Init I/O settings
        expe['_output_path'] = self.make_output_path(expe, _null=self._tensors._null, _nonunique=self.get_nounique_keys())
        expe['_input_path'] = self.make_input_path(expe)

        return self._seed


    def execute(self):
        ''' Execute Exp Sequentially '''

        if 'script' in self._conf:
            script = self._conf.pop('script')
            self._tensors.remove_all('script')
            self._tensors.remove_all('_do')
        else:
            self.log.error('==> Error : You need to specify a script. (--script)')
            exit(10)

        try:
            _script, script_args = Script.get(script[0], script[1:])
        except (ValueError, TypeError) as e:
            self.log.warning('Script not found, re-building Scripts indexes...')
            self.update_index('script')
            #self._spec = Spec.get_all()
            try:
                res = Script.get(script[0], script[1:])
                if not res:
                    print('error: Unknown script: %s' % (script[0]))
                    exit(404)
                else:
                    _script, script_args = res
            except:
                raise
        except IndexError as e:
                self.log.error('Script arguments error : %s -- %s' % (e, script))
                exit(2)

        # Raw search
        #script_name = script[0]
        #script_args = script[1:]
        #Scripts = mloader.ScriptsLoader.get_packages()
        #if not script_name in Scripts:
        #    method_by_cls = mloader.ScriptsLoader.lookup_methods()
        #    if script_name in sum(method_by_cls.values(), []):
        #        # check if it is a method reference
        #        script_args = [script_name] + script_args
        #        script_name = next(k.lower() for (k,v) in method_by_cls.items() if script_name in v)
        #    else:
        #        raise ValueError('error: Unknown script: %s' % (script_name))

        if script_args:
            self._tensors.update_all(_do=script_args)
        #self.pymake(sandbox=Scripts[script_name])
        self.pymake(sandbox=_script)

    def execute_parallel(self):

        basecmd = sys.argv.copy()
        #try:
        #    Target = subprocess.check_output(['which','pymake']).strip().decode()
        #except:
        #    Target = 'python3 /home/ama/adulac/.local/bin/pymake'
        Target = 'pmk'
        basecmd = [Target] + basecmd[1:]

        # Create commands indexs
        indexs = self._conf.get('_run_indexs')
        if not indexs:
            indexs = range(len(self))
        else:
            [basecmd.remove(str(i)) for i in indexs]

        # Creates a list of commands that pick each index
        # from base requests commands.
        cmdlines = []
        for index in indexs:
            id_cmd = 'run %s' % (index)
            if 'runpara' in basecmd:
                cmd = ' '.join(basecmd).replace('runpara', id_cmd, 1)
            else:
                idx = basecmd.index(Target)
                cmd = ' '.join(basecmd[:idx] + [Target + ' '+ id_cmd] + basecmd[idx+1:])
            cmdlines.append(cmd)

        n_cores = str(self._conf.get('_cores', 1))

        # remove the --cores options
        for i, cmd in enumerate(cmdlines):
            cmd = cmd.split()
            try: idx = cmd.index('--cores')
            except: continue
            cmd.pop(idx); cmd.pop(idx) # pop --cores int
            cmdlines[i] = ' '.join(cmd)


        if self._conf.get('simulate'):
            self.simulate()

        #for r in cmdlines:
        #    print(r)
        #exit()

        cmd = ['parallel', '-j', n_cores, '-u', '-C', "' '", '--eta', '--progress', ':::', '%s'%('\n'.join(cmdlines))]

        #stdout = subprocess.check_output(cmd)
        #print(stdout.decode())
        for line in self.subexecute1(cmd):
            print(line, end='')

        #self.subexecute2(cmd)

    def execute_parallel_net(self, nhosts=None):
        ''' run X process by machine !
            if :nhosts: (int) is given, limit the numner of remote machines.
        '''

        NDL = get_pymake_settings('loginfile')
        workdir = get_pymake_settings('remote_pwd')
        remotes = list(filter(None, [s for s in open(NDL).read().split('\n') if not s.startswith(('#', '%'))]))

        basecmd = sys.argv.copy()
        #Target = './zymake.py'
        #basecmd = ['python3', Target] + basecmd[1:]
        Target = 'pmk'
        basecmd = [Target] + basecmd[1:]
        cmdlines = None

        # Create commands indexs
        indexs = self._conf.get('_run_indexs')
        if not indexs:
            indexs = range(len(self))
        else:
            [basecmd.remove(str(i)) for i in indexs]

        # Get chunked indexs
        if nhosts is None:
            # cores by host
            n_cores = int(self._conf.get('_cores', 1))
        else:
            # share run per host
            n_cores = len(indexs) // int(nhosts)

            # Only --net 1, implement/understood from now.
            if int(nhosts) == 1:
                indexs = []
                cmdlines = [' '.join(basecmd)]
            else:
                raise NotImplementedError

        if cmdlines is None:
            indexs = list(map(str, indexs))
            chunk = int(np.ceil(len(indexs) / len(remotes)))
            indexs = [indexs[i:i+chunk] for i in range(0, len(indexs), chunk)]

            # Creates a list of commands that pick each index
            # from base requests commands.
            cmdlines = []
            for index in indexs:
                id_cmd = '%s' % (' '.join(index))
                if 'runpara' in basecmd:
                    cmd = ' '.join(basecmd).replace('runpara', id_cmd, 1)
                else:
                    idx = basecmd.index(Target)
                    cmd = ' '.join(basecmd[:idx] + [Target + ' '+ id_cmd] + basecmd[idx+1:])

                cmdlines.append(cmd)

        # remove the --net options
        for i, cmd in enumerate(cmdlines):
            #cmdlines[i] = cmd.replace('--net', '').strip()
            cmdlines[i] = re.sub(r'--net\s*[0-9]*', '', cmd).strip()

        if nhosts is not None:
            tempf = '/tmp/pmk_' + uuid.uuid4().hex
            nhosts = int(nhosts)
            with open(tempf, 'w') as _f_w:
                with open(NDL) as _f_r:
                    for i, l in enumerate(_f_r.readlines()):
                        if i >= nhosts:
                            break
                        _f_w.write(l)
            NDL = tempf

        #for r in cmdlines:
        #    print(r)
        #exit()

        cmd = ['parallel', '-u', '-C', "' '", '--eta', '--progress',
               '--sshloginfile', NDL, '--workdir', workdir,
               '--env', 'OMP_NUM_THREADS', '--env', 'PYTHONPATH', '--env', 'PATH',
               ':::', '%s'%('\n'.join(cmdlines))]

        env = {'PYTHONPATH': '~/.local/lib/:',
               'PATH': '~/.local/bin:/usr/local/bin:/usr/bin:/bin:'}

        if self._conf.get('simulate'):
            self.simulate()


        #stdout = subprocess.check_output(cmd)
        #print(stdout.decode())
        for line in self.subexecute1(cmd, **env):
            print(line, end='')

    @staticmethod
    def subexecute1(cmd, **env):
        _env = os.environ
        _env.update(env)
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, env=_env)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

    @staticmethod
    def subexecute2(cmd):
        ''' trying to print colors here ...'''
        popen = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, universal_newlines=True)
        while True:
            output = popen.stdout.readline()
            if output == '' and popen.poll() is not None:
                break
            if output:
                print(popen.strip())
        return process.poll()

    def notebook(self):
        from nbformat import v4 as nbf
        nb = nbf.new_notebook()

        text = ''
        # get the expe
        # get the script
        # get the figure
        code = ''
        nb['cells'] = [nbf.new_markdown_cell(text),
                       nbf.new_code_cell(code) ]
        return

    @classmethod
    def is_pymake_dir(cls):
        pwd = cls.getenv('PWD')
        return os.path.isfile(os.path.join(pwd, cls._cfg_name))

    def init_folders(self):
        from pymake.util.utils import reset_pymake_settings
        join = os.path.join

        if self.is_pymake_dir():
            print('fatal: %s file already exists.' % (self._cfg_name))
            exit(11)

        pwd = self.getenv('PWD')
        cwd = os.path.dirname(__file__)
        folders = ['_current', 'spec', 'script', 'model']
        #open(join(pwd, '__init__.py'), 'a').close()
        spec = {'projectname':os.path.basename(pwd)}
        print('Creating project: {projectname}'.format(**spec))

        settings = {}
        for d in folders:
            if d in ['model', 'spec', 'script']:
                os.makedirs(d, exist_ok=True)
                with open(join(cwd,'..', 'template', '%s.template'%(d))) as _f:
                    template = PmkTemplate(_f.read())
                open(join(pwd, d,  '__init__.py'), 'a').close()
                with open(join(pwd, d,  'template_%s.py'%(d)), 'a') as _f:
                    _f.write(template.substitute(spec))
                settings.update({'default_%s'%(d):'.'.join((spec['projectname'], d))})
            elif d == '_current':
                # Gramarg
                with open(join(cwd,'..', 'template', 'gramarg.template')) as _f:
                    template = PmkTemplate(_f.read())
                with open(join(pwd, 'gramarg.py'), 'a') as _f:
                    _f.write(template.substitute(spec))
                settings.update({'default_gramarg':'.'.join((spec['projectname'], 'gramarg'))})
            else:
                raise ValueError('Doh, Directory unknwow: %s' % d)


        # Set and write pymake.cfg
        reset_pymake_settings(settings)

        return self.update_index()

    def pushcmd2hist(self):
        from pymake.util.utils import tail
        bdir = os.path.join(self._data_path, self._base_name)
        fn = os.path.join(bdir, '.pymake-hist')
        if not os.path.isfile(fn):
            open(fn, 'a').close()
            return

        cmd = sys.argv.copy()
        cmd[0] = os.path.basename(cmd[0])
        cmd = ' '.join(cmd)
        _tail = tail(fn, 10)

        with open(fn, 'a') as _f:
            if not cmd in _tail:
                return _f.write(cmd+'\n')


    def show_history(self):
        from pymake.util.utils import tail
        n = self._conf.get('N', 42)
        if n == 'all':
            n_lines = -1
        else:
            n_lines = int(n)

        bdir = self._data_path
        fn = os.path.join(bdir, self._base_name, '.pymake-hist')
        if not os.path.isfile(fn):
            self.log.error('hist file does not exist.')
            return

        _tail = tail(fn, n_lines)

        print('\n'.join(_tail))

    def show_diff(self):
        from tabulate import tabulate
        import itertools

        diff_expe = dict()
        for tensor in self._tensors:
            for k in self._tensors.get_keys():

                if k not in diff_expe:
                    diff_expe[k] = tensor.get(k, ['--'])
                    continue

                v = tensor.get(k, ['--'])
                s = diff_expe[k]
                r = list(itertools.filterfalse(lambda x: x in v, s)) + list(itertools.filterfalse(lambda x: x in s, v))
                #try:
                #    r = (set(s) - set(v)).union(set(v) - set(s))
                #except TypeError as e:
                #    r = list(itertools.filterfalse(lambda x: x in v, s)) + list(itertools.filterfalse(lambda x: x in s, v))


                diff_expe[k] = r


        for k in list(diff_expe):
            if len(diff_expe[k]) == 0:
                diff_expe.pop(k)

        if diff_expe:
            print('exp differs:')
            print(tabulate(diff_expe.items()))

        exit()


    @classmethod
    def update_index(cls, *index_name):
        from pymake.index.indexmanager import IndexManager as IX

        pwd = cls.getenv('PWD')
        os.chdir(pwd)

        ## Update index
        if len(index_name) == 0:
            IX.build_indexes()
        else:
            for name in index_name:
                IX.build_indexes(name)

        ## Update bash_completion file
        home = os.path.expanduser('~')
        cwd = os.path.dirname(__file__)
        pjt = os.path.basename(pwd)

        template = None
        completion_fn = os.path.join(home, '.bash_completion.d', 'pymake_completion')
        if os.path.exists(completion_fn):
            with open(completion_fn) as _f:
                template = _f.read()

            # Reset completion file if version differs
            verpos = template.find('%%PMK')
            _ver = re.search(r'version=([0-9\.\-a-zA-Z_]*)', template[verpos:])
            if not _ver:
                template = None
            else:
                _ver = _ver.groups()[0]
                if _ver != cls._version:
                    template = None

        if template is None:
            with open(os.path.join(cwd, '..', 'template', 'pymake_completion.template')) as _f:
                template = _f.read()


        # Get Specs
        specs = ' '.join(list(cls._spec))

        # Get Scripts
        _scripts = Script.get_all(_type='hierarchical')
        scripts = defaultdict(list)
        all_scripts = set()
        sur_scripts = set()
        for _o in _scripts:
            script = _o['scriptsurname']
            action = _o['method']
            scripts[script].append(action)
            all_scripts.update([script, action])
            sur_scripts.add(script)

        # Create a Bash array of strings.
        dict_scripts = []
        for sur in sur_scripts:
            dict_scripts.append('"'+ ' '.join(scripts[sur]) +'"')

        # get Models
        models = None

        # get Corpus
        corpus = None


        all_scripts = ' '.join(all_scripts)
        sur_scripts = ' '.join(sur_scripts)
        dict_scripts = '(' + ' '.join(dict_scripts) + ')'
        hook = '''
                elif [[ "$project" == "$$projectname" ]]; then
                    specs="$$specs"
                    all_scripts="$$all_scripts"
                    sur_scripts="$$sur_scripts"
                    dict_scripts=$$dict_scripts
               '''

        _match = '[[ "$project" == "%s" ]]' % (pjt)
        back_pos = template.find(_match)
        if back_pos  >= 0:
            # Remove old lines
            template = template.split('\n')
            pt = None
            for pos, line in enumerate(template):
                if _match in line:
                    pt = pos
                    break
            _hook = hook.strip().split('\n')
            n_lines = len(_hook)
            template = template[:pt] +_hook+ template[pt+n_lines:]
            template = '\n'.join(template)
        else:
            insert_pos = template.find('%%PMK')
            insert_pos = insert_pos - template[:insert_pos][::-1].find('\n')
            template = template[:insert_pos] +\
                    hook +\
                    template[insert_pos:]

        os.makedirs(os.path.join(home, '.bash_completion.d'), exist_ok=True)
        with open(completion_fn, 'w') as _f:
            template = PmkTemplate(template)
            template = template.substitute(projectname=pjt, version=cls._version,
                                           specs=specs,
                                           all_scripts=all_scripts,
                                           sur_scripts=sur_scripts,
                                           dict_scripts=dict_scripts)
            _f.write(template)

        #os.execl("/bin/bash", "/bin/bash", "-c", "source ~/.bash_completion.d/pymake_completion")

    def pymake(self, sandbox=ExpeFormat):
        ''' Walk Through experiments. '''

        if 'do_list' in self._conf:
            print('Available methods for %s: ' % (sandbox))
            print(*self.functable(sandbox) , sep='\n')
            exit()

        # Default spec, expVector ?
        #if hasattr(sandbox, '_expe_default'):
        #    print(sandbox._expe_default)

        sandbox._preprocess_(self)
        self.io_format_check()

        if self._conf.get('simulate'):
            self.simulate()

        n_errors = 0
        for id_expe, _expe in enumerate(self.lod):
            expe = ExpSpace(**_expe)

            pt = dict((key, value.index(expe[key])) for key, value in self._tensors.get_gt().items()
                      if (isinstance(expe.get(key), (basestring, int, float)) and key not in self._reserved_keywords))
            pt['expe'] = id_expe

            # Init Expe
            expdesign = self._tensors._ds[id_expe]
            self.expe_init(expe)

            try:
                expbox = sandbox(pt, expe, expdesign, self)
            except FileNotFoundError as e:
                self.log.error('ignoring %s'%e)
                continue
            except Exception as e:
                print(('Error during '+colored('%s', 'red')+' Initialization.') % (str(sandbox)))
                traceback.print_exc()
                exit(2)

            # Expe Preprocess
            expbox._expe_preprocess()

            # Setup handler
            if '_do' in expe and len(expe._do) > 0:
                # ExpFormat task ? decorator and autoamtic argument passing ....
                do = expe._do
                pmk = getattr(expbox, do[0])
            else:
                do = ['__call__']
                pmk = expbox

            # Launch handler
            args = do[1:]
            try:
                res = pmk(*args)
                if res is not None:
                    print(res)
            except KeyboardInterrupt:
                # it's hard to detach matplotlib...
                traceback.print_exc(file=sys.stdout)
                break
            except Exception as e:
                n_errors += 1
                self.log.critical(('Error during '+colored('%s', 'red')+' Expe no %d.') % (do, id_expe))
                traceback.print_exc()
                with open(self._pmk_error_file, 'a') as _f:
                    lines = []
                    lines.append('%s' % (datetime.now()))
                    lines.append('Error during %s Expe no %d' % (do, id_expe))
                    lines.append('Output path: %s' % (expe.get('_output_path')))
                    _f.write('\n'.join(lines) + '\n')
                    traceback.print_exc(file=_f)
                    _f.write('\n')
                continue

            # Expe Postprocess
            expbox._expe_postprocess()

        if n_errors > 0:
            self.log.warning("There was %d errors, logged in `%s'" % (n_errors, self._pmk_error_file))
            with open(self._pmk_error_file, 'a') as _f:
                _f.write(100*'='+'\n')

        sandbox._postprocess_(self)
        self.pushcmd2hist()

        exit_status = 0 if n_errors == 0 else 1
        return exit(exit_status)




