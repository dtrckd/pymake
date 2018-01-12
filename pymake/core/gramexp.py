# -*- coding: utf-8 -*-

import shlex
import uuid
import sys
import os
from datetime import datetime
import re
import logging
import operator
import fnmatch
import pickle
import subprocess
import inspect, traceback, importlib
from collections import defaultdict
from functools import reduce, wraps
from copy import deepcopy
import numpy as np

import argparse

from pymake import ExpDesign, ExpTensor, ExpSpace, ExpeFormat, Model, Corpus, Script, Spec, ExpVector
from pymake.util.utils import colored, basestring, get_global_settings

# @debug name integration
from pymake.core.format import ExpTensorV2

import pymake.frontend.frontend_io as mloader

from pymake.frontend.frontend_io import _DATA_PATH, ext_status, is_empty_file


''' Grammar Expe '''
_version = 0.1
lgg = logging.getLogger('root')

# Custom formatter
# From : https://stackoverflow.com/questions/14844970/modifying-logging-message-format-based-on-message-logging-level-in-python3
class MyLogFormatter(logging.Formatter):

    critical_fmt  = "====>>> CRITICAL: %(msg)s"
    err_fmt  = "===>> ERROR: %(msg)s"
    warn_fmt  = "==> Warning: %(msg)s"
    #info_fmt = '%(module): %(msg)s'
    #info_fmt = '%(levelno)d: %(msg)s'
    default_fmt = '%(msg)s'

    # CUstom Level
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


def setup_logger(level=logging.INFO, name='root'):
    #formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    if level == 1:
        level = logging.DEBUG
    elif level == 2:
        level = logging.VDEBUG
    elif level == 3:
        print ('what level of verbosity heeere ?')
        exit(2)
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
        exp_tensor :ExpTensor
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
           for this entry. (see frontend.frontend_io)
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
    _examples = '''Examples:
        >> Text fitting
        fit.py -k 6 -m ldafullbaye -p
        >> Network fitting :
        fit.py -m immsb -c alternate -n 100 -i 10'''

    # has special semantics on **output_path**.
    _special_keywords = [ '_refdir', '_data_type',
                         '_format', '_csv_typo',
                         '_repeat',
                        ] # output_path => pmk-basedire/{base_type}/{refdir}/{repeat}/${format}:${csv_typo}

    # Reserved by GramExp for expe identification
    _reserved_keywords = ['_spec', # for nested specification.
                          '_id_expe', # unique expe identifier.
                          '_name_expe', # exp name identifier.
                         ]
    _private_keywords = _reserved_keywords + _special_keywords

    _exp_default = {
        #'host'      : 'localhost',
        'verbose'   : logging.INFO,
        'write'     : False,
        #'_load_data' : True, # if .pk corpus is here, load it.
        #'_save_data' : False, # save corpus as .pk.
    }

    _pmk_error_file = '.pymake-errors'
    _spec = Spec.get_all() #_spec = mloader.SpecLoader.default_spec()

    def __init__(self, conf, usage=None, parser=None, parseargs=True, expdesign=None):
        # @logger One logger by Expe ! # in preinit
        setup_logger(level=conf.get('verbose'))

        if conf is None:
            conf = {}

        if parseargs is True:
            kwargs, self.argparser, _ = self.parseargsexpe(usage)
            conf.update(kwargs)
        if parser is not None:
            # merge parser an parsargs ?
            self.argparser = parser

        [conf.update({k:v}) for k,v in self._exp_default.items() if k not in conf]
        #conf = deepcopy(conf) # who want this ?

        # Set expTensor init.
        self._user_spec = conf.get('_spec', {})

        # Make main data structure
        self.exp_tensor = ExpTensorV2.from_conf(conf, private_keywords=self._private_keywords)
        self.exp_setup()

        if expdesign is not None:
            # Init or not init ?
            self._expdesign = expdesign
        else:
            self._expdesign = ExpDesign

    def exp_setup(self):
        ''' work in self.exp_tensor '''

        # Global settings (unique argument)
        self._conf = self.exp_tensor.get_conf()

        # makes it contextual.
        self._preprocess_exp()

        # Make lod
        self.lod = self.exp_tensor.make_lod()

        indexs = self._conf['_run_indexs']
        if indexs:
            self.exp_tensor.remake(indexs)
            self._update()

    def _update(self):
        # Seems obsolete, _conf is not writtent in exp_tensor, right ?
        self._conf = self.exp_tensor._conf
        #
        self.lod = self.exp_tensor._lod


    def _preprocess_exp(self):
        #self._check_exp(self.exp_tensor)
        self.exp_tensor.check_bind()
        self.exp_tensor.check_model_typo()
        self.exp_tensor.check_null()


    def io_format_check(self):
        if len(self) > 1 and 'write' in self._conf:
            self.check_format()

        # Clean pymake extra args:
        extra_args = ['_ignore_format_unique', ('_net', False)]
        keys_to_remove = []
        for _key in extra_args:
            if type(_key) is tuple:
                if self._conf.get(_key[0]) is _key[1]:
                    keys_to_remove.append(_key[0])
            elif _key in self._conf:
                    keys_to_remove.append(_key)
        # |
        for key in keys_to_remove:
            self.exp_tensor.remove_all(key)


    def check_format(self):
        ''' Check if all expVector are distinguishable in _format.

            @debug: not valid check accros tensors !!!
            @debug: integration in ExpTensorV2 ?
        '''

        for tensor in self.exp_tensor:
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
                lgg.error('The following settings are not set in _format:')
                print(' '+ '  '.join(hidden_key))
                print('Possible conflicts in experience results outputs.')
                print('Please correct {_format} key to fit the experience settings.')
                print('To force the runs, use:  --ignore-format-unique')
                print('Exiting...')
                exit(2)

        if self._conf.get('_ignore_format_unique') is True:
            _format = '{_name}-{_id}'
            self.exp_tensor.update_all(_format=_format)



    @classmethod
    def _check_exp(cls, tensor):
        ''' check format and rules of exp_tensor. '''
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
        ''' Return the set of values of expVector of that {key}. '''
        return sorted(set(self.exp_tensor.get_all(key, default)))

    def get_nounique_keys(self, *args):
        ''' return list of keys that are non unique in expe_tensor
            except if present in :args:.
        '''
        nk = self.exp_tensor.get_nounique_keys()
        for o in args:
            if o in nk:
                nk.remove(o)

        return nk

    def get_array_loc(self, key1, key2, params):
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

        floc = lambda k1, k2, z:(d1[k1], d2[k2], d3[z])
        array = np.ma.array(np.empty(list(map(len, loc)))*np.nan, mask=True)
        return array, floc


    def __len__(self):
        #return reduce(operator.mul, [len(v) for v in self.exp_tensor.values()], 1)
        return len(self.lod)

    @staticmethod
    def make_forest_path(lod, _type, status='f', full_path=False):
        """ Make a list of path from a spec/dict, the filename are
            oredered need to be align with the get_from_conf_file.

            *args -> make_output_path
        """
        targets = []
        for spec in lod:
            filen = GramExp.make_output_path(spec, _type, status=status)
            if filen:
                s = filen.find(_DATA_PATH)
                pt = 0
                if not full_path and s >= 0:
                    pt = s + len(_DATA_PATH)
                targets.append(filen[pt:])
        return targets

    @classmethod
    def make_output_path(cls, expe, _type=None, status=None):
        """ Make a single output path from a expe/dict
            @status: f finished
            @type: pk, json or inference.
        """
        expe = defaultdict(lambda: None, expe)
        base = expe.get('_data_type', 'pmk-temp')
        hook = expe.get('_refdir', '_default')

        basedir = os.path.join(_DATA_PATH, base, 'results')

        rep = ''
        if '_repeat' in expe and ( expe['_repeat'] is not None and expe['_repeat'] is not False):
            rep = str(expe['_repeat'])
            if rep == '-1':
                rep = ''

        p = os.path.join(hook, rep)

        if not expe['_format']:
            # or give a hash if write is False ?
            lgg.debug('No _format given, please set _format for output_path settings.')
            return None

        t = expe['_format'].format(**cls.get_file_format(expe))

        filen = os.path.join(basedir, p, t)

        ext = ext_status(filen, _type)
        if ext:
            filen = ext

        if status is 'f' and is_empty_file(filen):
            return  None
        else:
            return filen

    @classmethod
    def make_input_path(cls, expe, _type=None, status=None):
        """ Make a single input path from a expe/dict """
        expe = defaultdict(lambda: None, expe)
        filen = None
        base = expe.get('_data_type', 'pmk-temp')

        # Corpus is an historical exception and has its own subfolder.
        c = expe.get('corpus')
        if not c:
            c = ''
            lgg.debug('warning: No Corpus given')
        if c.lower().startswith(('clique', 'graph', 'generator')):
            c = c.replace('generator', 'Graph')
            c = c.replace('graph', 'Graph')
            c = 'generator/' + c

        input_dir = os.path.join(_DATA_PATH, base, 'training', c)

        return input_dir

    @staticmethod
    def get_file_format(expe):
        fmt_expe = expe.copy()
        # iteration over {expe} trow a 'RuntimeError: dictionary changed size during iteration',
        # maybe due to **expe pass in argument ?
        id_str = 'expe' + str(expe['_id_expe'])
        id_name = expe['_name_expe']

        fmt_expe['_id'] = id_str
        fmt_expe['_name'] = id_name
        for k, v in fmt_expe.items():
            if isinstance(v, (list, dict)):
                fmt_expe[k] = id_str

        return fmt_expe



    @staticmethod
    def get_parser(description=None, usage=None):
        import pymake.core.gram as _gram
        parser = _gram.ExpArgumentParser(description=description, epilog=usage,
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('--version', action='version', version='%(prog)s '+str(_version))

        return parser

    @classmethod
    def push_gramarg(cls, parser, gram=None):
        import pymake.core.gram as _gram
        if gram is None:
            gram = _gram._Gram
        else:
            gram = importlib.import_module(gram)
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
                    lgg.debug(str(e)+err_mesg)
                else:
                    lgg.error(str(e)+err_mesg)
                #exit(3)


    @classmethod
    def parseargsexpe(cls, usage=None, args=None, parser=None):
        description = 'Specify an experimentation.'
        if not usage:
            usage = GramExp._examples

        if parser is None:
            parser = GramExp.get_parser(description, usage)
        #s, remaining = parser.parse_known_args(args=args)

        # Push pymake grmarg.
        cls.push_gramarg(parser)
        # third-party
        gramarg = get_global_settings('gramarg')
        if gramarg:
            cls.push_gramarg(parser, gramarg)

        s = parser.parse_args(args=args)

        # Assume None value are non-filled options
        settings = dict((key,value) for key, value in vars(s).items() if value is not None)

        expdesign = GramExp.expVectorLookup(settings)

        return settings, parser, expdesign

    @classmethod
    def zymake(cls, request={}, usage='', firsttime=True, expdesign=None):
        usage ='''\
        ----------------
        Communicate with the data :
        ----------------
         |   zymake update  : update the pymake index
         |   zymake -l [spec(default)|model|script|topo]
         |   zymake show SPEC : show one spec details
         |   zymake run SPEC [--script [fun][*args]] ... : execute tasks (default is fit)
         |   zymake runpara SPEC [--script [fun][*args]] ...: parallelize tasks
         |   zymake hist [-n n_lines] : show command history
         |   zymake cmd SPEC ... : generate command-line
         |   zymake path SPEC Filetype(pk|json|inf) [status] ... : show output_path
        ''' + '\n' + usage

        s, parser, expdesign_lkp = GramExp.parseargsexpe(usage)
        request.update(s)

        ontology = dict(
            _do    = ['cmd', 'show', 'path', 'burn', 'run', 'update', 'init', 'runpara', 'hist'],
            _spec   = list(cls._spec),
            _ftype = ['json', 'pk', 'inf']
        )
        ont_values = sum([w for k, w in ontology.items() if k != '_spec'] , [])

        # Init _do value.
        if not request.get('_do'):
            request['_do'] = []

        # Special Case for CLI.
        if 'script' in request:
            # check if no command is specified, and
            # if 'script" is there, set 'run' command as default.
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
                checksum -= 1
            else:
                for ont, words in ontology.items():
                    # Walktrough the ontology to find arg semantics
                    if v in words:
                        if ont == '_spec':
                            if v in ont_values:
                                lgg.critical('=> Warning: conflict between name of ExpDesign and Pymake commands')
                            do.remove(v)
                            d, expdesign = Spec.load(v, cls._spec[v])
                            expgroup.append((v,d))
                        else:
                            request[ont] = v

                        checksum -= 1
                        break

        # Check erros in th command line
        if checksum != 0:
            if  firsttime == True:
                lgg.warning('Spec not found, re-building Spec indexes...')
                cls.update_index('spec')
                cls._spec = Spec.get_all()
                return cls.zymake(firsttime=False)
            else:
                lgg.error('==> Error : unknown argument: %s\n\nAvailable Exp : %s' % (do, list(cls._spec)))
                exit(10)


        # Setup the exp inputs
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

    def make_commandline(self):
        commands = self.make_commands()
        commands = [' '.join(c) for c in commands]
        return commands

    def make_path(self, ftype=None, status=None, fullpath=None):
        return self.make_forest_path(self.lod, ftype, status, fullpath)


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
                            _arg = ' '.join(_arg)
                        command += [a.option_strings[0]] + [str(_arg)]
                        args_seen.append(a.dest)

            commands.append(command)
        return commands

    def reorder_firstnonvalid(self, _type='pk'):
        for i, e in enumerate(self.lod):
            if not self.make_output_path(e, _type=_type, status='f'):
                self.lod[0], self.lod[i] = self.lod[i], self.lod[0]
                break
        return

    def exptable(self):
        return self.exp_tensor.table()

    def spectable(self):
        return Spec.table()

    def scripttable(self):
        return Script.table()

    def topotable(self):
        return Spec.table_topos(self._spec)

    def modeltable(self, _type='short'):
        return Model.table(_type)

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

        print('-'*30, file=file)
        print('PYMAKE Exp: %d expe' % (len(self) ), file=file)
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
            opts = ['%s [%s]'% (o.name, o.default) for o in m[1] if o]
            opts = ' '.join(opts)
            line = name + ' ' + opts
            lines.append(line)
        return lines

    @staticmethod
    def load(fn, silent=False):
        # Pickle class
        fn = fn + '.pk'
        if not silent:
            lgg.info('Loading frData : %s' % fn)
        with open(fn, 'rb') as _f:
            return pickle.load(_f)

    @staticmethod
    def save(data, fn, silent=False):
        # Pickle class
        fn = fn + '.pk'
        if not silent:
            lgg.info('Saving frData : %s' % fn)
        with open(fn, 'wb') as _f:
            return pickle.dump(data, _f, protocol=pickle.HIGHEST_PROTOCOL)

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


    def expe_init(self, expe, _seed_path='/tmp/pymake.seed'):
        _seed = expe.get('seed')

        if _seed is None:
            seed = np.random.randint(0, 2**32)
        elif type(_seed) is int:
            seed = _seed
        elif _seed is True:
            seed = None
            try:
                np.random.set_state(self.load(_seed_path))
            except FileNotFoundError as e:
                lgg.error("Cannot initialize seed, %s file does not exist." % _seed_path)
                self.save(np.random.get_state(), _seed_path, silent=True)
                raise FileNotFoundError('%s file just created, try again !')

        if seed:
            # if no seed is given, it impossible ti get a seed from numpy
            # https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
            np.random.seed(seed)

        # Init I/O settings
        expe['_output_path'] = self.make_output_path(expe)
        expe['_input_path'] = self.make_input_path(expe)

        self.save(np.random.get_state(), _seed_path, silent=True)
        self._seed = seed
        return seed


    def execute(self):
        ''' Execute Exp Sequentially '''

        if 'script' in self._conf:
            script = self._conf.pop('script')
            self.exp_tensor.remove_all('script')
            self.exp_tensor.remove_all('_do')
        else:
            lgg.error('==> Error : Who need to specify a script. (--script)')
            exit(10)

        try:
            _script, script_args = Script.get(script[0], script[1:])
        except ValueError as e:
            lgg.warning('Script not found, re-building Scripts indexes...')
            self.update_index('script')
            #cls._spec = Spec.get_all()
            try:
                _script, script_args = Script.get(script[0], script[1:])
            except:
                raise
        except IndexError as e:
                lgg.error('Script arguments error : %s -- %s' % (e, script))
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
            self.exp_tensor.update_all(_do=script_args)
        #self.pymake(sandbox=Scripts[script_name])
        self.pymake(sandbox=_script)

    def execute_parallel(self):

        basecmd = sys.argv.copy()
        try:
            Target = subprocess.check_output(['which','pymake']).strip().decode()
            #Target = 'pymake'
        except:
            Target = 'python3 ./zymake.py'
        basecmd = [Target] + basecmd[1:]

        # Create commands indexs
        indexs = self._conf['_run_indexs']
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
            cmd.pop(idx); cmd.pop(idx) # pop --cores int
            cmdlines[i] = ' '.join(cmd)


        if self._conf.get('simulate'):
            self.simulate()

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

        basecmd = sys.argv.copy()
        Target = './zymake.py' # add remote binary in .pymake.cfg
        basecmd = ['python3', Target] + basecmd[1:]
        #basecmd = ['pymake'] + basecmd[1:] # PATH and PYTHONPATH varible missing to be able to execute "pymake"
        cmdlines = None

        # Create commands indexs
        indexs = self._conf['_run_indexs']
        if not indexs:
            indexs = range(len(self))
        else:
            [basecmd.remove(str(i)) for i in indexs]

        # Get chunked indexs
        if nhosts is None:
            # cores by host
            n_cores = int(self._conf.get('_cores', 1))
        else:
            # share run per host
            n_cores = len(indexs) // int(nhosts)

            # Only --net 1, implement/understood from now.
            if int(nhosts) == 1:
                indexs = []
                cmdlines = [' '.join(basecmd)]
            else:
                raise NotImplementedError

        if cmdlines is None:
            indexs = list(map(str, indexs))
            indexs = [indexs[i:i+n_cores] for i in range(0, len(indexs), n_cores)]

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
            cmdlines[i] = re.sub(r'--net\s+[0-9]*', '', cmd).strip()

        NDL = get_global_settings('loginfile')
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
        PWD = get_global_settings('remote_pwd')
        cmd = ['parallel', '-u', '-C', "' '", '--eta', '--progress',
               '--sshloginfile', NDL, '--workdir', PWD, '--env', 'OMP_NUM_THREADS', ':::', '%s'%('\n'.join(cmdlines))]

        if self._conf.get('simulate'):
            self.simulate()

        #stdout = subprocess.check_output(cmd)
        #print(stdout.decode())
        for line in self.subexecute1(cmd):
            print(line, end='')

    @staticmethod
    def subexecute1(cmd):
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
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
        # get the expe
        # get the script
        # get the figure
        code = ''
        nb['cells'] = [nbf.new_markdown_cell(text),
                       nbf.new_code_cell(code) ]
        return

    def init_folders(self):
        from string import Template
        from os.path import join
        from pymake.util.utils import set_global_settings

        pwd = os.getenv('PWD')
        if os.path.isfile(join(pwd, 'pymake.cfg')):
            print('pymake.cfg file already exists.')
            exit(11)


        cwd = os.path.dirname(__file__)
        folders = ['spec', 'script', 'model']
        open(join(pwd, '__init__.py'), 'a').close()
        spec = {'projectname':os.path.basename(pwd)}
        settings = {}
        for d in folders:
            os.makedirs(d, exist_ok=True)
            with open(join(cwd, '%s.template'%(d))) as _f:
                template = Template(_f.read())

            open(join(pwd, d,  '__init__.py'), 'a').close()
            with open(join(pwd, d,  'template_%s.py'%(d)), 'a') as _f:
                _f.write(template.substitute(spec))

            if d in ['spec', 'script']:
                settings.update({'default_%s'%(d):'.'.join((spec['projectname'], d))})
            else: # share model
                settings.update({'contrib_%s'%(d):'.'.join((spec['projectname'], d))})

        settings.update(project_data='data')
        set_global_settings(settings)
        print('update project: {projectname}'.format(**spec))
        return self.update_index()

    def pushcmd2hist(self):
        from pymake.util.utils import tail
        bdir = self.data_path
        fn = os.path.join(bdir, '.pymake_hist')
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
        n_lines = int(self._conf.get('N', 42))
        bdir = self.data_path
        fn = os.path.join(bdir, '.pymake_hist')
        if not os.path.isfile(fn):
            lgg.error('hist file does not exist.')
            return

        _tail = tail(fn, n_lines)

        print('\n'.join(_tail))


    @staticmethod
    def update_index(*index_name):
        from pymake.index.indexmanager import IndexManager as IX
        os.chdir(os.getenv('PWD'))

        if len(index_name) == 0:
            IX.build_indexes()
        else:
            for i in index_name:
                IX.build_indexes(i)


    def pymake(self, sandbox=ExpeFormat):
        ''' Walk Trough experiments. '''

        if 'do_list' in self._conf:
            print('Available methods for %s: ' % (sandbox))
            print(*self.functable(sandbox) , sep='\n')
            exit()

        sandbox._preprocess_(self)
        self.io_format_check()

        if self._conf.get('simulate'):
            self.simulate()

        n_errors = 0
        for id_expe, _expe in enumerate(self.lod):
            expe = ExpSpace(**_expe)

            pt = dict((key, value.index(expe[key])) for key, value in self.exp_tensor.get_gt().items()
                      if (isinstance(expe.get(key), (basestring, int, float)) and key not in self._reserved_keywords))
            pt['expe'] = id_expe

            # Init Expe
            self.expe_init(expe)
            try:
                expbox = sandbox(pt, expe, self)
            except FileNotFoundError as e:
                lgg.error('ignoring %s'%e)
                continue
            except Exception as e:
                print(('Error during '+colored('%s', 'red')+' Initialization.') % (str(sandbox)))
                traceback.print_exc()
                exit(2)

            # Expe Preprocess
            expbox._preprocess()

            # Setup handler
            if '_do' in expe and len(expe._do) > 0:
                # ExpFormat task ? decorator and autoamtic argument passing ....
                do = expe._do
                pmk = getattr(expbox, do[0])
            else:
                do = ['__call__']
                pmk = expbox

            # Launch handler
            args = do[1:]
            try:
                pmk(*args)
            except KeyboardInterrupt:
                # it's hard to detach matplotlib...
                break
            except Exception as e:
                n_errors += 1
                lgg.critical(('Error during '+colored('%s', 'red')+' Expe no %d.') % (do, id_expe))
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
            expbox._postprocess()

        if n_errors > 0:
            lgg.warning("There was %d errors,  logged in `%s'" % (n_errors, self._pmk_error_file))
            with open(self._pmk_error_file, 'a') as _f:
                _f.write(100*'='+'\n')

        return sandbox._postprocess_(self)

    @property
    def data_path(self):
        return get_global_settings('project_data')

