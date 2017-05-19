# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os
import operator
import fnmatch
import pickle
import inspect, traceback, importlib
from collections import defaultdict
from itertools import product
from functools import reduce, wraps
from copy import deepcopy
import numpy as np

from argparse import RawDescriptionHelpFormatter

import pymake.expe.gram as gram
from pymake import basestring, ExpTensor, ExpSpace, ExpeFormat, Model, Corpus, Script, ExpVector
from pymake.plot import colored

import pymake.frontend.frontend_io as mloader

from pymake.frontend.frontend_io import _DATA_PATH, ext_status, is_empty_file

import logging
lgg = logging.getLogger('root')

''' Grammar Expe '''
_version = 0.1


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
            *  [a.b]  --- a shoudld occur only with b,
            * [a.b.c] --- for a, key b take only c.
            Warning : it does only check the last words when parameter are
                      separated by a dot (.) as for model module for example.

        * if an attribute  take the value "_null",
          then it will be removed from the Exp. This can be usefull when piping
          ouptuf of pymake to some script to parallelize.

        ### I/O Concerns
        An experiment typically write three kind of files :
        Corpus are load/saved using Pickle format in:
        * bdir/corpus_name.pk
        Models are load/saved using pickle/json/cvs in :
        * bdir/debug/rept/model_name_parameters.pk   <--> ModelManager
        * bdir/debug/rept/model_name_parameters.json <--> DataBase
        * bdir/debug/rept/model_name_parameters.inf  <--> ModelBase

    '''
    _examples = '''Examples:
        >> Text fitting
        fit.py -k 6 -m ldafullbaye -p
        >> Network fitting :
        fit.py -m immsb -c alternate -n 100 -i 10'''

    # Reserved by GramExp
    _reserved_keywords = ['spec', # for nested specification
                         ]

    _exp_default = {
        #'host'      : 'localhost',
        'verbose'   : logging.INFO,
        'load_data' : True,
        'save_data' : False,
        'write'     : False,
    }

    def __init__(self, conf={}, usage=None, parser=None, parseargs=True):
        self._spec = mloader.SpecLoader.default_spec()
        if parseargs is True:
            kwargs, self.argparser = self.parseargsexpe(usage)
            conf.update(kwargs)
        if parser is not None:
            # merge parser an parsargs ?
            self.argparser = parser

        [conf.update({k:v}) for k,v in self._exp_default.items() if k not in conf]
        conf = deepcopy(conf)

        # in expt_setup...
        self._default_spec = conf.get('spec', {})

        # @logger One logger by Expe !
        level = self.setup_logger(fmt='%(message)s', level=conf.get('verbose'))
        conf.update(verbose=level)

        # Make main data structure
        self.exp_tensor = ExpTensor.from_expe(conf)
        self.checkExp(self.exp_tensor)
        self.exp_setup()

    def _preprocess_exp(self):
        # @improvment: do filter from spec
        exp = self.exp_tensor

        # Rules Filter
        if '_bind' in exp:
            self._bind = exp.pop('_bind')
            if not isinstance(self._bind, list):
                self._bind = [self._bind]
        else:
            self._bind = getattr(self, '_bind', [])

        # Assume default module is pymake
        models = exp.get('model', [])
        for i, m in enumerate(models):
            if not '.' in m:
                models[i] = 'pmk.%s'%(m)

        # Filter _null
        for k in exp.copy():
            if '_null' in exp.get(k, []):
                exp.pop(k)
                _null = getattr(self, '_null',[])
                _null.append(k)


    @classmethod
    def checkExp(cls, exp):
        ''' check format and rules of exp_tensor '''

        # check reserved keyword are absent from exp
        for m in cls._reserved_keywords:
            if m in exp:
                raise ValueError('%m is a reserved keyword of gramExp.')

        # Format
        assert(isinstance(exp, ExpTensor))
        for k, v in exp.items():
            if not issubclass(type(v), (list, tuple, set)):
                raise ValueError('error, exp value should be iterable: %s' % k, v)


    def make_lod(self, exp):
        ''' Make a list of Expe from tensor, with filtering '''

        lod = self.make_forest_conf(exp)

        # POSTFILTERING
        # Bind Rules
        itoremove = []
        for i, d in enumerate(lod):
            for rule in self._bind:
                _bind = rule.split('.')
                values = list(d.values())

                # only last dot separator
                for j, e in enumerate(values):
                    if type(e) is str:
                        values[j] = e.split('.')[-1]

                if len(_bind) == 2:
                    # remove all occurence if this bind don't occur
                    # simltaneous in each expe.
                    a, b = _bind
                    if a in values and not b in values:
                        itoremove.append(i)
                elif len(_bind) == 3:
                    # remove occurence of this specific key:value if
                    # it does not comply with this bind.
                    a, b, c = _bind
                    # Get the type of this key:value.
                    _type = type(d[b])
                    if a in values and _type(c) != d[b]:
                        itoremove.append(i)

        lod = [d for i,d in enumerate(lod) if i not in itoremove]
        return lod

    def update(self, **kwargs):
        self._conf.update(kwargs)
        self.exp_tensor.update_from_dict(kwargs)

        for d in self.lod:
            d.update(kwargs)

    # @Debug self.lod is left untouched...
    def remove(self, k):
        if k in self._conf:
            self._conf.pop(k)
        if k in self.exp_tensor:
            self.exp_tensor.pop(k)
        for d in self.lod:
            if k in d:
                d.pop(k)

    def exp_setup(self, exp=None):
        if exp is not None:
            self.exp_tensor = exp

        # makes it contextual.
        self._preprocess_exp()

        # Make lod
        self.lod = self.make_lod(self.exp_tensor)

        # Global settings (unique argument)
        self._conf = {k:v[0] for k,v in self.exp_tensor.items() if len(v) == 1}

    def getConfig(self):
        # get global conf...
        raise NotImplementedError

    def getCorpuses(self):
        return self.exp_tensor.get('corpus', [])

    def getModels(self):
        return self.exp_tensor.get('model',[])

    def get(self, key, default=None):
        return self.exp_tensor.get(key, default)

    def __len__(self):
        #return reduce(operator.mul, [len(v) for v in self.exp_tensor.values()], 1)
        return len(self.lod)

    @staticmethod
    def make_forest_conf(dol_spec):
        """ Make a list of config/dict.
            Convert a dict of list to a list of dict.
        """
        if len(dol_spec) == 0:
            return []

        len_l = [len(l) for l in dol_spec.values()]
        keys = sorted(dol_spec)
        lod = [dict(zip(keys, prod)) for prod in product(*(dol_spec[key] for key in keys))]

        return lod

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

    @staticmethod
    def make_output_path(expe, _type=None, status=None):
        """ Make a single output path from a expe/dict
            @status: f finished
            @type: pk, json or inference.
        """
        expe = defaultdict(lambda: None, expe)
        filen = None
        lgg.debug('heere, get data_type from loader corpus info !')
        base = expe.get('data_type', 'networks')
        hook = expe.get('refdir', '')
        c = expe.get('corpus')
        if not c:
            return None
        if c.lower().startswith(('clique', 'graph', 'generator')):
            c = c.replace('generator', 'Graph')
            c = c.replace('graph', 'Graph')
            c = 'generator/' + c

        basedir = os.path.join(os.path.dirname(__file__), _DATA_PATH, base, c)

        if 'repeat' in expe and ( expe['repeat'] is not None and expe['repeat'] is not False):
            p = os.path.join(hook, str(expe['repeat']))
        else:
            p = os.path.join(hook)

        if not expe['_format']:
            _format = '{model}_{K}_{hyper}_{homo}_{N}'
        else:
            _format = expe['_format']
        t = _format.format(**expe)

        filen = os.path.join(basedir, p, t)

        ext = ext_status(filen, _type)
        if ext:
            filen = ext
        else:
            filen = (basedir, filen)

        if status is 'f' and is_empty_file(filen):
            return  None
        else:
            return filen


    @staticmethod
    def get_parser(description=None, usage=None):

        parser = gram.ExpArgumentParser(description=description, epilog=usage,
                                        formatter_class=RawDescriptionHelpFormatter)
        g = gram._Gram
        grammar = []
        args = []
        for e in g:
            if not isinstance(e, dict):
                args.append(e)
                continue
            grammar.append((args, e))
            args = []

        parser.add_argument('--version', action='version', version='%(prog)s '+str(_version))
        [parser.add_argument(*r[0], **r[1]) for r in grammar]

        return parser


    @staticmethod
    def parseargsexpe(usage=None, args=None, parser=None):
        description = 'Specify an experimentation.'
        if not usage:
            usage = GramExp._examples

        if parser is None:
            parser = GramExp.get_parser(description, usage)
        #s, remaining = parser.parse_known_args(args=args)

        s = parser.parse_args(args=args)

        settings = dict((key,value) for key, value in vars(s).items() if value)
        GramExp.expVectorLookup(settings)
        return settings, parser

    @classmethod
    def zymake(cls, request={}, usage=''):
        usage ='''\
        ----------------
        Communicate with the data :
        ----------------
         |   zymake -l [expe(default)|atom|script] : (default) show available spec
         |   zymake update  : update the pymake index
         |   zymake show SPEC : show one spec details
         |   zymake cmd SPEC [fun][*args] : generate command-line
         |   zymake exec SPEC [fun][*args][--script ...] : execute tasks (default is fit)
         |   zymake burn SPEC [fun][*args][--script ...] : parallelize tasks
         |   zymake path SPEC Filetype(pk|json|inf) [status]
        ''' + '\n' + usage

        s, parser = GramExp.parseargsexpe(usage)
        request.update(s)

        _spec = mloader.SpecLoader.default_spec()
        ontology = dict(
            _do    = ['cmd', 'show', 'path', 'burn', 'exec', 'update'],
            spec   = _spec._specs(),
            _ftype = ['json', 'pk', 'inf']
        )
        ont_values = sum([w for k, w in ontology.items() if k != 'spec'] , [])

        # Init _do value
        if not request.get('_do'):
            request['_do'] = []
            if not request.get('do_list'):
                request['do_list'] = None

        # Hack for script/exec
        if request.get('do_list'):
            if 'script' in request:
                request['_do'] = ['exec']

        do = request['_do']
        checksum = len(do)
        # No more Complex !
        for i, v in enumerate(do):
            for ont, words in ontology.items():
                # Walktrough the ontology to find arg meaning
                if v in words:
                    if ont == 'spec':
                        if v in ont_values:
                            lgg.error('warning: conflict between name of ExpDesign and GramExp ontology keywords ')
                        ont = 'spec'
                        do.remove(v)
                        v = _spec[v]
                    request[ont] = v
                    checksum -= 1
                    break

        if request.get('spec') and len(do) == 0:
            request['_do'] = 'show'

        #if '-status' in clargs.grouped:
        #    # debug status of filr (path)
        #    request['_status'] = clargs.grouped['-status'].get(0)

        if checksum != 0:
            raise ValueError('unknow argument: %s\n\nAvailable SPEC : %s' % (do, sorted(_spec.keys())))
        return cls(request, usage=usage, parser=parser, parseargs=False)

    @classmethod
    def generate(cls, request={},  usage=''):
        usage = '''\
        ----------------
        Execute scripts :
        ----------------
         |   generate [method][fun][*args] [EXP]
         |
         |  -g: generative model -- evidence
         |  -p: predicted data -- model fitted
         |  --hyper-name *
         |
         ''' +'\n'+usage

        parser = GramExp.get_parser(usage=usage)
        parser.add_argument('-g', '--generative', dest='_mode', action='store_const', const='generative')
        parser.add_argument('-p', '--predictive', dest='_mode', action='store_const', const='predictive')

        s, parser = GramExp.parseargsexpe(parser=parser)

        request.update(s)

        do = request.get('_do') or ['list']
        if not isinstance(do, list):
            # Obsolete ?
            do = [do]
            request['_do'] = do

        # Update the spec is a new one is argified
        _spec = mloader.SpecLoader.default_spec()
        for a in do:
            if a in _spec.keys() and issubclass(type(_spec[a]), ExpTensor):
                request['spec'] = _spec[a]
                do.remove(a)

        return cls(request, usage=usage, parser=parser, parseargs=False)

    @staticmethod
    def expVectorLookup(request):
        ''' set exp from spec if presents '''

        # get list of scripts/*
        # get list of method
        _spec = mloader.SpecLoader.default_spec()
        ontology = dict(
            # Allowed multiple flags keywords
            check_spec = {'corpus':Corpus, 'model':Model},
            spec = _spec.keys()
        )

        # Handle spec and multiple flags arg value
        # Too comples, better Grammat/Ast ?
        for k in ontology['check_spec']:
            sub_request = ExpVector()
            for v in request.get(k, []):
                if v in ontology['spec'] and k in ontology['check_spec']:
                    # Flag value is in specification
                    if issubclass(type(_spec[v]), ontology['check_spec'][k]):
                        sub_request.extend( _spec[v] )
                    else:
                        raise ValueError('%s not in Spec' % v)
                else:
                    # Multiple Flags
                    sub_request.extend([v])
            if sub_request:
                request[k] = sub_request

    @classmethod
    def exp_tabulate(cls, conf={}, usage=''):

        gargs = clargs.grouped['_'].all
        for arg in gargs:
            try:
                conf['K'] = int(arg)
            except:
                conf['model'] = arg
        return cls(conf, usage=usage)

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

    def make_commandline(self):
        commands = self.make_commands()
        commands = [' '.join(c) for c in commands]
        return commands

    def make_path(self, ftype=None, status=None, fullpath=None):
        return self.make_forest_path(self.lod, ftype, status, fullpath)

    def reorder_lastvalid(self, _type='pk'):
        for i, e in enumerate(self.lod):
            if self.make_output_path(e, _type=_type, status='f'):
                self.lod[-1], self.lod[i] = self.lod[i], self.lod[-1]
                break
        return

    def expname(self):
        return self.exp_tensor.name()

    def exptable(self, extra=[]):
        if self._bind:
            extra += [('_bind', self._bind)]
        return self.exp_tensor.table(extra)

    def spectable(self):
        return self._spec._table()

    def atomtable(self, _type='short'):
        return self._spec._table_atoms(_type=_type)

    def scripttable(self):
        return Script.table()

    def getSpec(self):
        return self._spec

    @staticmethod
    def Spec():
        return mloader.SpecLoader.default_spec()

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
        exit(2)

    def simulate(self, halt=True, file=sys.stdout):
        print('-'*30, file=file)
        print('PYMAKE Request %s : %d expe' % (self.exp_tensor.name(), len(self) ), file=file)
        print(self.exptable(), file=file)
        if halt:
            exit(2)
        else:
            return

    @staticmethod
    def setup_logger(fmt='%(message)s', level=logging.INFO, name='root'):
        #formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        if level == 1:
            level = logging.DEBUG
        elif level == 2:
            print ('what level of verbosity heeere ?')
            exit(2)
        elif level == -1:
            level = logging.ERROR
        elif level is None:
            level = logging.INFO
        else:
            level = level

        # Get logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Format handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=fmt))
        logger.addHandler(handler)

        # Prevent logging propagation of handler,
        # who results in logging things multiple times
        logger.propagate = False

        return level

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


    def expe_init(self, expe):
        if not 'seed' in expe:
            return

        if expe.seed is True:
            try:
                np.random.set_state(self.load('/tmp/pymake.seed', silent=True))
            except FileNotFoundError as e:
                print('error: %s, Try a seed (--seed VALUE).' % (e))
                exit()
        else:
            np.random.seed(expe.seed)

        self._seed = np.random.get_state()
        self.save(self._seed, '/tmp/pymake.seed', silent=True)


    def execute(self):
        ''' Execute Exp Sequentially '''
        if 'script' in self._conf:
            script = self._conf.pop('script')
            self.remove('script')
            self.remove('_do')
        else:
            raise ValueError('Who need to specify a script. (--script)')

        _script, script_args = Script.get(script[0], script[1:])

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
            self.update(_do=script_args)
        #self.pymake(sandbox=Scripts[script_name])
        self.pymake(sandbox=_script)

    def notebook(self):
        from nbformat import v4 as nbf
        nb = nbf.new_notebook()
        text = ''
        code = ''
        nb['cells'] = [nbf.new_markdown_cell(text),
                       nbf.new_code_cell(code) ]
        return

    def update_index(self):
        from pymake.index.indexmanager import IndexManager as IX
        IX.build_indexes()

    def pymake(self, sandbox=ExpeFormat):
        ''' Walk Trough experiments. '''

        if 'do_list' in self.exp_tensor:
            print('Available methods for %s: ' % (sandbox))
            print(*self.functable(sandbox) , sep='\n')
            exit(2)

        sandbox._preprocess(self)
        if self._conf.get('simulate'):
            self.simulate()

        for id_expe, expe in enumerate(self.lod):
            _expe = ExpSpace(**expe)

            pt = dict((key, value.index(_expe[key])) for key, value in self.exp_tensor.items() if isinstance(_expe[key], (basestring, int, float)))
            pt['expe'] = id_expe

            # Init Expe
            self.expe_init(_expe)
            try:
                expbox = sandbox(pt, _expe, self)
            except FileNotFoundError as e:
                lgg.error('ignoring %s'%e)
                continue
            except Exception as e:
                print(('Error during '+colored('%s', 'red')+' Initialization.') % (str(sandbox)))
                traceback.print_exc()
                exit()

            # Setup handler
            if hasattr(_expe, '_do') and len(_expe._do) > 0:
                # ExpFormat task ? decorator and autoamtic argument passing ....
                do = _expe._do
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
                print(('Error during '+colored('%s', 'red')+' Expe.') % (do))
                traceback.print_exc()
                exit()

        return sandbox._postprocess(self)


