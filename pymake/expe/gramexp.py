# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from collections import OrderedDict
from functools import reduce, wraps
import operator
import inspect, traceback
from copy import deepcopy

from argparse import RawDescriptionHelpFormatter

from .gram import _Gram, ExpArgumentParser
from pymake import basestring, ExpTensor, Expe, ExpSpace, ExpeFormat, Model, Corpus, ExpVector
from pymake.frontend.frontend_io import make_forest_conf, make_forest_path
from pymake.expe.spec import _spec
from pymake.plot import colored

import logging
lgg = logging.getLogger('root')

''' Grammar Expe '''

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

        Expe* can contains **special** keywords and value :
            * _bind rules ->  [a.b or a.b.c] --- a shoudld occur only with b,
                        or for a, key b take only c.
                List of rules of constraintson the design.
            * if an attribute  take the value "_null",
              then it will be removed from the Exp. This can be usefull when piping
              ouptuf of pymake to some script to parallelize.
    '''
    _examples = '''Examples:
        >> Text fitting
        fit.py -k 6 -m ldafullbaye -p
        >> Network fitting :
        fit.py -m immsb -c alternate -n 100 -i 10'''

    # Reserved by GramExp
    _reserved_keywords = ['spec', # for nested specification
                         'exp' # to track expe walking # Unused
                         ]

    # Avoid conflict between tasks
    _forbiden_keywords = {'fit' : ['gen_size', 'epoch', '_do', '_mode', '_type'],
                          'generate' : ['iterations']}

    _exp_default = {
        'host'      : 'localhost',
        'verbose'   : logging.INFO,
        'load_data' : True,
        'save_data' : False,
        'write'     : False,
    }

    def __init__(self, conf={}, usage=None, parser=None, parseargs=True):
        if parseargs is True:
            kwargs = self.parseargsexpe(usage)
            conf.update(kwargs)
        if parser is not None:
            self.argparser = parser
        else:
            self.argparser = GramExp.get_parser()

        [conf.update({k:v}) for k,v in self._exp_default.items() if k not in conf]
        conf = deepcopy(conf)
        if 'spec' in conf:
            self.exp_tensor = self.exp2tensor(conf.pop('spec'))
            self.exp_tensor.update_from_dict(conf)
        elif isinstance(conf, ExpTensor):
            self.exp_tensor = self.exp2tensor(conf)
        elif isinstance(conf, (dict, Expe)):
            # Assume Single Expe (type dict or Expe)
            if type(conf) is dict:
                # @zymake ?
                lgg.debug('warning : type dict for expe settings may be deprecated.')
            #self.exp_tensor = self.dict2tensor(conf)
            self.exp_tensor = self.exp2tensor(conf)
        else:
            raise ValueError('exp/conf not understood: %s' % type(conf))

        self.checkConf(conf)
        self.checkExp(self.exp_tensor)

        # Make Rules
        if '_bind' in self.exp_tensor:
            self._bind = self.exp_tensor.pop('_bind')
            if not isinstance(self._bind, list):
                self._bind = [self._bind]
        else:
            self._bind = []

        # Make lod
        self.lod = self.lodify(self.exp_tensor)

        # global conf
        self.expe = conf

        # @logger One logger by Expe !
        self.setup_logger(fmt='%(message)s', level=self.expe.get('verbose'))

        if self.expe.get('simulate'):
            self.simulate()

    @classmethod
    def checkConf(cls, settings):
        for m in cls._reserved_keywords:
            if m in settings:
                raise ValueError('%m is a reserved keyword of gramExp.')
        # @todo : verify  forbidden keywords

    @staticmethod
    def checkExp(exp):
        assert(isinstance(exp, ExpTensor))
        for k, v in exp.items():
            if not issubclass(type(v), (list, tuple, set)):
                raise ValueError('error, exp value should be iterable: %s' % k, v)

    def lodify(self, exp):
        ''' Make a list of Expe from tensor, with filtering '''

        # PREFILTERING
        for k in exp.copy():
            if '_null' in exp.get(k, []):
                exp.pop(k)

        lod = make_forest_conf(exp)

        # POSTFILTERING
        # Bind Rules
        itoremove = []
        for i, d in enumerate(lod):
            for rule in self._bind:
                _bind = rule.split('.')
                values = list(d.values())
                if len(_bind) == 2:
                    # remove all occurence of this
                    a, b = _bind
                    if a in values and not b in values:
                        itoremove.append(i)
                elif len(_bind) == 3:
                    # remove occurence of this specific key:value
                    # Get the type of this key:value
                    a, b, c = _bind
                    _type = type(d[b])
                    if a in values and _type(c) != d[b]:
                        itoremove.append(i)

        return [d for i,d in enumerate(lod) if i not in itoremove]

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
    # deprecated
    def dict2tensor(conf):
        ''' Return the tensor who is a Orderedict of iterable.
            Assume unique expe. Every value will be listified.
        '''
        tensor = ExpTensor([(k, [v]) for k, v in conf.items()])
        return tensor

    @staticmethod
    def exp2tensor(conf):
        ''' Return the tensor who is an Orderedict of iterable.
            Assume conf is an exp. Non list value will be listified.
        '''
        if issubclass(type(conf), Corpus):
            tensor = ExpTensor(corpus=conf)
        elif issubclass(type(conf), Model):
            tensor = ExpTensor(model=conf)
        elif not isinstance(conf, ExpTensor):
            tensor = ExpTensor(conf)
        else:
            tensor = conf

        for k, v in tensor.items():
            if not issubclass(type(v), (list, set, tuple)):
                tensor[k] = [v]
        return tensor

    @staticmethod
    def get_parser(description=None, usage=None):

        parser = ExpArgumentParser(description=description, epilog=usage,
                                   formatter_class=RawDescriptionHelpFormatter)
        g = _Gram
        grammar = []
        args = []
        for e in g:
            if not isinstance(e, dict):
                args.append(e)
                continue
            grammar.append((args, e))
            args = []

        parser.add_argument('--version', action='version', version='%(prog)s 0.1')
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
        return settings

    @classmethod
    def zymake(cls, request={}, usage=''):
        usage ='''\
        ----------------
        Communicate with the data :
        ----------------
         |   zymake -l         : (default) show available spec
         |   zymake show SPEC  : show one spec details
         |   zymake cmd SPEC [fun][*args]   : generate command-line
         |   zymake burn SPEC [fun][*args][--script ...] : parallelize tasks
         |   zymake path SPEC Filetype(pk|json|inf) [status]
        ''' + '\n' + usage

        s = GramExp.parseargsexpe(usage)
        request.update(s)

        ontology = dict(
            _do    = ['cmd', 'show', 'path', 'burn', 'list'],
            spec   = _spec.keys(),
            _ftype = ['json', 'pk', 'inf']
        )
        ont_values = sum([w for k, w in ontology.items() if k != 'spec'] , [])

        do = request.get('_do') or ['list']
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
        if request.get('do_list'):
            request['_do'] = 'list'

        if checksum != 0:
            raise ValueError('unknow argument: %s\n\nAvailable SPEC : %s' % (do, _spec.keys()))
        return cls(request, usage=usage, parseargs=False)

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

        s = GramExp.parseargsexpe(parser=parser)
        request.update(s)

        do = request.get('_do') or ['_list']
        if not isinstance(do, list):
            do = [do]
            request['_do'] = do

        # Update the spec is a new one is argified
        for a in do:
            if a in _spec.keys() and issubclass(type(_spec[a]), ExpTensor):
                request['spec'] = _spec[a]
                do.remove(a)

        return cls(request, usage=usage, parseargs=False)

    @staticmethod
    def expVectorLookup(request):
        ''' set exp from spec if present '''

        # get list of scripts/*
        # get list of method
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
                    if isinstance(e[a.dest], (list, dict, tuple, set)):
                        # Assume extra args, not relevant for expe commands
                        continue
                    if a.nargs == 0:
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
                        command += [a.option_strings[0]] + [str(e[a.dest])]
                        args_seen.append(a.dest)

            commands.append(command)
        return commands

    def make_commandline(self):
        commands = self.make_commands()
        commands = [' '.join(c) for c in commands]
        return commands

    def make_path(self, ftype, status=None, fullpath=None):
        return make_forest_path(self.lod, ftype, status, fullpath)


    def expname(self):
        return self.exp_tensor.name()

    def exptable(self, extra=[]):
        if self._bind:
            extra += [('_bind', self._bind)]
        return self.exp_tensor.table(extra)


    def simulate_short(self):
        ''' Simulation Output '''
        print('''
              Nb of experiments : %s
              Corpuses : %s
              Models : %s
              ''' % (len(self), self.getCorpuses(), self.getModels()))
        exit(2)

    def simulate(self, halt=True, file=sys.stdout):
        print('-'*30)
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

        return logger

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
        lines = []
        for m in  GramExp.tb_expeformat(obj):
            name = m[0]
            opts = ['%s [%s]'% (o.name, o.default) for o in m[1] if o]
            opts = ' '.join(opts)
            line = name + ' ' + opts
            lines.append(line)
        return lines

    # @do parralize
    def pymake(self, sandbox=ExpeFormat):
        ''' Walk Trough experiments.  '''

        if 'do_list' in self.expe:
            print('Available methods for %s: ' % (sandbox))
            print(*self.functable(sandbox) , sep='\n')
            exit(2)

        sandbox.preprocess(self)

        for id_expe, expe in enumerate(self.lod):
            pt = dict((key, value.index(expe[key])) for key, value in self.exp_tensor.items() if isinstance(expe[key], (basestring, int, float)))
            pt['expe'] = id_expe
            _expe = ExpSpace(**expe)

            # Init Expe
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

        return sandbox.postprocess(self)


