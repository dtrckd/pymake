# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from functools import reduce, wraps
import operator
import inspect, traceback
from copy import deepcopy

import args as clargs
import argparse

from pymake import basestring, ExpTensor, Expe, ExpeFormat
from pymake.frontend.frontend_io import make_forest_conf, make_forest_path
from pymake.expe.spec import _spec
from pymake.plot import colored

import logging
lgg = logging.getLogger('root')

''' Grammar Expe '''

class ExpArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(ExpArgumentParser, self).__init__(**kwargs)

    def error(self, message):
        self.print_usage()
        print('error', message)
        #self.print_help()
        #print()
        #print_available_ model, datasets ?
        sys.exit(2)

class VerboseAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        # print 'values: {v!r}'.format(v=values)
        if values==None:
            values='1'
        try:
            values=int(values)
        except ValueError:
            values=values.count('v')+1
        setattr(args, self.dest, values)

class SmartFormatter(argparse.HelpFormatter):
    # Unused -- see RawDescriptionHelpFormatter
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def check_positive_integer(value):
    try:
        ivalue = int(value)
    except:
        raise argparse.ArgumentTypeError("%s is invalid argument value, need integer." % value)

    if ivalue < 0:
        return ''
    else:
        return ivalue

# * wraps cannot handle the decorator chain :(, why ?
class askseed(object):
    ''' Load previous random seed '''
    def __init__(self, func, help=False):
        self.func = func
    def __call__(self, *args, **kwargs):

        response = self.func(*args, **kwargs)

        if clargs.flags.contains('--seed'):
            response['seed'] = True
        return response


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

        Expe* can contains keywords :
            * _bind ->  [a.b , ...] --- a shoudld occur with b only.
                List of rules of constraintson the design.
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
    _forbiden_keywords = dict(fit = ['gen_size', 'epoch', '_do', '_mode', '_type'],
                              generate = ['iterations'])

    def __init__(self, conf={}, usage=None, parseargs=True):

        if 'spec' in conf:
            self.exp_tensor = self.exp2tensor(conf.pop('spec'))
            self.exp_tensor.update_from_dict(conf)
        elif isinstance(conf, ExpTensor):
            self.exp_tensor = self.exp2tensor(conf)
        elif isinstance(conf, (dict, Expe)):
            if type(conf) is dict:
                # @zymake ?
                lgg.debug('warning : type dict for expe settings may be deprecated.')
            self.exp_tensor = self.dict2tensor(conf)
        else:
            raise ValueError('exp/conf not understood: %s' % type(conf))

        conf = deepcopy(conf)
        if parseargs:
            # Populate conf from command line
            kwargs = self.parseargsexpe(usage)
            self.exp_tensor.update_from_dict(kwargs)
            conf.update(kwargs)

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
        self.setup_logger(fmt='%(message)s', level=self.expe['verbose'])

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

        lod = make_forest_conf(exp)

        # Bind Rules
        itoremove = []
        for i, d in enumerate(lod):
            for rule in self._bind:
                a, b = rule.split('.')
                values = list(d.values())
                if a in values and not b in values:
                    itoremove.append(i)

        return [d for i,d in enumerate(lod) if i not in itoremove]

    def getConfig(self):
        # get global conf...
        raise NotImplementedError

    def getCorpuses(self):
        return self.exp_tensor['corpus']

    def getModels(self):
        return self.exp_tensor['model']

    def __len__(self):
        return reduce(operator.mul,
                      [len(v) for v in self.exp_tensor.values()], 1)

    @staticmethod
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
        if not isinstance(conf, ExpTensor):
            tensor = ExpTensor(conf)
        else:
            tensor = conf
        for k, v in conf.items():
            if not issubclass(type(v), (list, set, tuple)):
                tensor[k] = [v]
        return tensor

    # @askhelp
    # @askseed
    # @overload argparse
    # @Debug use _do and _type from argparse
    @classmethod
    def zymake(cls, conf={}, usage=None):
        USAGE = usage or '''\
-----------
Communicate with the data :
-----------
 |   zymake -l         : (default) show available spec
 |   zymake show SPEC  : show one spec details
 |   zymake cmd SPEC   : generate command-line
 |   zymake burn SPEC [--script ...] : parallelize tasks
 |   zymake path SPEC Filetype(pk|json|inf) [status]
'''

        # Default request
        req = dict(
            _do = 'list',
            spec = _spec['RAGNRK'],
            _ftype = 'pk',
            _status = None )

        ontologies = dict(
            _do = ['cmd', 'show', 'path', 'burn'],
            _out_type = ('cmd', 'path', 'show'),
            spec = _spec.keys(),
            _ftype = ('json', 'pk', 'inf') )

        ### Making ontologie based argument attribution
        ### Used the grouped argument whitout flags ['_']
        gargs = clargs.grouped['_'].all
        checksum = len(gargs)
        for i, v in enumerate(gargs):
            for ont, words in ontologies.items():
                # Walktrough the ontology to find arg meaning
                if v in words:
                    if ont == 'spec' and i > 0:
                        v = _spec[v]
                    req[ont] = v
                    checksum -= 1
                    break

        if '-status' in clargs.grouped:
            # debug status of filr (path)
            req['_status'] = clargs.grouped['-status'].get(0)
        if '-l' in clargs.grouped or '--list' in clargs.grouped:
            # debug show spec
            req['_do'] = 'list'

        if checksum != 0:
            raise ValueError('unknow argument: %s\n\nAvailable SPEC : %s' % (gargs, _spec.keys()))
        conf.update(req)
        return cls(conf, usage=USAGE)

    # @askhelp
    # @askseed
    # @overload argparse
    @classmethod
    def generate(cls, conf={},  USAGE=''):
        write_keys = ('-w', 'write', '--write')
        # Write plot
        for write_key in write_keys:
            if write_key in clargs.all:
                conf['save_plot'] = True
        gargs = clargs.grouped.pop('_')

        ### map config dict to argument
        for key in clargs.grouped:
            if '-n' in key:
                conf['gen_size'] = clargs.grouped['-n'].get(0)
            elif '--alpha' in key:
                try:
                    conf['alpha'] = float(clargs.grouped['--alpha'].get(0))
                except ValueError:
                    # list ?
                    conf['alpha'] = clargs.grouped['--alpha'].get(0)
            elif '--gmma' in key:
                conf['gmma'] = float(clargs.grouped['--gmma'].get(0))
            elif '--delta' in key:
                conf['delta'] = float(clargs.grouped['--delta'].get(0))
            elif '-g' in key:
                conf['_mode'] = 'generative'
            elif '-p' in key:
                conf['_mode'] = 'predictive'

        return cls(conf, usage=USAGE)

    # @askhelp
    # @askseed
    # @overload argparse
    @classmethod
    def exp_tabulate(cls, conf={}, usage=''):

        gargs = clargs.grouped['_'].all
        for arg in gargs:
            try:
                conf['K'] = int(arg)
            except:
                conf['model'] = arg
        return cls(conf, usage=usage)

    def parseargsexpe(self, usage=None, args=None):
        description = 'Specify an experimentation.'
        if not usage:
            epilog = self._examples
        else:
            epilog = usage
            pass

        parser = ExpArgumentParser(description=description, epilog=epilog,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)

        ### Global settings
        parser.add_argument(
            '--host',  default='localhost',
            help='name to append in data/<bdir>/<refdir>/ for th output path.')

        ### Global settings
        parser.add_argument(
            '-v', nargs='?', action=VerboseAction, dest='verbose', default=logging.INFO,
            help='Verbosity level (-v | -vv | -v 2)')
        parser.add_argument(
            '-s', '--simulate', action='store_true',
            help='Offline simulation')
        parser.add_argument(
            '--epoch', type=int,
            help='number for averaginf generative process')
        parser.add_argument(
            '-nld','--no-load-data', dest='load_data',  action='store_false', default=True,
            help='Try to load pickled frontend data')
        parser.add_argument(
            '--save-fr-data', dest='save_data',  action='store_true', default=False,
            help='Picked the frontend data.')
        parser.add_argument(
            '--seed', nargs='?', const=42, type=int,
            help='set seed value.')
        parser.add_argument(
            '-w', '--write', action='store_true', default=False,
            help='Write Fitted Model On disk.')

        ### Expe Settings
        ### get it from key -- @chatting
        parser.add_argument(
            '_do', nargs='*',
            help='Commands to pass to sub-machines.')

        parser.add_argument(
            '-c','--corpus', dest='corpus',
            help='ID of the frontend data.')
        parser.add_argument(
            '-r','--random', dest='corpus',
            help='Random generation of synthetic frontend  data [uniforma|alternate|cliqueN|BA].')
        parser.add_argument(
            '-m','--model', dest='model',
            help='ID of the model.')
        parser.add_argument(
            '-n','--N', type=str,
            help='Size of frontend data [int | all].')
        parser.add_argument(
            '-k','--K', type=int,
            help='Latent dimensions')
        parser.add_argument(
            '-i','--iterations', type=int,
            help='Max number of iterations for the optimization.')
        parser.add_argument(
            '--repeat', type=check_positive_integer,
            help='Index of tn nth repetitions/randomization of an design of experiments. Impact the outpout path as data/<bdir>/<refdir>/<repeat>/...')
        parser.add_argument(
            '--hyper', dest='hyper', type=str,
            help='type of hyperparameters optimization [auto|fix|symmetric|asymmetric]')
        parser.add_argument(
            '--hyper-prior','--hyper_prior', dest='hyper_prior', action='append',
            help='Set paramters of the hyper-optimization [auto|fix|symmetric|asymmetric]')
        parser.add_argument(
            '--refdir', '--debug', dest='refdir',
            help='Name to append in data/<bdir>/<refdir>/ for th output path.')
        parser.add_argument(
            '--testset-ratio',  dest='testset_ratio', type=float,
            help='testset/learnset ratio for testing.')
        parser.add_argument(
            '--homo', type=str,
            help='Centrality type (NotImplemented)')
        parser.add_argument(
            '--script', nargs='*',  type=list,
            help='Script request : name *args.')

        #s, remaining = parser.parse_known_args(args=args)
        s = parser.parse__args(args=args)
        settings = vars(s)
        # Remove None value
        settings = dict((key,value) for key, value in settings.items() if value is not None)

        self.argparser = parser

        return settings

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

    def simulate(self):
        print('Exp : %s' % self.exp_tensor.name())
        print(self.exp_tensor.table())
        exit(2)


    @staticmethod
    def setup_logger(fmt='%(message)s', level=logging.INFO, name='root'):
        #formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        if level == 1:
            level = logging.DEBUG
        elif level == 2:
            print ('what level of verbosity heeere ?')
            exit(2)
        else:
            # make a --silent option for juste error and critial log ?
            level = logging.INFO

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

    # @do parralize
    def pymake(self, sandbox=ExpeFormat):
        ''' Walk Trough experiments.  '''

        sandbox.preprocess(self)

        for id_expe, expe in enumerate(self.lod):
            pt = dict((key, value.index(expe[key])) for key, value in self.exp_tensor.items() if isinstance(expe[key], basestring))
            pt['expe'] = id_expe
            _expe = argparse.Namespace(**expe)

            # Init Expe
            try:
                expbox = sandbox(pt, _expe, self)
            except Exception as e:
                print(('Error during '+colored('%s', 'red')+' Initialization.') % (str(sandbox)))
                traceback.print_exc()
                exit()

            # Setup handler
            if hasattr(_expe, '_do') and len(_expe._do) > 0:
                # ExpFormat task ? decorator and autoamtic argument passing ....
                pmk = getattr(expbox, _expe._do[0])
                do = _expe._do
            else:
                pmk = expbox
                do = ['__call__']

            # Launch handler
            args = do[1:]
            try:
                try:
                    pmk(*args)
                except Exception as e:
                    print(('Error during '+colored('%s', 'red')+' Expe.') % (do))
                    traceback.print_exc()
                    exit()
            except KeyboardInterrupt:
                # it's hard to detach matplotlib...
                break

        return sandbox.postprocess(self)


