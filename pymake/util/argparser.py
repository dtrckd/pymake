# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import argparse
import logging

import inspect
from functools import wraps
import args as clargs
from pymake.frontend.frontend_io import *
from pymake.expe.spec import _spec_; _spec = _spec_()

'''
    Here we build Custom Args Parser
'''

#################
### ARGPARSE ZONE
#################

def setup_logger(fmt='%(message)s', level=logging.INFO, name='root'):
    #formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    if level == 1:
        level = logging.DEBUG
    elif level == 2:
        print ('what level of verbosity heeere ?')
        exit()
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
    # who reults in logging things multiple times
    logger.propagate = False

    return logger

class ExpArgumentParser(argparse.ArgumentParser):

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


def check_positive_integer(value):
    try:
        ivalue = int(value)
    except:
        raise argparse.ArgumentTypeError("%s is invalid argument value, need integer." % value)

    if ivalue < 0:
        return ''
    else:
        return ivalue
#\

#########
# @TODO:
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

class askhelp(object):
    ''' Print help and exit on -h'''
    def __init__(self, func, help=False):
        self.func = func
        self.help = help
        #wraps(func)(self)
        #functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        try:
            usage = args[0]
        except:
            usage = self.func.__name__ + ' ?'
        # function name
        #(inspect.currentframe().f_code.co_name
        if clargs.flags.contains('--help') or clargs.flags.contains('-h') or self.help:
            print(usage)
            exit()

        response = self.func(*args, **kwargs)
        return response

class askverbose(object):
    ''' Augment verbosity on -c '''
    def __init__(self, func):
        self.func = func
        #wraps(func)(self)
        #functools.update_wrapper(self, func)
        pass

    def __call__(self, *args, **kwargs):

        if clargs.flags.contains('-v'):
            self.logger = self.setup_logger('root','%(message)s', logging.DEBUG)
        else:
            self.logger = self.setup_logger('root','%(message)s', logging.INFO)

        response = self.func(*args, **kwargs)

        if clargs.flags.contains('-s'):
            response['simul'] = True

        return response

    def setup_logger(self, name, fmt, level):
        #formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        # Get logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Format handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=fmt))
        logger.addHandler(handler)

        return logger


class argparser(object):
    ''' Utility class for parsing arguments of various script of @project.
        Each method has the same name of the function/scrpit for which it is used.
        @return dict of variables used by function/scritpts
    '''

    @staticmethod
    @askhelp
    @askverbose
    def zymake(USAGE=''):
        ''' Generates output (files or line arguments) according to the SPEC
            @return OUT_TYPE: runcmd or path
                    SPEC: expe spec
                    FTYPE: filetype targeted
                    STATUS: status of file required  on the filesystem
        '''
        USAGE = '''\
                # Usage:
            zymake path[default] SPEC Filetype(pk|json|inf)
            zymake runcmd SPEC
            zymake -l : show available spec
            '''

        # Default request
        req = dict(
            OUT_TYPE = 'path',
            request = 'RUN_DD',
            SPEC = _spec.RUN_DD, # gatattr(_spec, request)
            FTYPE = 'pk',
            STATUS = None )

        ontologies = dict(
            out_type = ('runcmd', 'path'),
            spec = list(map(str.lower, _spec.repr())),
            ftype = ('json', 'pk', 'inf') )

        ### Making ontologie based argument attribution
        ### Used the grouped argument whitout flags ['_']
        gargs = clargs.grouped['_'].all
        checksum = len(gargs)
        for v in gargs:
            v = v.lower()
            for ont, words in ontologies.items():
                if v in words:
                    if ont == 'spec':
                        req['request'] = v
                        v = getattr(_spec, v.upper())
                    req[ont.upper()] = v
                    checksum -= 1
                    break

        if '-status' in clargs.grouped:
            # debug status of filr (path)
            req['STATUS'] = clargs.grouped['-status'].get(0)
        if '-l' in clargs.grouped or '--list' in clargs.grouped:
            # debug show spec
            req['OUT_TYPE'] = 'list'

        if checksum != 0:
            raise ValueError('unknow argument: %s\n available SPEC: %s' % (gargs, _spec.repr()))
        return req

    @staticmethod
    @askverbose
    @askhelp
    @askseed
    def generate(USAGE=''):
        conf = {}
        write_keys = ('-w', 'write', '--write')
        # Write plot
        for write_key in write_keys:
            if write_key in clargs.all:
                conf['save_plot'] = True

        gargs = clargs.grouped.pop('_')

        ### map config dict to argument
        for key in clargs.grouped:
            if '-k' == key:
                conf['K'] = int(clargs.grouped['-k'].get(0))
            elif '-n' in key:
                conf['gen_size'] = int(clargs.grouped['-n'].get(0))
            elif '--alpha' in key:
                try:
                    conf['alpha'] = float(clargs.grouped['--alpha'].get(0))
                except ValueError:
                    conf['hyper'] = clargs.grouped['--alpha'].get(0)
            elif '--gmma' in key:
                conf['gmma'] = float(clargs.grouped['--gmma'].get(0))
            elif '--delta' in key:
                conf['delta'] = float(clargs.grouped['--delta'].get(0))
            elif '-g' in key:
                conf['generative'] = 'evidence'
            elif '-p' in key:
                conf['generative'] = 'predictive'
            elif '-m' in key:
                conf['model'] = clargs.grouped['-m'].get(0)

        if clargs.last and clargs.last not in map(str, clargs.flags.all + list(conf.values())):
            conf['do'] = clargs.last

        return conf

    @staticmethod
    @askverbose
    @askhelp
    def exp_tabulate(USAGE=''):
        conf = {}

        gargs = clargs.grouped['_'].all
        for arg in gargs:
            try:
                conf['K'] = int(arg)
            except:
                conf['model'] = arg
        return conf


    @staticmethod
    def gramexp():
        parser = ExpArgumentParser(description='Default load corpus and run a model.',
                                   epilog='''Examples: \n
                                   # Load corpus and infer modef (eg LDA)
                                   ./lda_run.py -k 6 -m ldafullbaye -p:
                                   # assort
                                   ./assortt.py -n 1000 -k 10 --alpha auto --homo 0 -m ibp_cgs -c generator3 -l model --refdir debug5 -nld
                                   # Load corpus and model:
                                   ./lda_run.py -k 6 -m ldafullbaye -lall -p
                                   # Network corpus:
                                   ./fit.py -m immsb -c generator1 -n 100 -i 10
                                   # Various networks setting:
                                   ./fit.py -m ibp_cgs --homo 0 -c clique6 -n 100 -k 3 -i 20
                                   ''')

        ### Global settings
        parser.add_argument(
            '--host',  default='localhost',
            help='name to append in data/<bdir>/<refdir>/ for th output path.')

        ### I/O settings
        parser.add_argument(
            'datatype', nargs='?',
            help='Force the type of data (NotImplemented')
        parser.add_argument(
            '-v', nargs='?', action=VerboseAction, dest='verbose', default=logging.INFO,
            help='verbosity level (-v | -vv | -v 2)')
        parser.add_argument(
            '-s', '--simulate', action='store_true',
            help='offline simulation')
        parser.add_argument(
            '--epoch', type=int,
            help='number for averaginf generative process')
        parser.add_argument(
            '-p', '--predict', type=str,
            help='Do predict some data (NotImplemented: precise fit/predict..)')
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

        ### Exp Settings
        ### get it from key -- chatting

        #parser.add_argument( ?
        #    '-d', '--datatype','--data_type','--data-type',  type=str,
        #    help='Set type of data and relative path data/<bdir>/<datatype>/...')

        parser.add_argument(
            '-c','--corpus', '--corpus_name', '--corpus-name', dest='corpus_name',
            help='ID of the frontend data.')
        parser.add_argument(
            '-r','--random', dest='corpus_name',
            help='Random generation of synthetic frontend  data [uniforma|alternate|cliqueN|BA].')
        parser.add_argument(
            '-m','--model', '--model_name', '--model-name', dest='model_name',
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
            '--alpha','--hyper', dest='hyper', type=str,
            help='type of hyperparameters optimization [auto|fix|symmetric|asymmetric]')
        parser.add_argument(
            '--hyper-prior','--hyper_prior', dest='hyper_prior', action='append',
            help='Set paramters of the hyper-optimization [auto|fix|symmetric|asymmetric]')
        parser.add_argument(
            '--refdir', '--debug', dest='refdir', default='debug',
            help='name to append in data/<bdir>/<refdir>/ for th output path.')

        parser.add_argument(
            '--homo', type=str,
            help='Centrality type (NotImplemented)')

        settings = vars( parser.parse_args())
        # Remove None value
        settings = dict((key,value) for key, value in settings.items() if value is not None)
        setup_logger(fmt='%(message)s', level=settings['verbose'])

        return settings

    @staticmethod
    def simulate(exp):
        ''' Simulation Output '''
        if exp.get('simulate'):
            print('''--- Simulation settings ---
            Model : %s
            Corpus : %s
            K : %s
            N : %s
            hyper : %s ''' % (exp['model_name'], exp['corpus_name'],
                         exp['K'], exp['N'], exp['hyper'])
                 )
            exit()
