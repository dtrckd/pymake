# -*- coding: utf-8 -*-
import inspect
from functools import wraps
import args as clargs
from pymake.frontend.frontend_io import *
from pymake.expe.spec import _spec_; _spec = _spec_()

#########
# @TODO:
#   * wraps cannot handle the decorator chain :(, why ?


class askseed(object):
    """ Load previous random seed """
    def __init__(self, func, help=False):
        self.func = func
    def __call__(self, *args, **kwargs):

        response = self.func(*args, **kwargs)

        if clargs.flags.contains('--seed'):
            response['seed'] = True
        return response

class askhelp(object):
    """ Print help and exit on -h"""
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
    """ Augment verbosity on -c """
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
    """ Utility class for parsing arguments of various script of @project.
        Each method has the same name of the function/scrpit for which it is used.
        @return dict of variables used by function/scritpts
    """

    @staticmethod
    @askhelp
    @askverbose
    def zymake():
        """ Generates output (files or line arguments) according to the SPEC
            @return OUT_TYPE: runcmd or path
                    SPEC: expe spec
                    FTYPE: filetype targeted
                    STATUS: status of file required  on the filesystem
        """
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
                conf['write_to_file'] = True

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
    def expe_tabulate(USAGE=''):
        conf = {}

        gargs = clargs.grouped['_'].all
        for arg in gargs:
            try:
                conf['K'] = int(arg)
            except:
                conf['model'] = arg
        return conf


