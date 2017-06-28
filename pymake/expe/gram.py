# -*- coding: utf-8 -*-
import sys, argparse
from functools import partial
from pymake import ExpVector

#class SmartFormatter(argparse.HelpFormatter):
#    # Unused -- see RawDescriptionHelpFormatter
#    def _split_lines(self, text, width):
#        if text.startswith('R|'):
#            return text[2:].splitlines()
#        # this is the RawTextHelpFormatter._split_lines
#        return argparse.HelpFormatter._split_lines(self, text, width)

class ExpArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(ExpArgumentParser, self).__init__(**kwargs)

    def error(self, message):
        self.print_usage()
        print('error', message)
        #self.print_help()
        sys.exit(2)

class VerboseAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if option_string in ('-nv', '--silent'):
            setattr(args, self.dest, -1)
        else:
            # print 'values: {v!r}'.format(v=values)
            if values==None:
                values='1'
            try:
                values=int(values)
            except ValueError:
                values=values.count('v')+1
            setattr(args, self.dest, values)

class exp_append(argparse.Action):
    def __init__(self, *args, **kwargs):
        self._type = kwargs.pop('_t', str)
        super(exp_append, self).__init__(*args, **kwargs)

    def __call__(self, parser, args, values, option_string=None):
        try:
            exp_values = []
            for v in values:
                if v == '_null':
                    exp_values.append(v)
                else:
                    exp_values.append(self._type(v))
            setattr(args, self.dest, ExpVector(exp_values))
        except Exception as e:
            parser.error(e)

class exp_uniq_append(argparse.Action):
    def __init__(self, *args, **kwargs):
        self._type = kwargs.pop('_t', str)
        super(exp_uniq_append, self).__init__(*args, **kwargs)

    def __call__(self, parser, args, values, option_string=None):
        try:
            _exp = getattr(args, self.dest) or []
            exp_values = []
            for v in values:
                if v == '_null':
                    exp_values.append(v)
                else:
                    exp_values.append(self._type(v))
            _exp.extend([exp_values])
            setattr(args, self.dest, ExpVector(_exp))
        except Exception as e:
            parser.error(e)


class unaggregate_append(argparse.Action):
    ''' Option that would not be aggregated in making command line'''
    def __call__(self, parser, namespace, values, option_string=None):
        uniq_values = values
        setattr(namespace, self.dest, uniq_values)

def check_positive_integer(value):
    try:
        ivalue = int(value)
    except:
        raise argparse.ArgumentTypeError("%s is invalid argument value, need integer." % value)

    if ivalue < 0:
        return ''
    else:
        return ivalue


_Gram = [
    #
    #
    #
    # Global settings
    #
    #   * No repetions
    #
    #

    '--host', dict(
        help='Host database'),

    '-v', dict(
        nargs='?', action=VerboseAction, dest='verbose',
        help='Verbosity level (-v | -vv | -v 2)'),
    '-nv', '--silent', dict(
        nargs=0, action=VerboseAction, dest='verbose',
        help='Silent option'),

    '-np', '--save_plot', dict(
        action='store_true', help="don't block figure"),

    '-s', '--simulate',  dict(
        action='store_true',
        help='Offline simulation'),

    '-nld','--no-load-data', dict(
        dest='load_data', action='store_false',
        help='Try to load pickled frontend data'),

    '--save-fr-data', dict(
        dest='save_data', action='store_true',
        help='Picked the frontend data.'),

    '-w', '--write', dict(
        action='store_true',
        help='Write Fitted Model On disk.'),

    '--seed', dict(
        nargs='?', const=True, type=int,
        help='set seed value. If no seed specified but flag given, it will save/load the current state.'),

    '--refdir', '--debug', dict(
        dest='refdir',
        help='Name to append in data/<bdir>/<refdir>/ for th output path.'),

    '--format', dict(
        dest='_format', type=str,
        help='File format for saving results and models.'),


    #
    #
    #
    #  Expe Settings -- Context-Free
    #
    #  * Are repeatable
    #
    #

    '--epoch', dict(
        nargs='*', action=partial(exp_append, _t=int),
        help='number for averaginf generative process'),

    '-c','--corpus', dict(
        nargs='*', dest='corpus', action=exp_append,
        help='ID of the frontend data.'),

    '-r','--random', dict(
        nargs='*', dest='corpus', action=exp_append,
        help='Random generation of synthetic frontend  data [uniforma|alternate|cliqueN|BA].'),

    '-m','--model', dict(
        nargs='*',dest='model', action=exp_append,
        help='ID of the model.'),

    '-n','--N', dict(
        nargs='*', action=exp_append, # str because keywords "all"
        help='Size of frontend data [int | all].'),

    '-k','--K', dict(
        nargs='*', action=partial(exp_append, _t=int),
        help='Latent dimensions'),

    '-i','--iterations', dict(
        nargs='*', action=partial(exp_append, _t=int),
        help='Max number of iterations for the optimization.'),

    '--repeat', dict(
        nargs='*', action=exp_append, #type=check_positive_integer,
        help='Index of tn nth repetitions/randomization of an design of experiments. Impact the outpout path as data/<bdir>/<refdir>/<repeat>/...'),

    '--hyper',  dict(
        dest='hyper', nargs='*', action=exp_append,
        help='type of hyperparameters optimization [auto|fix|symmetric|asymmetric]'),

    '--hyper-prior','--hyper_prior', dict(
        dest='hyper_prior', action=exp_uniq_append, nargs='*',
        help='Set paramters of the hyper-optimization [auto|fix|symmetric|asymmetric]'),

    '--testset-ratio', dict(
        dest='testset_ratio', nargs='*', action=partial(exp_append, _t=int),
        help='testset/learnset percentage for testing.'),

    '--homo', dict(
        nargs='*', action=exp_append,
        help='Centrality type (NotImplemented)'),

    '--alpha', dict(
        nargs='*', action=partial(exp_append, _t=float),
        help='First hyperparameter.'),
    '--gmma', dict(
        nargs='*', action=partial(exp_append, _t=float),
        help='Second hyperparameter.'),
    '--delta', dict(
        nargs='*', action=partial(exp_append, _t=float),
        help='Third hyperparameter.'),
    '--chunk', dict(
        nargs='*', action=partial(exp_append, _t=float),
        help='Chunk size for online learning.'),
    '--burnin', dict(
        nargs='*', action=partial(exp_append, _t=int),
        help='Number of samples used for burnin period.'),

    #
    #
    #
    #  Context-sensitive
    #
    #  * Special meaning arguments
    #
    #
    #
    '-g', '--generative',dict(
        dest='_mode', action='store_const', const='generative'),
    '-p', '--predictive', dict(
        dest='_mode', action='store_const', const='predictive'),
    #\#


    '_do', dict(
        nargs='*',
        help='Commands to pass to sub-machines.'),

    '--script', dict(
        nargs='*', action=unaggregate_append,
        help='Script request : name *args.'),
    '--bind', dict(
        type=str, dest='_bind', action='append',
        help='Rules to filter the Exp Request.'),

    '-l', '--list', dict(
        dest='do_list', const='expe',  nargs='?', action=unaggregate_append,
        help='Request to print informations.'),
    ]
