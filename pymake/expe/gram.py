# -*- coding: utf-8 -*-
import argparse
from pymake import ExpVector

class SmartFormatter(argparse.HelpFormatter):
    # Unused -- see RawDescriptionHelpFormatter
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)

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
    def __call__(self, parser, args, values, option_string=None):
        setattr(args, self.dest, ExpVector(values))


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
        help='name to append in data/<bdir>/<refdir>/ for th output path.'),

    '-v', dict(
        nargs='?', action=VerboseAction, dest='verbose',
        help='Verbosity level (-v | -vv | -v 2)'),
    '-nv', '--silent', dict(
        nargs=0, action=VerboseAction, dest='verbose',
        help='Silent option'),

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
        nargs='?', const=42, type=int,
        help='set seed value.'),

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
    #  * Are repatable
    #
    #

    '--epoch', dict(
        type=int, nargs='*', action=exp_append,
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
        type=str, nargs='*', action=exp_append,
        help='Size of frontend data [int | all].'),

    '-k','--K', dict(
        type=int, nargs='*', action=exp_append,
        help='Latent dimensions'),

    '-i','--iterations', dict(
        type=int, nargs='*', action=exp_append,
        help='Max number of iterations for the optimization.'),

    '--repeat', dict(
        nargs='*', action=exp_append, #type=check_positive_integer,
        help='Index of tn nth repetitions/randomization of an design of experiments. Impact the outpout path as data/<bdir>/<refdir>/<repeat>/...'),

    '--hyper',  dict(
        type=str, dest='hyper', nargs='*', action=exp_append,
        help='type of hyperparameters optimization [auto|fix|symmetric|asymmetric]'),

    '--hyper-prior','--hyper_prior', dict(
        dest='hyper_prior', action='append', nargs='*',
        help='Set paramters of the hyper-optimization [auto|fix|symmetric|asymmetric]'),

    '--testset-ratio', dict(
        dest='testset_ratio', type=str, nargs='*', action=exp_append,
        help='testset/learnset percentage for testing.'),

    '--homo', dict(
        type=str, nargs='*', action=exp_append,
        help='Centrality type (NotImplemented)'),

    #
    #
    #
    #  Context-sensitive
    #
    #  * Special meaning arguments
    #
    #

    '_do', dict(
        nargs='*',
        help='Commands to pass to sub-machines.'),

    '--script', dict(
        nargs='*',
        help='Script request : name *args.'),
    '--bind', dict(
        type=str, dest='_bind',
        help='Rules to filter the Exp Request.'),

    '-l', '--list', dict(
        action='store_true', dest='do_list',
        help='Request to print informations.'),
    ]
