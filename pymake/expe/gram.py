import argparse

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


_Gram = [
    ### Global settings

    '--host', dict(
        help='name to append in data/<bdir>/<refdir>/ for th output path.'),

    '-v', dict(
        nargs='?', action=VerboseAction, dest='verbose',
        help='Verbosity level (-v | -vv | -v 2)'),

    '-s', '--simulate',  dict(
        action='store_true',
        help='Offline simulation'),

    ### Expe Settings -- Context-Free

    '--epoch', dict(
        type=int,
        help='number for averaginf generative process'),

    '-nld','--no-load-data', dict(
        dest='load_data', action='store_false',
        help='Try to load pickled frontend data'),

    '--save-fr-data', dict(
        dest='save_data', action='store_true',
        help='Picked the frontend data.'),

    '--seed', dict(
        nargs='?', const=42, type=int,
        help='set seed value.'),

    '-w', '--write', dict(
        action='store_true',
        help='Write Fitted Model On disk.'),

    '-c','--corpus', dict(
        nargs='*', dest='corpus',
        help='ID of the frontend data.'),

    '-r','--random', dict(
        nargs='*', dest='corpus',
        help='Random generation of synthetic frontend  data [uniforma|alternate|cliqueN|BA].'),

    '-m','--model', dict(
        nargs='*',dest='model',
        help='ID of the model.'),

    '-n','--N', dict(
        type=str,
        help='Size of frontend data [int | all].'),

    '-k','--K', dict(
        type=int,
        help='Latent dimensions'),

    '-i','--iterations', dict(
        type=int,
        help='Max number of iterations for the optimization.'),

    '--repeat', dict(
        type=check_positive_integer,
        help='Index of tn nth repetitions/randomization of an design of experiments. Impact the outpout path as data/<bdir>/<refdir>/<repeat>/...'),

    '--hyper',  dict(
        type=str, dest='hyper',
        help='type of hyperparameters optimization [auto|fix|symmetric|asymmetric]'),

    '--hyper-prior','--hyper_prior', dict(
        dest='hyper_prior', action='append',
        help='Set paramters of the hyper-optimization [auto|fix|symmetric|asymmetric]'),

    '--refdir', '--debug', dict(
        dest='refdir',
        help='Name to append in data/<bdir>/<refdir>/ for th output path.'),

    '--testset-ratio', dict(
        dest='testset_ratio', type=int,
        help='testset/learnset percentage for testing.'),

    '--format', dict(
        dest='_format', type=str,
        help='File format for saving results and models.'),

    '--homo', dict(
        type=str,
        help='Centrality type (NotImplemented)'),

    # Context-sensitive

    '_do', dict(
        nargs='*',
        help='Commands to pass to sub-machines.'),

    '--script', dict(
        nargs='*',
        help='Script request : name *args.'),

    '-l', '--list', dict(
        action='store_true', dest='do_list',
        help='Request to print informations.'),
    ]
