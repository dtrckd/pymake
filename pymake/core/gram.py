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



# <begin arg-semantics>

class exp_append(argparse.Action):
    ''' Append arguments in e expTensor by repetition after a flag (ie -n 10 20 30...).
        If several flags are present the last one will overwrite other.
    '''
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

class exp_append_uniq(argparse.Action):
    ''' Append arguments in e expTensor by flag repetition (ie -q 10 -q 20...).
        It is useful to hanlde arguments that are tuple or list in a flag.
    '''
    def __init__(self, *args, **kwargs):
        self._type = kwargs.pop('_t', str)
        super(exp_append_uniq, self).__init__(*args, **kwargs)

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

# </end arg sementics>


_Gram = [
    # Global settings
    #
    #   * No repetions

    '--host', dict(
        help='Host database'),

    '-v', dict(
        nargs='?', action=VerboseAction, dest='verbose',
        help='Verbosity level (-v | -vv | -v 2)'),
    '-nv', '--silent', dict(
        nargs=0, action=VerboseAction, dest='verbose',
        help='Silent option'),

    '-s', '--simulate',  dict(
        action='store_true',
        help='Offline simulation'),

    '-w', '--write', dict(
        action='store_true',
        help='Write Fitted Model On disk.'),

    '--seed', dict(
        nargs='?', const=True, type=int,
        help='set seed value. If no seed specified but flag given, it will save/load the current state.'),

    '--cores', dict(
        type=int, dest='_cores',
        help='number of cores to run with runpara command.'),

    '-l', '--list', dict(
        nargs='?', dest='do_list', const='spec', action=unaggregate_append,
        help='Request to print informations.'),
    '-ll', dict(
        nargs='?', dest='do_list', const='topo', action=unaggregate_append,
        help='Request to print global informations.'),

    '--net', dict(
        nargs='?', dest='_net', const=True,
        help='[with runpara, send run to remote via loginfile. Max number of remote can be specified.'),

    '--ifu', '--ignore-format-unique', dict(
        action='store_true', dest='_ignore_format_unique',
        help='dont check that if _format overllaping expe outpath name.'),



    #  Context-sensitive
    #
    #  * Special meaning arguments


    '_do', dict(
        nargs='*',
        help='Commands to pass to sub-machines.'),

    '-x', '--script', dict(
        nargs='*', action=unaggregate_append,
        help='Script request : name *args.'),

    '--pmk', dict(
        nargs='*', dest='_pmk', action=unaggregate_append,
        help='force a an expe settings ex: --pmk myvar=2'),

    '--bind', dict(
        type=str, dest='_bind', action='append',
        help='Rules to filter the Exp Request.'),

    '--repeat', dict(
        nargs='*', dest='_repeat', action=exp_append, #type=check_positive_integer,
        help='Index of tn nth repetitions/randomization of an design of experiments. Impact the outpout path as data/<bdir>/<refdir>/<repeat>/...'),

    '--refdir', dict(
        nargs='*', dest='_refdir', action=exp_append,
        help='Name to append in data/<data-typer>/<refdir>/ for the output path.'),

    '--data-type', dict(
        dest='_data_type',
        help='Name to prepend in data/<data-type>/<refdir>/ for the output path.'),

    '--data-format', dict(
        dest='_data_format',
        help='The type/format of data to use.'),

    '--format', dict(
        dest='_format', type=str,
        help='File format for saving results and models.'),

    ]
