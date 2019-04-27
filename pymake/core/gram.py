import sys, os
import argparse
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
        try:
            os.remove('.pmk-db.db')
        except FileNotFoundError:
            pass
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

    '-v', dict(dest='_verbose',
               nargs='?', action=VerboseAction,
               help='Verbosity level (-v | -vv | -v 2)'),

    '-nv', '--silent', dict(dest='_verbose',
                            nargs=0, action=VerboseAction,
                            help='Silent option'),

    '-s', '--simulate', dict(
        action='store_true',
        help='Offline simulation'),

    '-w', '--write', dict(dest='_write',
                          action='store_true',
                          help='Write Fitted Model On disk.'),

    '--seed', dict(dest='_seed',
        nargs='?', const=True,
        help='set seed value. If no seed specified but flag given, it will save/load the current state.'),

    '--cores', dict(dest='_cores',
                    type=int,
                    help='number of cores to run with runpara command.'),

    '-l', '--list', dict(dest='do_list',
                         nargs='?', const='topo', action=unaggregate_append,
                         help='Request to print informations.'),
    '-ll', dict(dest='do_list',
                nargs='?', const='spec_topo', action=unaggregate_append,
                help='Request to print global informations.'),

    '--net', dict(
        nargs='?', dest='_net', const=True,
        help='[with runpara, send run to remote via loginfile. Max number of remote can be specified.'),



    '-nsd', '--no-save-data', dict(dest='_force_save_data',
                                   action='store_false',
                                   help='Do no save the data frontend after parsing.'),

    '-nld', '--reload-data', dict(dest='_force_load_data',
                                  action='store_false',
                                  help='Reload data from raw parsing.'),

    '-nbp', '--no-block-plot', dict(dest='_no_block_plot',
                                    action='store_true',
                                    help='Make the pyplot figure non blocking.'),




    # @Debug allocate unique filname for expe base on a hash of its spec.
    '--ifu', '--ignore-format-unique', dict(dest='_ignore_format_unique',
                                        action='store_true',
                                            help='dont check that if there is some outputpath overlaping due to lacking parameters in  _format.'),


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

    '--bind', dict(dest='_bind',
        type=str, action='append',
        help='Rules to filter the Exp Request.'),

    '--repeat', dict(dest='_repeat',
        nargs='*', action=exp_append, #type=check_positive_integer,
        help='Index of tn nth repetitions/randomization of an design of experiments. Impact the outpout path as data/<bdir>/<refdir>/<repeat>/...'),

    '--refdir', dict(dest='_refdir',
        nargs='*', action=exp_append,
        help='Name to append in data/<data-typer>/<refdir>/ for the output path.'),

    '--format', dict(dest='_format',
         help='File format for saving results and models.'),

    '--type', '--data-format', dict(dest='_data_format',
         help='The type/format of data to use [b|w].'),

    '--fast', '--deactivate-measures', dict(dest='deactivate_measures',
         action='store_true',
         help='Do not compute measures (log-likelihood/entropy etc) during fitting to speed up the process.'),


    ]
