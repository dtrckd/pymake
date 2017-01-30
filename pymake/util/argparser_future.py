import sys
import argparse
import logging

from util.utils import print_available_scenario

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

    return logger


class PMArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        #super(BhpArgumentParser, self).error(message)
        self.print_usage()
        print('error', message)
        #self.print_help()
        print()
        print_available_scenario()
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


def argParser():
    parser = PMArgumentParser(description='pymake arguments parser')
    parser.add_argument("experience", nargs=1,
                        help="Specify the experience")
    parser.add_argument('-v', nargs='?', action=VerboseAction, dest='verbose',
                        help='verbosity level (-v | -vv | -v 2)')
    parser.add_argument("-s", "--simulate", action="store_true",
                        help="offline simulation")


    settings = vars( parser.parse_args())

    setup_logger(fmt='%(message)s', level=settings['verbose'])

    return settings
