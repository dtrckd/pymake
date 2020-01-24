import sys
from loguru import logger

class LogFormatter(object):

    SINK = "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    INFO = "<level>{message}</level>"

    DEBUG=SINK

    def __init__(self, logger):
        self.padding = 0

        #fmt = "{time} | {level: <8} | {name}:{function}:{line}{extra[padding]} | {message}\n{exception}"
        #fmt = "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

        #self.fmt_vdebug = self.SINK
        self.fmt_trace = self.SINK
        self.fmt_debug = self.SINK
        self.fmt_info = self.INFO
        self.fmt_success = self.SINK
        self.fmt_warning = self.SINK
        self.fmt_error = self.SINK
        self.fmt_critical = self.SINK

        #self.extra_leval = ['vdebug']

        self.logger = logger


    def format(self, record):
        level = record['level'].lower()
        fmt = getattr(self, 'fmt_'+level)
        return fmt + '\n' # <- bug ?


#def vdebug(_, message, *args, **kwargs):
#    logger.opt(depth=1).log('vdebug', message, *args, **kwargs)



def setup_logger(level=None):

    logformatter = LogFormatter(logger)
    level = 0 if level is None else level


    if level == -1: # --silent | -nv
        level = 'WARNING'
    elif level == 0:
        # Default level
        level = 'INFO'
    elif level == 1: # -v
        level = 'DEBUG'
    elif level >= 2: # -vv
        level = 'TRACE'
        #level = 'VDEBUG'
    else:
        level = 'INFO'

    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True, format=logformatter.format)

    #logger.level("vdebug", no=33, icon="🤖", color="<blue>")
    #logger.__class__.vdebug = vdebug

# For logging info prior calling setup_logger
# @DEBUG case -v -1, -nv
_LEVEL = 'DEBUG' if ('-v' in sys.argv or '-vv' in sys.argv) else 'INFO'
logger.remove()
logger.add(sys.stderr, level=_LEVEL, format=getattr(LogFormatter, _LEVEL)) # default formatting

