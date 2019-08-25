import sys
from loguru import logger

class LogFormatter(object):

    DEBUG = "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    INFO = "<level>{message}</level>"

    def __init__(self):
        self.padding = 0

        #fmt = "{time} | {level: <8} | {name}:{function}:{line}{extra[padding]} | {message}\n{exception}"
        #fmt = "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

        self.fmt_debug = self.DEBUG

        self.fmt_info = self.INFO

        self.fmt_warning = self.fmt_debug
        self.fmt_error = self.fmt_debug
        self.fmt_critical = self.fmt_debug


    def format(self, record):
        level = record['level'].lower()
        fmt = getattr(self, 'fmt_'+level)
        return fmt + '\n' # <- bug ?


def setup_logger(level=None):

    logformatter = LogFormatter()
    level = 0 if level is None else level

    if level == -1: # --silent | -nv
        level = 'WARNING'
    elif level == 1: # -v
        level = 'DEBUG'
    elif level >= 2: # -vv
        level = 'VDEBUG'
    else: # default
        level = 'INFO'

    logger.remove()

    logger.add(sys.stderr, level=level, colorize=True, format=logformatter.format)

# For logging info prior calling setup_logger
# @DEBUG case -v -1, -nv
_LEVEL = 'DEBUG' if ('-v' in sys.argv or '-vv' in sys.argv) else 'INFO'
logger.remove()
logger.add(sys.stderr, level=_LEVEL, format=getattr(LogFormatter, _LEVEL)) # default formatting

