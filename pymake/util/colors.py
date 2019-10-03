__COLORS =   {
    'normal'    : '\033[0;37;40m%s',
    'bold'      : '\033[;1m%s\033[0m',
    'underline' : '\033[2;37;40m%S\033[0;37;40m',
    'bright'    : '\033[1;37;40m%s\033[0;37;40m',
    'negative'  : '\033[3;37;40m Negative Colour\033[0;37;40m',
    'blink'     : '\033[5;37;40m%s\033[0;37;40m',
    'dark grey' : '\033[1;30;40m%s\033[0m',
    'red'       : '\033[1;31;40m%s\033[0m',
    'green'     : '\033[1;32;40m%s\033[0m',
    'yellow'    : '\033[1;33;40m%s\033[0m',
    'blue'      : '\033[1;34;40m%s\033[0m',
    'magenta'   : '\033[1;35;40m%s\033[0m',
    'cyan'      : '\033[1;36;40m%s\033[0m',
    'white'     : '\033[1;37;40m%s\033[0m',
    'grey'      : '\033[0;37;40m%s\033[0m',
    'black'     : '\033[0;37;48m%s\033[0m',
}

def colored(astring, color):
    surround = __COLORS[color]
    return surround % astring
