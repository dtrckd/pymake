import os
from collections import defaultdict
from string import Template
import shelve
import tempfile
import configparser


global __config
__config = {}


class PmkTemplate(Template):
    delimiter = '$$'
    #idpattern = r'[a-z][_a-z0-9]*'



def parse_file_conf(fn, dir, sep='=', comments=('#','%')):
    #with open(fn) as f:
    #    parameters = f.read()
    #parameters = filter(None, parameters.split('\n'))
    #parameters = dict((p[0].strip(), p[1].strip()) for p in (t.strip().split(sep) for t in parameters if not t.strip().startswith(comments)))
    #for k, v in parameters.items():
    #    if  '.' in v:
    #        try:
    #            parameters[k] = float(v)
    #        except:
    #            pass
    #    else:
    #        try:
    #            parameters[k] = int(v)
    #        except:
    #            pass
    #return parameters

    config = configparser.ConfigParser()


    try:
        c = config.read(fn)
    except configparser.MissingSectionHeaderError as e:
        # Add a dummy header
        with open(fn) as _f:
            c = config.read_string('[top]\n'+_f.read())

    for s in config.sections():
        for k,v in config.items(s):
            # Preprocess options
            if '~' in v:
                v = os.path.expanduser(v)
            elif v and (v.startswith(('./', '../')) or not v.startswith('/')):
                fpath = os.path.join(dir, v)
                if os.path.exists(fpath):
                    v = fpath

            __config[k] = v

    return __config


def get_db_file(name="pmk-db"):
    fn = os.path.join(tempfile.gettempdir(), name, hex(hash(os.getcwd()) & ((1<<32)-1)))
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    return fn


def reset_pymake_settings(settings, cfg_name='pmk.cfg', db_name='pmk-db'):
    cwd = os.path.dirname(__file__)
    with open(os.path.join(cwd, '..', 'template', '%s.template'%(cfg_name))) as _f:
        template = PmkTemplate(_f.read())
        try:
            ctnt = template.substitute(settings)
        except KeyError as e:
            print('The following key is missing in the config file: %s' % e)
            print('aborting...')
            exit(10)

    try:
        db = shelve.open(get_db_file(db_name))
        dir = db['PWD']
        db.close()
    except Exception as e:
        print("Bootstrap error (%s) => PWD path not initialized ? key: %s" % (e, key))
        dir = os.getenv('PWD')

    cfg_file = os.path.join(dir, cfg_name)
    with open(cfg_file, 'wb') as _f:
        return _f.write(ctnt.encode('utf8'))

def get_pymake_settings(key, cfg_name='pmk.cfg', db_name='pmk-db'):
    try:
        db = shelve.open(get_db_file(db_name))
        dir = db['PWD']
        db.close()
    except Exception as e:
        print("Bootstrap error (%s) => PWD path not initialized ? key: %s" % (e, key))
        #print("probably mismatch between the python working directory (%s) and the bash working directory (%s)" % (os.getcwd(), dir))
        dir = os.getenv('PWD')

    if key == 'PWD':
        return dir

    if not __config :
        cfg_file = os.path.join(dir, cfg_name)
        parse_file_conf(cfg_file, dir)

    config = __config

    if key.startswith('_'):
        # @DEBUG
        # pull / push etc
        value = [config['default' + key]]
    else:

        if not key in config:
            raise AttributeError("Error: '%s` key not in pmk.cfg. aborting..." % (str(key)))

        value = config[key]


    return value

