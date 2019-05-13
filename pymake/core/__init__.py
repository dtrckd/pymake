import os
from collections import defaultdict
from string import Template
import shelve


# Global settings
__default_config = defaultdict(lambda: '', dict(project_data = os.path.expanduser('data/'),
                                                project_figs = os.path.expanduser('data/plot/figs/'),
                                                # @debug repo access ??
                                                default_spec = 'pymake.spec',
                                                default_script = 'pymake.script',
                                                default_model = 'pymake.model',
                                                default_corpus = '?')
                              )

class PmkTemplate(Template):
    delimiter = '$$'
    #idpattern = r'[a-z][_a-z0-9]*'


def parse_file_conf(fn, sep=':', comments=('#','%')):
    with open(fn) as f:
        parameters = f.read()
    parameters = filter(None, parameters.split('\n'))
    parameters = dict((p[0].strip(), p[1].strip()) for p in (t.strip().split(sep) for t in parameters if not t.strip().startswith(comments)))
    for k, v in parameters.items():
        if  '.' in v:
            try:
                parameters[k] = float(v)
            except:
                pass
        else:
            try:
                parameters[k] = int(v)
            except:
                pass
    return parameters

def reset_pymake_settings(settings, default_config=__default_config, cfg_name='pmk.cfg', db_name='.pmk-db'):
    _settings = default_config.copy()
    _settings.update(settings)
    #ctnt = '\n'.join(('{0} = {1}'.format(k,v) for k,v in _settings.items()))
    cwd = os.path.dirname(__file__)
    with open(os.path.join(cwd, '..', 'template', '%s.template'%(cfg_name))) as _f:
        template = PmkTemplate(_f.read())
        ctnt = template.substitute(_settings)

    try:
        db = shelve.open(os.path.join(os.getcwd(), db_name))
        dir = db['PWD']
        db.close()
    except Exception as e:
        print("Bootstrap warning (%s) => PWD path not initialized ? key: %s" % (e, key))
        dir = os.getenv('PWD')

    cfg_file = os.path.join(dir, cfg_name)
    with open(cfg_file, 'wb') as _f:
        return _f.write(ctnt.encode('utf8'))

def get_pymake_settings(key=None, default_config=__default_config, cfg_name='pmk.cfg', db_name='.pmk-db'):
    try:
        db = shelve.open(os.path.join(os.getcwd(), db_name))
        dir = db['PWD']
        db.close()
    except Exception as e:
        print("Bootstrap warning (%s) => PWD path not initialized ? key: %s" % (e, key))
        #print("probably mismatch between the python working directory (%s) and the bash working directory (%s)" % (os.getcwd(), dir))
        dir = os.getenv('PWD')

    cfg_file = os.path.join(dir, cfg_name)

    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(os.path.expanduser('~') ,'.pymake', cfg_name)
        if not os.path.isfile(cfg_file):
            dir_cfg = make_path(cfg_file)
            #ctnt = '\n'.join(('{0} = {1}'.format(k,v) for k,v in  default_config.items()))
            ctnt = ''
            with open(cfg_file, 'wb') as _f:
                _f.write(ctnt.encode('utf8'))

    config = parse_file_conf(cfg_file, sep='=')

    for k in list(config):
        v = config[k]
        if '~' in v:
            config[k] = os.path.expanduser(v)
        elif v and (v.startswith(('./', '../')) or not v.startswith('/')):
            fpath = os.path.join(dir, v)
            if os.path.exists(fpath):
                config[k] = fpath

    if not key:
        settings = config
    elif key.startswith('_'):
        res = []
        for k in ['default'+key, 'contrib'+key]:
            res += config.get(k, default_config[k]).split(',')
        settings =  [e for e in map(str.strip, res) if e]
    else:
        settings = config.get(key, default_config[key])

    return settings
