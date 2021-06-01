#!/usr/bin/python3
#import pyximport; pyximport.install(pyimport = True)
import sys, os
from pymake import GramExp


''' A Command line controler of Pymake '''


def bootstrap():
    # Manage working directory

    env = dict(os.environ)
    pwd = env.get('PWD')
    ## change directory if asked
    if '-cd' in sys.argv:
        i = sys.argv.index('-cd')
        p = sys.argv[i+1]
        os.chdir(p)

        # Debug ?
        sys.argv.pop(i+1)
        sys.argv.pop(i)
        pwd = os.path.join(pwd, p)
        env['PWD'] = pwd
    else:
        # @Warning: prevent other library from moving pymake CWD
        os.chdir(env.get('PWD'))

    ## Search in the project and current repo. Awesome !
    # @Warning: can cause import conflict if folder/file name confilct with systemo module
    #sys.path.insert(0, pwd + '/.')
    #sys.path.insert(0, pwd + '/..')
    sys.path.append(pwd + '/.')
    sys.path.append(pwd + '/..')

    return env


def main():

    env = bootstrap()
    GramExp.setenv(env)

    gexp = GramExp.zymake()
    expe = gexp._conf

    if expe.get('simulate') and (not expe['_do'] in ['run', 'runpara', 'path'] or not expe.get('script')):
        # same as show !
        gexp.simulate()

    lines = None
    line_prefix = ''
    if expe['_do'] == 'init':
        gexp.init_folders()
        exit()
    else:
        if (expe['_do'] or expe.get('do_list')) and not GramExp.is_pymake_dir():
            print('fatal: Not a pymake directory: %s not found.' % (GramExp._cfg_name))
            exit(10)

    if expe['_do'] == 'update':
        gexp.update_index()
    elif expe['_do'] == 'show':
        gexp.simulate()
    elif expe['_do'] in ['doc', 'help']:
        lines = gexp.show_doc()
    elif expe['_do'] == 'run':
        lines = gexp.execute()
    elif expe['_do'] == 'runpara':
        is_distrib = expe.get('_net')
        if is_distrib:
            if is_distrib != True:
                nhosts = int(is_distrib)
            else:
                nhosts = None
            lines = gexp.execute_parallel_net(nhosts)
        else:
            lines = gexp.execute_parallel()
        gexp.pushcmd2hist()
    elif expe['_do'] == 'cmd':
        lines = gexp.make_commandline()
    elif expe['_do'] == 'path':
        lines = gexp.make_path(ext=expe.get('_ext'), status=expe.get('_status'))
    elif expe['_do'] == 'hist':
        lines = gexp.show_history()
    elif expe['_do'] == 'diff':
        lines = gexp.show_diff()
    elif expe['_do'] == 'notebook': # @Todo
        lines = gexp.notebook()
    else: # list things

        if not 'do_list' in expe and expe['_do']:
            raise ValueError('Unknown command : %s' % expe['_do'])

        if 'spec' == expe.get('do_list'):
            print(gexp.spectable())
        elif 'model' == expe.get('do_list'):
            print(gexp.modeltable())
        elif 'script' == expe.get('do_list'):
            print(gexp.scripttable())
        elif 'model_topo' == expe.get('do_list'):
            print(gexp.modeltable(_type='topos'))
        elif 'spec_topo' == expe.get('do_list'):
            print(gexp.spectable_topo())
        elif 'topo' == expe.get('do_list'):
            print(gexp.alltable_topo())
        else:
            print(gexp.help_short())
            if expe.get('do_list'):
                print('Unknow options %s ?' % expe.get('do_list'))
        exit()

    if lines is None:
        # catch signal ?
        exit()

    if 'script' in expe:
        lines = [' '.join((line_prefix, l)).strip() for l in lines]

    gexp.simulate(halt=False, file=sys.stderr)
    print('\n'.join(lines), file=sys.stdout)


if __name__ == '__main__':
    main()
