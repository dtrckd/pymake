#!/usr/bin/python3
#import pyximport; pyximport.install(pyimport = True)
import sys, os
from pymake import GramExp


''' A Command line controler of Pymake '''

def bootstrap():

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
        # @debug this, I dont know whyiam in lib/package sometimes, annoying !
        os.chdir(env.get('PWD'))


    ## Search in the project and current repo. Awesome !
    sys.path.insert(0, pwd + '/.')
    sys.path.insert(0, pwd + '/..')

    return env



def main():

    env = bootstrap()
    GramExp.setenv(env)

    zymake = GramExp.zymake()
    zyvar = zymake._conf

    if zyvar.get('simulate') and ( not zyvar['_do'] in ['run', 'runpara', 'path'] or not zyvar.get('script')):
        # same as show !
        zymake.simulate()

    lines = None
    line_prefix = ''
    if zyvar['_do'] == 'init':
        zymake.init_folders()
        exit()

    if zyvar['_do'] == 'update':
        zymake.update_index()
    elif zyvar['_do'] == 'show':
        zymake.simulate()
    elif zyvar['_do'] ==  'run':
        lines = zymake.execute()
    elif zyvar['_do'] == 'runpara':
        is_distrib = zyvar.get('_net')
        if is_distrib:
            if is_distrib != True:
                nhosts = int(is_distrib)
            else:
                nhosts = None
            lines = zymake.execute_parallel_net(nhosts)
        else:
            lines = zymake.execute_parallel()
        zymake.pushcmd2hist()
    elif zyvar['_do'] == 'cmd':
        lines = zymake.make_commandline()
    elif zyvar['_do'] == 'path':
        lines = zymake.make_path(ext=zyvar.get('_ext'), status=zyvar.get('_status'))
    elif zyvar['_do'] == 'hist':
        lines = zymake.show_history()
    elif zyvar['_do'] == 'diff':
        lines = zymake.show_diff()
    elif zyvar['_do'] == 'notebook': # @Todo
        lines = zymake.notebook()
    else: # list things

        if not 'do_list' in zyvar and zyvar['_do']:
            raise ValueError('Unknown command : %s' % zyvar['_do'])

        if 'spec' == zyvar.get('do_list'):
            print (zymake.spectable())
        elif 'model' == zyvar.get('do_list'):
            print (zymake.modeltable())
        elif 'script' == zyvar.get('do_list'):
            print(zymake.scripttable())
        elif 'model_topos' == zyvar.get('do_list'):
            print (zymake.modeltable(_type='topos'))
        elif 'spec_topo' ==  zyvar.get('do_list'):
            print (zymake.spectable_topo())
        elif 'topo' ==  zyvar.get('do_list'):
            print (zymake.alltable_topo())
        else:
            print(zymake.help_short())
            if zyvar.get('do_list'):
                print ('Unknow options %s ?' % zyvar.get('do_list'))
        exit()

    if lines is None:
        # catch signal ?
        exit()

    if 'script' in zyvar:
        lines = [' '.join((line_prefix, l)).strip() for l in lines]

    zymake.simulate(halt=False, file=sys.stderr)
    print('\n'.join(lines), file=sys.stdout)

if __name__ == '__main__':
    main()

