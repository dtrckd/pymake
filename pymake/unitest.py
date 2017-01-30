#!/usr/bin/env python
import subprocess

_py = 'python3'
_py = 'python'

tests = ('fit',
         'expe_meas',
         'expe_k',
         'check_networks homo',
         'check_networks pvalue',
         'generate',
         'zymake'
        )

for t in tests:

    cmdsplit = t.split()
    cmd = _py + ' ' + cmdsplit[0] + '.py ' + ' '.join(cmdsplit[1:])

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    result = out.split('\n')

    ### Output
    if False:
        for lin in result:
            if not lin.startswith('#'):
                print(lin)

    if p.returncode != 0:
       print("bitcoin failed %d %s %s" % (p.returncode, out, err))

    ### Error
#    print '### exec: %s' % (t)
#    if err:
#        print err
#    else:
#        print '...ok'
