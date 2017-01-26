#!/usr/bin/env python
import subprocess


tests = ('fit',
         'expe_meas',
         'expe_k',
         'check_networks',
         'generate',
         'zymake'
        )

for t in tests:

    cmd = 'python ' + t + '.py'

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
