#!/usr/bin/env python
import subprocess
import matplotlib; matplotlib.use('Agg')

_py = 'python'
_py = 'python3'

tests = (
    'zymake',
    'zymake show',
    'zymake -l',
    'zymake -l model',
    'fit -m immsb_cgs',
    'scripts/check_networks pvalue',
    'scripts/check_networks stats',
    'scripts/generate homo',
    'scripts/generate',
)

for t in tests:

    cmdsplit = t.split()
    cmd = _py + ' ' + cmdsplit[0] + '.py ' + ' '.join(cmdsplit[1:])

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    print('Testing :: %s' % cmd)
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
