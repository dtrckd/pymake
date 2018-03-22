#!/usr/bin/env python3
import subprocess
import os

os.chdir('../repo/ml')

tests = [
    'pmk',
    'pmk update',
    'pmk show',
    'pmk -l',
    'pmk -ll',
    'pmk -l spec',
    'pmk -l model',
    'pmk -l script',
    'pmk -l model',
    'pmk default_expe',
    'pmk path default_expe',
    'pmk default_expe -x fit -w',
    'pmk default_expe -x plot',
    'pmk default_expe -x plot fig corpus:_entropy',
    'pmk hist',
]

n_errors = 0

for test in tests:

    cmd = test

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    print('Testing :: %s' % cmd)
    out, err = p.communicate()
    result = str(out).split('\n')

    ### Output
    if False:
        for lin in result:
            if not lin.startswith('#'):
                print(lin)

    if p.returncode != 0:
        n_errors += 1
        print("test failed: %d,  %s" % (p.returncode, err))

print('Test Sucess: %d / %d' % (len(tests)-n_errors, len(tests)))

