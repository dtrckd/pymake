from pymake import ExpeFormat

import os
from subprocess import call




USAGE = """\
----------------
Manage the data : This script is part of the repo/ml of pymake.
----------------
"""



class Process(ExpeFormat):

    _default_expe = { '_expe_silent' : True }


    def killall(self):

        cmd = 'killall -9 -TERM pmk'
        call(cmd.split())
