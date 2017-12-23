from pymake import  ExpeFormat
from pymake.frontend.manager import ModelManager, FrontendManager


import matplotlib.pyplot as plt
from pymake.util.utils import colored, ask_sure_exit
from pymake.core.format import tabulate

class Corpus(ExpeFormat):

    _default_expe = { '_expe_silent' : True,
                     '_ignore_format_unique': True,
                    }

    def _preprocess(self):
        #ask_sure_exit('Sure to overwrite %s ?' % (self.expe['_do']))
        pass


    def build_net(self):

        frontend = FrontendManager.load(self.expe)
        prop = frontend.get_data_prop()
        msg = frontend.template(prop)
        print (msg)

    def build_text(self):
        pass


