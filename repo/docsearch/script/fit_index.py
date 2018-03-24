from pymake import ExpeFormat
from pymake.util.utils import ask_sure_exit


USAGE = """\
----------------
Fit Script :
----------------
"""

class fit(ExpeFormat):

    _default_expe = {'model' : 'docm.tfidf',
                     'path' : '~/Desktop/workInProgress/networkofgraphs/papers',
                     'exclude_path' : ['/figures/', '/fig/', '/img/'],
                     'reset' : False,
                     'extract_structure' : False,
                    }

    def _preprocess(self):
        expe = self.expe
        output_path = self.gramexp.make_output_path(expe)
        expe.path = expe.path if isinstance(expe.path, str) else ' '.join(expe.path)
        return

    def _postprocess(self):
        pass


    def __call__(self):
        expe = self.expe
        model = self.load_model(expe)

        if expe.reset is True:
            ask_sure_exit('reset option will destroy all the indexes, sure [y/n]?')

        try:
            model.fit()
        except KeyboardInterrupt:
            pass
        finally:
            model.close()


