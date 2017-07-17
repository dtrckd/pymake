from pymake import ExpeFormat, ModelManager


USAGE = """\
----------------
Fit Script :
----------------
"""

class fit(ExpeFormat):

    _default_expe = {'model' : 'docm.tfidf',
                     'path' : '~/Desktop/workInProgress/networkofgraphs/papers',
                     'reset' : False,
                     'extract_structure' : True,
                    }

    def _preprocess(self):
        expe = self.expe
        output_path = self.gramexp.make_output_path(expe)
        return

    def _postprocess(self):
        pass


    def __call__(self):
        expe = self.expe
        _model = ModelManager.from_name(expe)
        model = _model(expe)

        try:
            model.fit()
        except KeyboardInterrupt:
            pass
        finally:
            model.close()


