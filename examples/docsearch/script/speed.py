from pymake import ExpeFormat, ModelManager
from pymake.util.utils import colored
from ..model.search_engine import extract_pdf


USAGE = """\
----------------
Search Script :
----------------
"""

class Speed(ExpeFormat):
    _default_expe = {'model' : 'docm.tfidf',
                     'highlight' : True,
                     'number_highlight' : 3 ,
                     'number_results' : 20 ,
                    }

    def _preprocess(self):
        if 'N' in self.expe:
            self.expe.number_results = int(self.expe.N)



    def search(self, *query):
        expe = self.expe
        _model = ModelManager.from_name(expe)
        model = _model(expe)

        query = ' '.join(query)
        res = model.search(query, limit=expe.number_results)
        self.format_results(res)



# To complete :
#   * count method : time to get the answer and save it with a simple format (take thinhs from blue).
#   * plot method : show speed of search in function of max return results.



