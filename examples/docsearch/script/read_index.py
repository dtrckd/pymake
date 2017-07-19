from pymake import ExpeFormat, ModelManager
from pymake.util.utils import colored
import textwrap
from ..model.search_engine import extract_pdf


USAGE = """\
----------------
Search Script :
----------------
"""

class IR(ExpeFormat):
    _default_expe = {'model' : 'docm.tfidf',
                     'highlight' : True,
                     'number_highlight' : 3 ,
                     'number_results' : 20 ,
                    }

    def _preprocess(self):
        if 'N' in self.expe:
            self.expe.number_results = int(self.expe.N)

    def format_results(self, results):
        ''' Show query  search results '''
        expe = self.expe
        startchar, endchar = colored('splitme', 'bold').split('splitme')

        #print('total matched: %d' % (len(results)))

        for rank, hit in enumerate(results):
            shortpath = hit['shortpath']
            fullpath = hit['fullpath']
            score = hit.score

            print('%d: '% (rank+1), colored(shortpath, 'green'), '%.1f' % score)
            if 'title' in hit:
                print('Title: %s: '% (colored(colored(hit['title'], 'bold'), 'red')))
            if expe.highlight:
                text = extract_pdf(fullpath, page_limit=42)
                if text:
                    fragments = hit.highlights('content', text=text, top=expe.number_highlight)
                    fragments = ' '.join(fragments.split())
                    fragments = textwrap.fill(' '*4+fragments, width=84, subsequent_indent=' '*4)
                    print(fragments)
                    print()

    def format_stats(self, model):
        ''' Show index statistics '''
        s = model.get_reader()
        ndoc = s.doc_count()
        print('Number of documents: %d' %(ndoc))
        #'mean /var  size of content'
        # Tree based on path / graph based on citation !
        #'waht field out there and type'
        s.close()


    def search(self, *query):
        expe = self.expe
        _model = ModelManager.from_name(expe)
        model = _model(expe)

        query = ' '.join(query)
        res = model.search(query, limit=expe.number_results)
        self.format_results(res)


    def stats(self):
        expe = self.expe
        _model = ModelManager.from_name(expe)
        model = _model(expe)

        self.format_stats(model)


