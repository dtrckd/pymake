from pymake import ExpeFormat
from pymake.util.utils import colored
import textwrap
from ..model.search_engine import extract_pdf, glob_path

from subprocess import call, Popen


USAGE = """\
----------------
Search Script :
----------------
"""

class IR(ExpeFormat):
    _default_expe = {'model' : 'docm.tfidf',
                     'highlight' : True,
                     'number_highlight' : 3 ,
                     'number_results' : 10 ,
                    }

    def _preprocess(self):
        pass

    def format_results(self, results):
        ''' Show query  search results '''
        expe = self.expe
        startchar, endchar = colored('splitme', 'bold').split('splitme')

        for rank, hit in enumerate(results):
            shortpath = hit['shortpath']
            fullpath = hit['fullpath']
            score = hit.score

            print('%d: '% (rank+1), colored(shortpath, 'green'), '%.2f' % score)
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
        # Tree based on path / graph based on citation !
        #'waht field out there and type'
        s.close()



    def sem(self, query):
        ''' Query semantics. '''

        squery = []
        for q in query:
            esc = q.split()
            if len(esc) > 1:
                squery.append('"'+' '.join(esc)+'"')
            else:
                squery.append(q)


        squery = ' '.join(squery)
        return squery


    def search(self, *query):
        expe = self.expe
        model = self.load_model(expe)

        res = model.search(self.sem(query), limit=expe.number_results)
        self.format_results(res)

        print('total matched: %d' % (model._last_total_match))

    def open(self, query, *hits):
        expe = self.expe
        model = self.load_model(expe)

        query = [query]

        res = model.search(self.sem(query), limit=expe.number_results)

        hits = list(map(lambda x: int(x)-1, hits))

        for rank, _hit in enumerate(res):
            if rank in hits:
                fpath = glob_path(_hit['fullpath'])
                print("opening: %s" % fpath)
                #call(['evince', fpath], timeout=0) # blocking
                Popen(['evince', fpath]) # non-blocking

    def authors(self, *query):
        expe = self.expe
        model = self.load_model(expe)

        res = model.search(self.sem(query), field='authors', limit=expe.number_results)
        self.format_results(res)


    def stats(self):
        expe = self.expe
        model = self.load_model(expe)

        self.format_stats(model)


