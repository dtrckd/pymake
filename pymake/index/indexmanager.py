# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os, shutil, logging
from pymake.util.utils import get_global_settings, colored


import whoosh as ws
import whoosh.highlight
from whoosh.qparser import QueryParser


class TerminalFormatter(ws.highlight.Formatter):

    def format_token(self, text, token, replace=False):
        # Use the get_text function to get the text corresponding to the
        # token
        tokentext = ws.highlight.get_text(text, token, replace)

        # Return the text as you want it to appear in the highlighted
        # string
        return "%s" % colored(tokentext, 'bold')


class IndexManager(object):

    _DATA_PATH = get_global_settings('project_data')

    _SCHEMA   = {'model' : ws.fields.Schema(name      = ws.fields.ID(stored = True),
                                            surname   = ws.fields.ID(stored = True),
                                            module    = ws.fields.ID(stored = True),
                                            category  = ws.fields.KEYWORD(stored = True),
                                            content   = ws.fields.TEXT),

                 'script' : ws.fields.Schema(scriptname    = ws.fields.ID(stored = True),
                                             scriptsurname = ws.fields.ID(stored = True),
                                             module   = ws.fields.ID(stored = True),
                                             method   = ws.fields.KEYWORD(stored = True),
                                             signature = ws.fields.TEXT(stored = True),
                                             content  = ws.fields.TEXT),

                }

    log = logging.getLogger('root')

    def __init__(self, default_index='model'):
        self._index_basename = 'ir_index/'
        self._default_index = default_index
        self._ix = {} # Index store by key

    def get_index_path(self, name=None):
        name = name or self._default_index
        return os.path.join(self._DATA_PATH, self._index_basename, name + '/')

    def clean_index(self, name=None, schema=None, **kwargs):
        ''' make the index `name\' according to its `schema\' '''
        name = name or self._default_index
        index_path = self.get_index_path(name)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        os.makedirs(index_path)

        if name in self._SCHEMA:
            schema = self._SCHEMA[name]
        else:
            raise NotImplementedError('Dont know what to do, no schema defined ...?')

        self._ix[name] = ws.index.create_in(index_path, schema)

        return self._ix[name]

    def load_index(self, name=None):
        name = name or self._default_index
        index_path = self.get_index_path(name)
        return ws.index.open_dir(index_path)

    def get_index(self, name=None):
        name = name or self._default_index
        if name in self._ix:
            return self._ix[name]
        elif os.path.exists(self.get_index_path(name)):
           return  self.load_index(name)
        else:
            return self.clean_index(name)

    def get_writer(self, reset=False, online=None):
        if reset:
            ix = self.clean_index()
        else:
            ix = self.get_index()

        if online:
            import whoosh.writing
            if online is True:
                online = {}
            period = online.get('period', 600)
            limit = online.get('limit', 2)
            return ws.writing.BufferedWriter(ix, period=period, limit=limit)
        else:
            return ix.writer()

    @classmethod
    def build_indexes(cls):
        ''' Update the system index '''
        idx = cls()
        for name  in idx._SCHEMA:
            func = 'update_' + name + '_index'
            builder = getattr(idx, func)
            builder()


    def update_corpus_index(self):
        raise NotImplementedError

    def update_script_index(self):
        ''' Update the schema of the Scripts index '''
        from pymake.frontend.frontend_io import ScriptsLoader
        model = 'script'
        self.log.info('Building %s index...' % model)
        Scripts = ScriptsLoader.get_packages()
        method_by_cls = ScriptsLoader.get_atoms()
        writer = self.clean_index(model).writer()
        for scriptname, module in Scripts.items():
            self.log.info('\tindexing %s' % (str(scriptname)+str(module)))

            # Loop is context/model dependant
            methods = method_by_cls[module.__name__]
            for method in methods:

                content = ''
                writer.add_document(scriptname = module.__name__,
                                    scriptsurname = scriptname,
                                    module = module.__module__,
                                    method = method,
                                    content = content)
        writer.commit()

    def update_model_index(self):
        ''' Update the schema of the Models index '''
        from pymake.frontend.frontend_io import ModelsLoader
        model = 'model'
        self.log.info('Building %s index...' % model)
        models = ModelsLoader.get_atoms()
        writer = self.clean_index(model).writer()
        for surname, module in models.items():
            self.log.info('\tindexing %s' % (str(surname)+str(module)))

            # Loop is context/model dependant
            topos = ' '.join(set(module.__module__.split('.')[1:]))
            content = ' '.join((surname, module.__name__, module.__module__))

            writer.add_document(surname = surname,
                                name = module.__name__,
                                category = topos,
                                module = module.__module__,
                                content = content)
        writer.commit()


    # @debug : online searcher
    def _search(self, query='',  field=None,  index=None, terms=False, limit=None):
        ''' query (exaxct mathch) search '''
        index = index or self._default_index
        ix = self.get_index(index)
        fieldin = field or 'content'

        qp = QueryParser(fieldin, ix.schema)
        qp.add_plugin(ws.qparser.SingleQuotePlugin())
        query = qp.parse(query, normalize=False)
        with ix.searcher() as searcher:
            if terms is True:
                results = searcher.search(query, terms=True, limit=limit).matched_terms()
            else:
                results = list(searcher.search(query, limit=limit).items())

        return results

    def search(self, query='',  field=None,  index=None, limit=None):
        ''' OR search '''
        index = index or self._default_index
        limit = None if limit == 'all' else limit
        ix = self.get_index(index)
        fieldin = field or 'content'

        qp = QueryParser(fieldin, ix.schema)
        qp.add_plugin(ws.qparser.SingleQuotePlugin())
        qp.add_plugin(ws.qparser.WildcardPlugin())
        query = qp.parse(query)
        with ix.searcher() as searcher:
                results = searcher.search(query, limit=limit)

                results.fragmenter = ws.highlight.SentenceFragmenter(maxchars=200, charlimit=100042)
                #results.fragmenter = ws.highlight.ContextFragmenter(maxchars=200, surround=43)
                results.formatter = TerminalFormatter()

                for r in results:
                    yield r


    def getbydocid(self, docid, index=None):
        ''' return the a document's stored fields in the index from docid '''
        index = index or self._default_index
        ix = self.get_index(index)
        with ix.searcher() as searcher:
            doc = searcher.stored_fields(docid)
        return doc

    def getfirst(self, query='', field=None, index=None):
        query = "'" + query + "'"
        results = self._search(query, field, index, limit=1)

        if not results:
            return None
        else:
            return self.getbydocid(results[0][0])

    def getbydocids(self, docids, index=None):
        ''' return the a list of document's stored fields in the index from docids '''
        index = index or self._default_index
        ix = self.get_index(index)
        docs = []
        with ix.searcher() as searcher:
            for docid in docids:
                docs.append(searcher.stored_fields(docid))
        return docs

    # @debug : online searcher
    def query(self, field=None, index=None, terms=False):
        index = index or self._default_index
        ix = self.get_index(index)
        field = field or ix.schema.stored_names()[0]

        query = ws.query.Every(field)
        with ix.searcher() as searcher:
            results = searcher.search(query, limit=None)
            if terms is False:
                results = [r[field] for r in results]
            else:
                results = [dict(r) for r in results]

        return results


if __name__ == '__main__':
    ix = IndexManager()
    #ix.update_model_index()
    q = ix.query(field='surname')
    for r in q:
        print(r)
