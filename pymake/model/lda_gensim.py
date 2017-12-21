# -*- coding: utf-8 -*-


# **kwargs !?
def lda_gensim(self, data=None, data_t=None, id2word=None, save=False, model='ldamodel', load=False, updatetype='batch'):

    try:
        sys.path.insert(1, '../../gensim')
        import gensim as gsm
        from gensim.models import ldamodel, ldafullbaye
        Models = {'ldamodel': ldamodel, 'ldafullbaye': ldafullbaye, 'hdp': 1}
    except:
        pass

    fname = self.output_path if self.write else None
    delta = self.hyperparams['delta']
    alpha = 'asymmetric'
    K = self.expe['K']

    data = kwargs['data']
    data_t = kwargs.get('data_t')

    if load:
        return Models[model].LdaModel.load(fname)

    if hasattr(data, 'tocsc'):
        # is csr sparse matrix
        data = data.tocsc()
        data = gsm.matutils.Sparse2Corpus(data, documents_columns=False)
        if heldout_data is not None:
            heldout_data = heldout_data.tocsc()
            heldout_data = gsm.matutils.Sparse2Corpus(heldout_data, documents_columns=False)
    elif isanparray:
        # up tocsc ??!!! no !
        dense2corpus
    # Passes is the iterations for batch onlines and
    # iteration the max it in the gamma treshold test
    # loop Batch setting !
    if updatetype == 'batch':
        lda = Models[model].LdaModel(data, id2word=id2word, num_topics=K, alpha=alpha, eta=delta,
                                     iterations=100, eval_every=None, update_every=None, passes=self.iterations, chunksize=200000,
                                     fname=fname, heldout_corpus=heldout_data)
    elif updatetype == 'online':
        lda = Models[model].LdaModel(data, id2word=id2word, num_topics=K, alpha=alpha, eta=delta,
                                     iterations=100, eval_every=None, update_every=1, passes=1, chunksize=2000,
                                     fname=fname, heldout_corpus=heldout_data)

    if save:
        lda.expElogbeta = None
        lda.sstats = None
        lda.save(fname)
    return lda

