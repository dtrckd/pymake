from pymake import ExpSpace, ExpTensor, ExpGroup, ExpDesign


class Wmmsb_Aistats(ExpDesign):

  _alias = {'ml.iwmmsb_scvb3' : 'WMMSB',
            'ml.iwmmsb_scvb3_auto' : 'WMMSB-bg',
            'ml.immsb_scvb3' : 'MMSB',
            'ml.sbm_gt' : 'SBM',
            'ml.wsbm_gt' : 'WSBM',
            'link-dynamic-simplewiki': 'wiki-link',
            'munmun_digg_reply': 'digg-reply',
            'slashdot-threads': 'slashdot', }

  corpus = ['fb_uc',
            'manufacturing',
            'hep-th',
            'link-dynamic-simplewiki',
            'enron',
            'slashdot-threads',
            'prosper-loans',
            'munmun_digg_reply',
            'moreno_names',
            'astro-ph' ]

  _base_graph = ExpTensor(
    driver = 'gt', # graph-tool driver
    corpus = corpus,  # data
    testset_ratio = 20, # percentage of the testset ratio
    validset_ratio = 10, # percentage of the validset ratio
    training_ratio=[1, 5,10,20,30, 50, 100],  # subsample the edges

    # Model global param
    N = 'all', # keep all vertices
    K = 10,    # number of latent classes
    kernel = 'none',

    # plotting
    fig_legend = 4,
    legend_size = 12,
    title_size = 20,
    fig_xaxis = [('time_it', 'time')],

    _seed = 'corpus', # consistent choice of edges across models when subsampling the corpus
    _write = True,
    _data_type    = 'networks',
    _refdir       = 'aistat_wmmsb',
    # Each word indicateds a measure that is done during the model inference
    _measures     = ['time_it',
                     'entropy@data=valid',
                     'roc@data=test&measure_freq=5',
                     'pr@data=test&measure_freq=5',
                     'wsim@data=test&measure_freq=5',],
    _format       = "{model}-{kernel}-{K}_{corpus}-{training_ratio}"
)

  _sbm_peixoto = ExpTensor(_base_graph, model='sbm_gt')
  _rescal = ExpTensor(_base_graph, model='rescal_als')
  _wsbm = ExpTensor(_base_graph, model='sbm_aicher',
                  kernel = ['bernoulli', 'normal', 'poisson'],
                 mu_tol = 0.001,
                 tau_tol = 0.001,
                  )
  _wmmsb = ExpTensor(_base_graph, model="iwmmsb_scvb3",
                   chunk='stratify',
                   delta='auto',
                   sampling_coverage = 0.9, zeros_set_prob=1/2, zeros_set_len=50,
                   tol = 0.001,
                  )
  _mmsb = ExpTensor(_wmmsb, model="immsb_scvb3")

  final_design = ExpGroup([_wmmsb, _mmsb, _wsbm, _sbm_peixoto])
