from pymake import ExpSpace, ExpTensor, Corpus, ExpDesign, ExpGroup

class Levy2(ExpDesign):

    _alias = {'ml.iwmmsb_scvb3' : 'WMMSB',
              'ml.iwmmsb_scvb3_auto' : 'WMMSB-bg',
              'ml.immsb_scvb3' : 'MMSB',
              'ml.sbm_gt' : 'SBM',
              'ml.wsbm_gt' : 'WSBM',
              'link-dynamic-simplewiki': 'wiki-link',
              'munmun_digg_reply': 'digg-reply',
              'slashdot-threads': 'slashdot', }

    net_final = Corpus(['fb_uc',
                        'manufacturing',
                        'hep-th',
                        'link-dynamic-simplewiki',
                        'enron',
                        'slashdot-threads',
                        'prosper-loans',
                        'munmun_digg_reply',
                        'moreno_names',
                        'astro-ph'])

    base_graph = dict(
        corpus = 'manufacturing',
        testset_ratio = 20,
        validset_ratio = 10,
        training_ratio = 100,

        # Model global param
        N = 'all',
        K = 10,
        kernel = 'none',

        # plotting
        fig_legend = 4,
        legend_size = 7,
        #ticks_size = 20,
        title_size = 18,
        fig_xaxis = ('time_it', 'time'),

        driver = 'gt', # graph-tool driver
        _write = True,
        _data_type    = 'networks',
        _refdir       = 'aistat_wmmsb',
        _measures     = ['time_it',
                         'entropy@data=valid',
                         'roc@data=test&measure_freq=5',
                         'pr@data=test&measure_freq=5',
                         'wsim@data=test&measure_freq=5',],
        _format       = "{model}-{kernel}-{K}_{corpus}-{training_ratio}"
    )


    wsbm = ExpSpace(base_graph, model = 'sbm_aicher',
                    kernel = 'normal',

                     mu_tol = 0.001,
                     tau_tol = 0.001,
                    )

    wmmsb = ExpSpace(base_graph, model="iwmmsb_scvb3",
                     chunk = 'stratify',
                     delta='auto',
                     sampling_coverage = 0.9, zeros_set_prob=1/2, zeros_set_len=50,
                     tol = 0.001,

                     #fig_xaxis = ('_observed_pt', 'visited edges'),
                    )
    mmsb = ExpSpace(wmmsb, model="immsb_scvb3")



