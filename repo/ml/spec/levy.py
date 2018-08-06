from pymake import ExpSpace, ExpTensor, Corpus, ExpDesign, ExpGroup


class Levy(ExpDesign):

    #refdir=gapv1 is old likelihood + sampling r/p.sampling
    #rocw1: new likelihood + sampling r/p
    #rocw2: newlikelihood + mixed samling/VB r/p
    #rocw3: new likelihood + mixed samling/VB r/p and N_Phi added. (partial some correction and inversion sample/expectation)
    #rocw4: new likelihood + samling r/p and

    _alias = {'ml.iwmmsb_scvb3' : 'WMMSB',
              'ml.iwmmsb_scvb3_auto' : 'WMMSB-bg',
              'ml.immsb_scvb3' : 'MMSB',
              'ml.sbm_gt' : 'SBM',
              'ml.wsbm_gt' : 'WSBM',
              'link-dynamic-simplewiki': 'wiki-link',
              'munmun_digg_reply': 'digg-reply',
              'slashdot-threads': 'slashdot',
             }

    net_old = Corpus(['manufacturing', 'fb_uc','blogs', 'emaileu', 'propro', 'euroroad', 'generator7', 'generator12', 'generator10', 'generator4'])
    net_gt = Corpus(['astro-ph', 'hep-th', 'netscience', 'cond-mat']) # all undirected
    net_random = Corpus(['clique6', 'BA'])
    net_test = Corpus(['manufacturing', 'fb_uc', 'netscience'])
    net_large = Corpus(['link-dynamic-simplewiki', 'enron', 'foldoc'])
    net_w = Corpus(['manufacturing', 'fb_uc']) + net_gt
    # 'actor-collaboration' # Too big?
    net_w2 = Corpus(['slashdot-threads','prosper-loans', 'munmun_digg_reply', 'moreno_names' ])
    # manufacturing
    net_final = Corpus(['fb_uc', 'manufacturing','hep-th', 'link-dynamic-simplewiki', 'enron', 'slashdot-threads', 'prosper-loans', 'munmun_digg_reply', 'moreno_names', 'astro-ph'])
    net_all = net_old + net_gt + net_large

    #
    # Poisson Point process :
    # * stationarity / ergodicity of p(d_i) ?
    # * Erny theorem (characterization by void probabilities ?
    # * Inference ? Gamma Process ?
    # * Sparsity ?

    wmmsb = ExpTensor (
        corpus        = ['BA'],
        model         = 'iwmmsb_scvb',
        N             = 'all',
        chunk         = 'adaptative_1',
        K             = 6,
        iterations    = 3,
        hyper         = 'auto',
        testset_ratio = 20,

        delta = [[2,2]],

        chi_a=10, tau_a=100, kappa_a=0.6,
        chi_b=10, tau_b=500, kappa_b=0.9,

        homo = 0,
        mask = 'unbalanced',

        _data_format = 'w',
        _data_type    = 'networks',
        _refdir       = 'debug_scvb' ,
        _format='{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}',
        _csv_typo     = '_iteration time_it _entropy _entropy_t _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b _elbo _roc _pr'
    )
    noelw3 = ExpGroup(wmmsb, N='all', chunk=['adaptative_0.1', 'adaptative_1'], corpus=net_old, mask=['unbalanced'], _refdir='noel3')



    warm = ExpTensor(
        corpus        = ['manufacturing'],
        model         = 'iwmmsb_scvb3',
        N             = 'all',
        K             = 10,
        hyper         = 'auto',
        homo = 0,
        testset_ratio = 20,
        validset_ratio = 10,


        # Sampling
        chunk         = 'stratify',
        sampling_coverage = 0.42,
        #chi_a=10, tau_a=100, kappa_a=0.6,
        #chi_b=10, tau_b=500, kappa_b=0.9,
        chi_a=1, tau_a=1024, kappa_a=0.5,
        chi_b=1, tau_b=1024, kappa_b=0.5,
        zeros_set_prob = 1/2,
        zeros_set_len = 50,

        #delta = [[1, 1]],
        #delta = [[0.5, 10]],
        delta = 'auto',

        fig_xaxis = [('_observed_pt', 'visited edges')],
        fig_legend = 4,
        legend_size = 12,
        #ticks_size = 20,
        title_size = 20,

        driver = 'gt', # graph-tool driver

        _data_type    = 'networks',
        _refdir       = 'debug_scvb3',
        _format='{corpus}_{model}_{N}_{K}_{hyper}_{homo}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}_{delta}_{zeros_set_len}_{zeros_set_prob}',
        _csv_typo     = '_observed_pt time_it _entropy _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b _roc _wsim _pr',
    )

    # Sparse sampling
    warm_sparse_chunk = ExpGroup(warm, sampling_coverage=1, chunk='sparse')

    # Sampling sensiblity | hyper-delta sensibility
    warm_sampling = ExpGroup(warm, delta=[[1,1],[0.5,10],[10,0.5]],
                             zeros_set_prob = [1/2, 1/3, 1/4],  zeros_set_len=[10, 50])

    arm_sampling = ExpGroup(warm_sampling, delta='auto', model='immsb_scvb3')

    sbm_base = ExpGroup(warm, model=['sbm_gt', 'wsbm_gt', 'rescal_als'],
                        zeros_set_prob=None, zeros_set_len=None, delta=None,
                        _csv_typo='time_it _entropy _K _roc _wsim _pr')
    wsbm2_base = ExpGroup(sbm_base, model = ['wsm_g', 'wsbm2_gt'])
    wsbm_base = ExpGroup(sbm_base, model = ['wsbm_gt'])

    # Compare sensibility
    eta_b = ExpGroup([arm_sampling], _refdir='eta', corpus=net_w,
                   zeros_set_prob=[1/2, 1/4])
    eta_w = ExpGroup([warm_sampling], _refdir='eta', corpus=net_w,
                   zeros_set_prob=[1/2, 1/4])
    eta = ExpGroup([eta_b, eta_w])

    eta_sbm = ExpGroup(sbm_base, _refdir='eta', corpus=net_w)


    # test/visu
    warm_visu = ExpGroup(warm, delta=[[1,1],'auto'],
                         zeros_set_prob = [1/2],  zeros_set_len=[10])

    # Best selection visu (eta)
    best_mmsb = ExpGroup(eta_b, zeros_set_prob=1/4, zeros_set_len=50, delta='auto')
    best_wmmsb = ExpGroup(eta_w, zeros_set_prob=1/4, zeros_set_len=50, delta=[[0.5,10]])
    best_scvb = ExpGroup([best_mmsb, best_wmmsb])
    eta_best = ExpGroup([best_scvb, eta_sbm])


    # Corrected zero sampling
    eta2_base = ExpGroup(warm, testset_ratio=20, _refdir='roc5', corpus=net_w) # roc5§roc5_N
    eta2_sbm = ExpGroup(eta2_base, model=['sbm_gt', 'wsbm_gt', 'rescal_als'],
                        zeros_set_prob=None, zeros_set_len=None, delta=None,
                        _csv_typo='time_it _entropy _K _roc _wsim _pr')

    eta2_b = ExpGroup(eta2_base, model="immsb_scvb3", zeros_set_prob=1/2, zeros_set_len=10, delta='auto')
    eta2_b10 = eta2_b
    eta2_b50 = ExpGroup(eta2_base, model="immsb_scvb3", zeros_set_prob=1/2, zeros_set_len=50, delta='auto')

    eta0_w = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=10, delta=[[10, 0.5]])
    eta1_w = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=10, delta=[[1, 1]])
    eta2_w = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=10, delta=[[0.5, 10]])
    eta3_w = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=10, delta=[[0.1, 10]])
    eta0_w50 = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=50, delta=[[10, 0.5]])
    eta1_w50 = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=50, delta=[[1, 1]])
    eta2_w50 = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=50, delta=[[0.5, 10]])

    eta2a_w = ExpGroup(eta2_base, model="iwmmsb_scvb3_auto", zeros_set_prob=1/2, zeros_set_len=10, delta='auto', _model="ml.iwmmsb_scvb3")
    eta2a_w10 = eta2a_w
    eta2a_w50 = ExpGroup(eta2_base, model="iwmmsb_scvb3_auto", zeros_set_prob=1/2, zeros_set_len=50, delta='auto', _model="ml.iwmmsb_scvb3")

    eta4_full = ExpGroup([eta2_b50, eta2_w50, eta2a_w50]) # weighte are squared

    roc_visu_sbm = ExpGroup(sbm_base, corpus=net_w, testset_ratio=20, model=['sbm_gt', 'wsbm_gt'], _refdir='roc5')
    roc_visu_sbm_full = ExpGroup(sbm_base, corpus=net_w, testset_ratio=20, model=['rescal_als', 'sbm_gt', 'wsbm_gt'], _refdir='roc5')
    roc_visu_full = ExpGroup([eta4_full, roc_visu_sbm_full])
    roc_visu_final10 = ExpGroup([eta2_b, eta2_w, eta2a_w, roc_visu_sbm])
    roc_visu_final50 = ExpGroup([eta2_b50, eta2_w50, eta2a_w50, roc_visu_sbm])
    roc_visu_final = roc_visu_final50


    roc_visu_final2 = ExpGroup([eta2_b50, eta2_w50, eta2a_w50], _refdir="roc5v")
    roc_visu_final2 = ExpGroup([roc_visu_final2, roc_visu_sbm])
    roc_visu_final2_full = ExpGroup([roc_visu_final2, roc_visu_sbm_full])

    roc_visu_final3 = ExpGroup([eta2_b50, eta1_w50, eta2a_w50], _refdir="roc5v")
    roc_visu_final3 = ExpGroup([roc_visu_final3, roc_visu_sbm])

    online_roc = ExpGroup(roc_visu_final3 , training_ratio=[1, 5,10,20,30, 50, 100], _refdir='online1w', corpus=net_final,
                          _seed='corpus', testset_ratio=20,
                          _format='{corpus}_{model}_{N}_{K}_{hyper}_{homo}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}_{delta}_{zeros_set_len}_{zeros_set_prob}--{training_ratio}',
                         )

    online_roc_ = ExpGroup([eta2_b50, eta2a_w50, roc_visu_sbm] , training_ratio=[1, 5,10,20,30,50,100], _refdir='online1w', corpus=net_final,
                          _seed='corpus', testset_ratio=20,
                          _format='{corpus}_{model}_{N}_{K}_{hyper}_{homo}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}_{delta}_{zeros_set_len}_{zeros_set_prob}--{training_ratio}',
                         )


    online_roc_sbm = ExpGroup(roc_visu_sbm , training_ratio=[1, 5,10,20,30,50,100], _refdir='online1w', corpus=net_final,
                          _seed='corpus', testset_ratio=20,
                          _format='{corpus}_{model}_{N}_{K}_{hyper}_{homo}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}_{delta}_{zeros_set_len}_{zeros_set_prob}--{training_ratio}',
                         )

    online_roc_mmsb = ExpGroup([eta2_b50, eta1_w50, eta2a_w50] , training_ratio=[1, 5,10,20,30,50,100], _refdir='online1w', corpus=net_final,
                          _seed='corpus', testset_ratio=20,
                          _format='{corpus}_{model}_{N}_{K}_{hyper}_{homo}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}_{delta}_{zeros_set_len}_{zeros_set_prob}--{training_ratio}',
                         )

    online_roc_w = ExpGroup([eta1_w50, eta2a_w50, wsbm_base] , training_ratio=[1, 5,10,20,30,50,100], _refdir='online1w', corpus=net_final,
                          _seed='corpus', testset_ratio=20,
                          _format='{corpus}_{model}_{N}_{K}_{hyper}_{homo}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}_{delta}_{zeros_set_len}_{zeros_set_prob}--{training_ratio}',
                         )

    online_roc_wm = ExpGroup([eta1_w50, eta2a_w50] , training_ratio=[1, 5,10,20,30,50,100], _refdir='online1w', corpus=net_final,
                          _seed='corpus', testset_ratio=20,
                          _format='{corpus}_{model}_{N}_{K}_{hyper}_{homo}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}_{delta}_{zeros_set_len}_{zeros_set_prob}--{training_ratio}',
                            )

    online_roc_wsbm = ExpGroup([wsbm_base] , training_ratio=[1, 5,10,20,30,50,100], _refdir='online1w', corpus=net_final,
                          _seed='corpus', testset_ratio=20,
                          _format='{corpus}_{model}_{N}_{K}_{hyper}_{homo}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}_{delta}_{zeros_set_len}_{zeros_set_prob}--{training_ratio}',
                         )

    roc_w = ExpGroup([eta0_w, eta1_w, eta2_w, eta0_w50, eta1_w50, eta2_w50]) # squared weight
    roc_b = ExpGroup(eta2_base, model="immsb_scvb3", zeros_set_prob=1/2, zeros_set_len=[10, 50], delta='auto')

    conv_w = ExpGroup([eta0_w50, eta1_w50, eta2_w50, eta2a_w50], corpus=['astro-ph', 'enron', 'munmun_digg_reply'])

    param1 = ExpGroup(eta2a_w50,  c0=[0.5, 1,10, 100], r0=[0.1, 0.5, 1])
    gap = ExpGroup([param1], _refdir='gap_hyper', testset_ratio=20, corpus=net_w,
                   ce=[100], eps=[1e-6],
                   _format='{corpus}_{model}_{N}_{K}_{hyper}_{homo}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}_{delta}_{zeros_set_len}_{zeros_set_prob}-{c0}-{r0}-{ce}-{eps}',
                  )


    gap_visu = ExpGroup([eta2a_w50], _refdir='gap_hyper', testset_ratio=20, corpus=net_w,
                    ce=[1, 10, 100], eps=[1e-5, 1e-6, 1e-7],
                    c0=10, r0=1,
                    _format='{corpus}_{model}_{N}_{K}_{hyper}_{homo}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}_{delta}_{zeros_set_len}_{zeros_set_prob}-{c0}-{r0}-{ce}-{eps}',
                  )


