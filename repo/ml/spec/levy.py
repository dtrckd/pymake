from pymake import ExpSpace, ExpTensor, Corpus, ExpDesign, ExpGroup


class Levy(ExpDesign):


    data_net_all = Corpus(['manufacturing', 'fb_uc','blogs', 'emaileu', 'propro', 'euroroad', 'generator7', 'generator12', 'generator10', 'generator4'])
    net_gt = Corpus(['astro-ph', 'hep-th', 'netscience', 'cond-mat']) # all undirected
    #net_gt = Corpus(['astro-ph', 'cond-mat', 'email-Enron', 'hep-th', 'netscience']) # all undirected
    net_all = data_net_all + net_gt
    net_random = Corpus(['clique6', 'BA'])
    net_test = Corpus(['manufacturing', 'fb_uc', 'netscience'])

    net_w = Corpus(['manufacturing', 'fb_uc']) + net_gt

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
        _csv_typo     = '_iteration time_it _entropy _entropy_t _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b _elbo _roc'
    )
    noelw3 = ExpGroup(wmmsb, N='all', chunk=['adaptative_0.1', 'adaptative_1'], corpus=data_net_all, mask=['unbalanced'], _refdir='noel3')



    warm = ExpTensor(
        corpus        = ['manufacturing'],
        model         = 'iwmmsb_scvb3',
        N             = 'all',
        K             = 10,
        hyper         = 'auto',
        homo = 0,
        testset_ratio = 10,


        # Sampling
        chunk         = 'stratify',
        sampling_coverage = 0.42,
        #chi_a=10, tau_a=100, kappa_a=0.6,
        #chi_b=10, tau_b=500, kappa_b=0.9,
        chi_a=1, tau_a=1024, kappa_a=0.5,
        chi_b=1, tau_b=1024, kappa_b=0.5,
        zeros_set_prob = 1/4,
        zeros_set_len = 50,

        #delta = [[1, 1]],
        delta = [[0.5, 10]],

        fig_xaxis = [('_observed_pt', 'visited edges')],
        fig_legend = 4,

        driver = 'gt', # graph-tool driver

        _data_type    = 'networks',
        _refdir       = 'debug_scvb3',
        _format='{corpus}_{model}_{N}_{K}_{hyper}_{homo}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}_{delta}_{zeros_set_len}_{zeros_set_prob}',
        _csv_typo     = '_observed_pt time_it _entropy _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b _roc _wsim',
    )

    # Sparse sampling
    warm_sparse_chunk = ExpGroup(warm, sampling_coverage=1, chunk='sparse')

    # Sampling sensiblity | hyper-delta sensibility
    warm_sampling = ExpGroup(warm, delta=[[1,1],[0.5,10],[10,0.5]],
                             zeros_set_prob = [1/2, 1/3, 1/4],  zeros_set_len=[10, 50])

    arm_sampling = ExpGroup(warm_sampling, delta='auto', model='immsb_scvb3')

    sbm_base = ExpGroup(warm, model=['sbm_gt', 'wsbm_gt', 'rescal_als'],
                        zeros_set_prob=None, zeros_set_len=None, delta=None,
                        _csv_typo='time_it _entropy _K _roc _wsim')
    wsbm_base = ExpGroup(sbm_base, model = ['wsm_g', 'wsbm2_gt'])

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
    eta2_base = ExpGroup(warm, testset_ratio=20, _refdir="eta2", corpus=net_w)
    eta2_sbm = ExpGroup(eta2_base, model=['sbm_gt', 'wsbm_gt', 'rescal_als'],
                        zeros_set_prob=None, zeros_set_len=None, delta=None,
                        _csv_typo='time_it _entropy _K _roc _wsim')
    eta2_wsbm = ExpGroup(eta2_base, model=['wsbm_gt'],
                        zeros_set_prob=None, zeros_set_len=None, delta=None,
                        _csv_typo='time_it _entropy _K _roc _wsim')
    eta2_b = ExpGroup(eta2_base, model="immsb_scvb3", zeros_set_prob=1/2, zeros_set_len=10, delta='auto')

    eta0_w = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=10, delta=[[10, 0.5]])
    eta1_w = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=10, delta=[[1, 1]])
    eta2_w = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=10, delta=[[0.5, 10]])
    eta3_w = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=10, delta=[[0.1, 10]])
    eta0_w50 = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=50, delta=[[10, 0.5]])
    eta1_w50 = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=50, delta=[[1, 1]])
    eta2_w50 = ExpGroup(eta2_base, model="iwmmsb_scvb3", zeros_set_prob=1/2, zeros_set_len=50, delta=[[0.5, 10]])


    eta2_ww = ExpGroup([eta2_w, eta2_wsbm])
    eta2 = ExpGroup([eta2_b, eta2_w]) # repeat 0 1 2 tstratio=20 have 2*n zeros in testset.

    roc_1 = ExpGroup([eta2, sbm_base], _refdir='roc1', corpus=net_w, testset_ratio=[10,20])
    roc_1_noise = ExpGroup([eta2, sbm_base], _refdir='roc1_noise', noise=20, corpus=net_w, testset_ratio=[10,20])
    roc_full = ExpGroup([roc_1, roc_1_noise])

    roc_1_visu_w = ExpGroup(eta2, _refdir='roc1')
    roc_1_visu_sbm = ExpGroup(sbm_base, _refdir='roc1', corpus=net_w, testset_ratio=20, model=['sbm_gt', 'wsbm_gt'])
    roc_1_visu = ExpGroup([roc_1_visu_w, roc_1_visu_sbm])

    roc1_w = ExpGroup([eta0_w, eta1_w, eta2_w, eta0_w50, eta1_w50, eta2_w50], _refdir='roc1')
    roc2_w = ExpGroup([eta2_w, eta3_w], _refdir='roc1', zeros_set_len=[5,10])

