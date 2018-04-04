from pymake import ExpSpace, ExpTensor, Corpus, ExpDesign, ExpGroup


class Levy(ExpDesign):


    data_net_all = Corpus(['manufacturing', 'fb_uc','blogs', 'emaileu', 'propro', 'euroroad', 'generator7', 'generator12', 'generator10', 'generator4'])
    net_gt = Corpus(['astro-ph', 'cond-mat', 'hep-th', 'netscience']) # all undirected
    #net_gt = Corpus(['astro-ph', 'cond-mat', 'email-Enron', 'hep-th', 'netscience']) # all undirected
    net_all = data_net_all + net_gt
    net_random = Corpus(['clique6', 'BA'])

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
        zeros_set_prob = 1/2,
        zeros_set_len = 50,

        delta = [[1, 1]],
        #delta = [[0.5, 10]],

        fig_xaxis = [('_observed_pt', 'visited edges')],
        fig_legend = 4,

        driver = 'gt', # graph-tool driver

        _data_type    = 'networks',
        _refdir       = 'debug_scvb3',
        _format='{corpus}_{model}_{N}_{K}_{hyper}_{homo}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}_{delta}_{zeros_set_len}_{zeros_set_prob}',
        _csv_typo     = '_observed_pt time_it _entropy _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b'
    )
    # Full measure
    warm_debug = ExpGroup(warm, _csv_typo='_observed_pt time_it _entropy _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b _roc _wsim')

    warm_sparse_chunk = ExpGroup(warm_debug, sampling_coverage=1, chunk='sparse')

    # Sampling sensiblity | hyper-delta sensibility
    warm_sampling = ExpGroup(warm, delta=[[1,1],[0.5,10],[10,0.5]],
                             zeros_set_prob = [1/2, 1/3, 1/4],  zeros_set_len=[10, 50])

    warm_sampling_d = ExpGroup(warm_sampling, _csv_typo='_observed_pt time_it _entropy _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b _roc _wsim')
    arm_sampling_d = ExpGroup(warm_sampling, delta='auto', model='immsb_scvb3',
                              _csv_typo='_observed_pt time_it _entropy _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b _roc _wsim')

    eta_b = ExpGroup([arm_sampling_d], _refdir='eta', corpus=net_w,
                   zeros_set_prob=[1/2, 1/4])
    eta_w = ExpGroup([warm_sampling_d], _refdir='eta', corpus=net_w,
                   zeros_set_prob=[1/2, 1/4])
    eta = ExpGroup([eta_b, eta_w])

    eta_sbm = ExpGroup(warm, _refdir='eta', corpus=net_w, model=['sbm_gt', 'wsbm_gt', 'wsbm2_gt', 'rescal_als'],
                       zeros_set_prob=None, zeros_set_len=None, delta=None,
                       _csv_typo='time_it _entropy _K _roc _wsim')



    warm_visu = ExpGroup(warm_debug, delta=[[1,1],'auto'],
                         zeros_set_prob = [1/2],  zeros_set_len=[10])

    # Best selection visu
    best_mmsb = ExpGroup(eta_b, zeros_set_prob=1/4, zeros_set_len=50, delta='auto')
    best_wmmsb = ExpGroup(eta_w, zeros_set_prob=1/4, zeros_set_len=50, delta=[[0.5,10]])
    best_scvb = ExpGroup([best_mmsb, best_wmmsb])
    eta_best = ExpGroup([best_scvb, eta_sbm])


