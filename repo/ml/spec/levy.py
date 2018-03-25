from pymake import ExpSpace, ExpTensor, Corpus, ExpDesign, ExpGroup


class Levy(ExpDesign):


    data_net_all = Corpus(['manufacturing', 'fb_uc','blogs', 'emaileu', 'propro', 'euroroad', 'generator7', 'generator12', 'generator10', 'generator4'])
    net_all = data_net_all + Corpus(['clique6', 'BA'])

    data_text_all = Corpus(['kos', 'nips12', 'nips', 'reuter50', '20ngroups']) # lucene

    #
    # Poisson Point process :
    # * stationarity / ergodicity of p(d_i) ?
    # * Erny theorem (characterization by void probabilities ?
    # * Inference ? Gamma Process ?
    # * Sparsity ?

    wmmsb = ExpTensor (
        corpus        = ['BA'],
        model         = 'iwmmsb_scvb',
        N             = 200,
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

    warm = ExpTensor(
        corpus        = ['manufacturing.gt'],
        model         = 'iwmmsb_scvb',
        N             = 'all',
        chunk         = 'adaptative_1',
        K             = 6,
        iterations    = 3,
        hyper         = 'auto',
        testset_ratio = 10,

        delta = [[2,2]],

        chi_a=10, tau_a=100, kappa_a=0.6,
        chi_b=10, tau_b=500, kappa_b=0.9,

        homo = 0,
        mask = 'balanced',

        _data_format = 'w',
        _data_type    = 'networks',
        _refdir       = 'debug_scvb' ,
        _format='{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}',
        _csv_typo     = '_iteration time_it _entropy _entropy_t _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b _elbo _roc'
    )


    noelw3 = ExpGroup(wmmsb, N='all', chunk=['adaptative_0.1', 'adaptative_1'], corpus=data_net_all, mask=['unbalanced'], _refdir='noel3')

