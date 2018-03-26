from pymake import ExpSpace, ExpTensor, Corpus, ExpDesign, ExpGroup


class Netw2(ExpDesign):


    data_net_all = Corpus(['manufacturing', 'fb_uc','blogs', 'emaileu', 'propro', 'euroroad', 'generator7', 'generator12', 'generator10', 'generator4'])
    net_all = data_net_all + Corpus(['clique6', 'BA'])


    # compare perplexity and rox curve from those baseline.
    compare_scvb = ExpTensor (
        corpus        = ['clique6', 'BA'],
        model         = ['immsb_cgs', 'ilfm_cgs', 'rescal', 'immsb_cvb'],
        N             = 200,
        K             = 6,
        iterations    = 30,
        hyper         = 'auto',
        testset_ratio = 20,
        homo = 0,
        mask = 'unbalanced',

        _data_type    = 'networks',
        _refdir       = 'debug_scvb' ,
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}',
        _csv_typo     = '_iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var'
    )
    compare_scvb_m = ExpGroup(compare_scvb, model=['immsb_cgs', 'immsb_cvb'])
    cvb = ExpGroup(compare_scvb, model='immsb_cvb')

    scvb = ExpTensor (
        corpus        = ['BA'],
        model         = 'immsb_scvb',
        N             = 200,
        chunk         = 'adaptative_1',
        K             = 6,
        iterations    = 3,
        hyper         = 'auto',
        testset_ratio = 20,
        #chi_a = 1,
        #tau_a = 42,
        kappa_a = 0.75,
        #chi_b = 42,
        #tau_b = 300,
        #kappa_b = 0.9,

        homo = 0,
        mask = 'unbalanced',

        _data_type    = 'networks',
        _refdir       = 'debug_scvb' ,
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}_{chunk}',
        _csv_typo     = '_iteration time_it _entropy _entropy_t _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b _elbo _roc'
    )
    scvb_t = ExpGroup(scvb, _refdir='debug_')

    scvb_chi = ExpGroup(scvb, chi_a=1, tau_a=42, kappa_a=[0.6, 0.7, 0.9],
                        chi_b=42, tau_b=300, kappa_b=[0.6, 0.7, 0.9],
                        _format='{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}'
                       )

    scvb_chi2 = ExpGroup(scvb, chi_a=[10], tau_a=[100], kappa_a=[0.6],
                        chi_b=[10], tau_b=[500], kappa_b=[0.9],
                        _format='{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}'
                       )

    scvb_chi_2 = ExpGroup(scvb, chi_a=[1, 10], tau_a=[42, 100, 500], kappa_a=[0.6],
                        chi_b=[1, 10], tau_b=[42, 100, 500], kappa_b=[0.6, 0.7, 0.9],
                        _format='{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}_{chunk}_{chi_a}-{tau_a}-{kappa_a}_{chi_b}-{tau_b}-{kappa_b}'
                       )


    #
    #
    #
    # * iter1 : don"t set masked, and gamma is not symmetric
    #
    # * iter2 : don"t set masked, and gamma is symmetric
    #
    #
    #


    scvb1_chi_a = ExpTensor (
        corpus        = ['blogs', 'manufacturing', 'generator7', 'generator10'],
        model         = 'immsb_scvb',
        N             = 'all',
        chunk         = 'adaptative_1',
        K             = 6,
        iterations    = 1,
        hyper         = 'auto',
        testset_ratio = 20,
        chi_a = [0.5, 1, 2, 10],
        tau_a = [0.5, 1, 2, 16, 64, 256, 1024],
        kappa_a = [0.51, 0.45, 1],
        #homo = 0,
        #mask = 'unbalanced',

        _data_type    = 'networks',
        _refdir       = 'debug_scvb_chia_i1' ,
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}_{chunk}-{_name}-{_id}',
        _csv_typo     = '_iteration time_it _entropy _entropy_t _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b'
    )

    scvb1_chi_b = ExpTensor (
        corpus        = ['blogs', 'manufacturing', 'generator7', 'generator10'],
        model         = 'immsb_scvb',
        N             = 'all',
        chunk         = 'adaptative_1',
        K             = 6,
        iterations    = 1,
        hyper         = 'auto',
        testset_ratio = 20,
        chi_b = [0.5, 1, 2, 10],
        tau_b = [0.5, 1, 2, 16, 64, 256, 1024],
        kappa_b = [0.51, 0.45, 1],
        #homo = 0,
        #mask = 'unbalanced',

        _data_type    = 'networks',
        _refdir       = 'debug_scvb_chib_i1' ,
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}_{chunk}-{_name}-{_id}',
        _csv_typo     = '_iteration time_it _entropy _entropy_t _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b'
    )


    # Noel expe
    noel = ExpGroup([scvb, compare_scvb], N='all', corpus=data_net_all, mask=['balanced', 'unbalanced'], _refdir='noel')

    noel_cvb = ExpGroup(cvb, N='all', corpus=data_net_all, mask=['balanced', 'unbalanced'], _refdir='noel')
    noel_scvb = ExpGroup(scvb, N='all', corpus=data_net_all, mask=['balanced', 'unbalanced'], _refdir='noel')
    noel_scvb_ada = ExpGroup(noel_scvb, chunk=['adaptative_0.1', 'adaptative_0.5', 'adaptative_1', 'adaptative_10'])

    noel_mmsb = ExpGroup([scvb, compare_scvb_m], N='all', corpus=data_net_all,
                    mask=['balanced', 'unbalanced'], _refdir='noel')

    compare_scvb2 = ExpGroup(compare_scvb, N='all', corpus=data_net_all, iterations=100,
                    mask=['balanced', 'unbalanced'], _refdir='noel2')

    noel3 = ExpGroup(scvb_chi2, N='all', chunk=['adaptative_0.1', 'adaptative_1'], corpus=data_net_all, mask=['unbalanced'], _refdir='noel3')


    # cvb debug
    pd = ExpGroup(compare_scvb, iterations=150, model='immsb_cvb', _repeat='debug_cvb',
                  N='all', corpus=data_net_all,
                  mask=['unbalanced'], _refdir='noel2')

    pd2n = ExpGroup(compare_scvb, iterations=150, model='immsb_cvb', _repeat='debug_cvb_2n',
                  N='all', corpus=data_net_all,
                  mask=['unbalanced'], _refdir='noel2')




