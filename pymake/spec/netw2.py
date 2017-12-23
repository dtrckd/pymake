# -*- coding: utf-8 -*-

from pymake import ExpSpace, ExpTensor, Corpus, ExpDesign, ExpGroup

__plot_font_size = 14

### PLOTLIB CONFIG
import matplotlib.pyplot as plt
plt.rc('font', size=__plot_font_size)  # controls default text sizes

class Netw2(ExpDesign):


    data_net_all = Corpus(['manufacturing', 'fb_uc','blogs', 'emaileu', 'propro', 'euroroad', 'generator7', 'generator12', 'generator10', 'generator4'])
    net_all = data_net_all + Corpus(['clique6', 'BA'])

    data_text_all = Corpus(['kos', 'nips12', 'nips', 'reuter50', '20ngroups']) # lucene


    # compare perplexity and rox curve from those baseline.
    compare_scvb = ExpTensor (
        corpus        = ['clique6', 'BA'],
        model         = ['immsb_cgs', 'ilfm_cgs', 'rescal'],
        N             = 'all',
        K             = 6,
        iterations    = 150,
        hyper         = 'auto',
        testset_ratio = 20,
        homo = 0,
        mask = 'unbalanced',

        _data_type    = 'networks',
        _refdir       = 'debug_scvb' ,
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}-{_name}',
        _csv_typo     = '# _iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var'
    )


    scvb = ExpTensor (
        corpus        = ['clique6'],
        model         = 'immsb_scvb',
        N             = 200,
        chunk         = 'adaptative_1',
        K             = 6,
        iterations    = 3,
        hyper         = 'auto',
        testset_ratio = 20,
        #chi = [0.5, 1, 2, 10],
        #tau = [0.5, 1, 2, 16, 64, 256, 1024],
        #kappa = [0.51, 0.45, 1],
        homo = 0,
        mask = 'unbalanced',

        _data_type    = 'networks',
        _refdir       = 'debug_scvb' ,
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}_{chunk}-{_name}',
        _csv_typo     = '# _iteration time_it _entropy _entropy_t _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b'
    )
    cvb = ExpTensor (
        corpus        = ['clique6'],
        model         = 'immsb_cvb',
        N             = 200,
        chunk         = 'adaptative_1',
        K             = 6,
        iterations    = 42,
        hyper         = 'auto',
        testset_ratio = 20,
        #chi = [0.5, 1, 2, 10],
        #tau = [0.5, 1, 2, 16, 64, 256, 1024],
        #kappa = [0.51, 0.45, 1],
        homo = 0,
        mask = 'unbalanced',

        _data_type    = 'networks',
        _refdir       = 'debug_scvb' ,
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}_{chunk}-{_name}',
        _csv_typo     = '# _iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var'
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



    scvb1_validation = ExpTensor ( # i0 experiments.
        corpus        = data_net_all,
        model         = 'immsb_scvb',
        N             = 'all',
        chunk         = ['adaptative_0.33','adaptative_1', 'adaptative_3'], # best seemes to be _1 !
        K             = 6,
        iterations    = [1, 10], # relaunch with 3 to see if any difference. not many ! but some fluctuation...!
        hyper         = 'auto',
        testset_ratio = 25,
        #chi = [0.5, 1, 2, 10],
        #tau = [0.5, 1, 2, 16, 64, 256, 1024],
        #kappa = [0.51, 0.45, 1],
        homo = 0,
        mask = 'unbalanced',

        _data_type    = 'networks',
        _refdir       = 'debug_scvb_1' , # it was done with i0.
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{testset_ratio}_{chunk}-{_name}',
        _csv_typo     = '# _iteration time_it _entropy _entropy_t _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b'
    )

    scvb1_chi_a = ExpTensor (
        corpus        = data_net_all,
        model         = 'immsb_scvb',
        N             = 'all',
        chunk         = 'adaptative_1',
        K             = 6,
        iterations    = 1,
        hyper         = 'auto',
        testset_ratio = 20,
        chi = [0.5, 1, 2, 10],
        tau = [0.5, 1, 2, 16, 64, 256, 1024],
        kappa = [0.51, 0.45, 1],
        #homo = 0,
        #mask = 'unbalanced',

        _data_type    = 'networks',
        _refdir       = 'debug_scvb_chia_i1' ,
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}_{chunk}-{_name}-{_id}',
        _csv_typo     = '# _iteration time_it _entropy _entropy_t _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b'
    )

    scvb1_chi_b = ExpTensor (
        corpus        = data_net_all,
        model         = 'immsb_scvb',
        N             = 'all',
        chunk         = 'adaptative_1',
        K             = 6,
        iterations    = 1,
        hyper         = 'auto',
        testset_ratio = 20,
        chi = [0.5, 1, 2, 10],
        tau = [0.5, 1, 2, 16, 64, 256, 1024],
        kappa = [0.51, 0.45, 1],
        #homo = 0,
        #mask = 'unbalanced',

        _data_type    = 'networks',
        _refdir       = 'debug_scvb_chi_i1' ,
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{mask}_{testset_ratio}_{chunk}-{_name}-{_id}',
        _csv_typo     = '# _iteration time_it _entropy _entropy_t _K _chi_a _tau_a _kappa_a _chi_b _tau_b _kappa_b'
    )

