
# -*- coding: utf-8 -*-

from pymake import ExpSpace, ExpTensor, Corpus, ExpDesign

__plot_font_size = 14

### PLOTLIB CONFIG
import matplotlib.pyplot as plt
plt.rc('font', size=__plot_font_size)  # controls default text sizes

class Netw2(ExpDesign):

    # Use for Name on figure and table
    _mapname = dict((
        ('propro'   , 'Protein')  ,
        ('blogs'    , 'Blogs')    ,
        ('euroroad' , 'Euroroad') ,
        ('emaileu'  , 'Emaileu') ,
        ('manufacturing'  , 'Manufacturing'),
        ('fb_uc'          , 'UC Irvine' ),
        ('generator7'     , 'Network1' ),
        ('generator12'    , 'Network2' ),
        ('generator10'    , 'Network3' ),

        ('generator4'     , 'Network4' ),
        #('generator4'     , 'Network2' ),
        ('pmk.ilfm_cgs'     , 'ILFM' ),
        ('pmk.immsb_cgs'     , 'IMMSB' ),
    ))

    corpus_net_all = Corpus(['manufacturing', 'fb_uc','blogs', 'emaileu', 'propro', 'euroroad', 'generator7', 'generator12', 'generator10', 'generator4'])

    corpus_text_all = Corpus(['kos', 'nips12', 'nips', 'reuter50', '20ngroups']) # lucene


    # compare perplexity and rox curve from those baseline.
    compare_scvb = ExpTensor (
        corpus        = ['clique6', 'BA'],
        model         = ['immsb_cgs', 'ilfm_cgs', 'rescal'],
        N             = 200,
        K             = 6,
        iterations    = 150,
        hyper         = 'auto',
        testset_ratio = 10,

        _data_type    = 'networks',
        _refdir       = 'debug_scvb' ,
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{testset_ratio}',
        _csv_typo     = '# _iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var'
    )

    # Test various gradient steps
    scvb_chi = ExpTensor (
        corpus        = ['clique6', 'BA', 'manufacturing'],
        model         = 'immsb_scvb',
        N             = 200,
        chunk         = 150,
        K             = 6,
        iterations    = 2,
        hyper         = 'auto',
        testset_ratio = 10,
        chi = [0.5, 1, 2, 10],
        tau = [0.5, 1, 2, 16, 64, 256, 1024],
        kappa = [0.51, 0.75, 1],

        _data_type    = 'networks',
        _refdir       = 'debug_scvb' ,
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{testset_ratio}_{chunk}-{chi}-{tau}-{kappa}',
        _csv_typo     = '# _iteration time_it _entropy _entropy_t _K _chi_a _tau_a _kappa_a _chi_a _tau_a _kappa_a'
    )

    scvb_test = Corpus(['propro', 'fb_uc', 'generator12'])
    scvb = ExpTensor (
        corpus        = ['clique6'],
        model         = 'immsb_scvb',
        N             = 'all',
        chunk         = 200,
        K             = 6,
        iterations    = 3,
        hyper         = 'auto',
        testset_ratio = 10,
        #chi = [0.5, 1, 2, 10],
        #tau = [0.5, 1, 2, 16, 64, 256, 1024],
        #kappa = [0.51, 0.45, 1],

        _data_type    = 'networks',
        _refdir       = 'debug_scvb' ,
        _format       = '{corpus}_{model}_{N}_{K}_{iterations}_{hyper}_{homo}_{testset_ratio}_{chunk}',
        _csv_typo     = '# _iteration time_it _entropy _entropy_t _K _chi_a _tau_a _kappa_a _chi_a _tau_a _kappa_a'
    )

