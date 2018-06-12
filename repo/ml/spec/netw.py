# -*- coding: utf-8 -*-

from pymake import ExpDesign, ExpSpace, ExpTensor, Corpus, Model, ExpVector

__plot_font_size = 14

### PLOTLIB CONFIG
import matplotlib.pyplot as plt
plt.rc('font', size=__plot_font_size)  # controls default text sizes

class Netw(ExpDesign):

    # Use for Name on figure and table
    _alias = dict((
        ('propro'   , 'Protein')  ,
        ('blogs'    , 'Blogs')    ,
        ('euroroad' , 'Euroroad') ,
        ('emaileu'  , 'Emaileu') ,
        ('manufacturing'  , 'Manufacturing'),
        ('fb_uc'          , 'UC Irvine' ),
        ('generator7'     , 'Network1' ),
        ('generator12'    , 'Network2' ),
        ('generator10'    , 'Network3' ),

        #('generator4'     , 'Network4' ),
        ('generator4'     , 'Network2' ),
        ('ml.ilfm_cgs'     , 'ILFM' ),
        ('ml.immsb_cgs'     , 'IMMSB' ),
    ))


    # Networks Data
    corpus_real_net = Corpus(['manufacturing', 'fb_uc','blogs', 'emaileu', 'propro', 'euroroad'])

    ### Bursty
    CORPUS_BURST_1     = Corpus(['generator3', 'generator11', 'generator12', 'generator7', 'generator14'])

    ### Non Bursty
    CORPUS_NBURST_1    = Corpus(['generator4', 'generator5', 'generator6', 'generator9', 'generator10'])

    CORPUS_SYN_ICDM  = Corpus(['generator7', 'generator12', 'generator10', 'generator4'])
    CORPUS_REAL_ICDM = Corpus(['manufacturing', 'fb_uc',])
    CORPUS_ALL_ICDM = CORPUS_SYN_ICDM + CORPUS_REAL_ICDM
    CORPUS_REAL_PNAS = Corpus(['manufacturing', 'fb_uc','blogs', 'emaileu', 'propro'])
    CORPUS_ALL_PNAS = CORPUS_REAL_PNAS +  CORPUS_SYN_ICDM
    pnas_short = Corpus([ 'blogs', 'manufacturing', 'generator7','generator4'])
    pnas_rest = (corpus_real_net + CORPUS_SYN_ICDM) - pnas_short

    #data_net_all
    #net_all

    # Text Corpus
    # intruder ?
    data_text_all = Corpus(['kos', 'nips12', 'nips', 'reuter50', '20ngroups']) # lucene

    # Tensor Exp
    EXPE_ICDM = ExpTensor((
        ('_data_type', ('networks',)),
        ('_refdir'  , ('debug10', 'debug11')),
        #('corpus' , ('fb_uc', 'manufacturing')),
        ('corpus' , CORPUS_ALL_ICDM),
        ('model'  , ('immsb_cgs', 'ilfm_cgs')),
        ('K'      , (5,10,15,20)),
        ('N'      , ('all',)),
        ('hyper'  , ('fix', 'auto')),
        ('homo'   , (0,)),
        #('_repeat'   , (0, 1, 2,3, 4, 5)),
    ))

    PNAS1 = ExpTensor ((
        ('corpus', CORPUS_ALL_PNAS),
        ('_data_type'    , 'networks'),
        ('_refdir'        , 'debug111111') , # ign in gen
        #('model'        , 'mmsb_cgs')   ,
        ('model'        , ['immsb_cgs', 'ilfm_cgs'])   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     , # ign in gen
        ('hyper'        , ['auto', 'fix'])    , # ign in gen
        ('homo'         , 0)         , # ign in gen
        ('_repeat'      , 1)       ,
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix']),
        ('iterations', '200'),
        ('_format', '{model}_{corpus}_{K}_{hyper}_{homo}_{N}')
    ))

    PNAS2 = ExpTensor ((
        ('corpus', CORPUS_ALL_PNAS),
        ('_data_type'    , 'networks'),
        ('_refdir'        , 'pnas2') , # ign in gen
        #('model'        , 'mmsb_cgs')   ,
        ('model'        , ['immsb_cgs', 'ilfm_cgs'])   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     , # ign in gen
        ('hyper'        , ['fix','auto'])    , # ign in gen
        ('homo'         , 0)         , # ign in gen
        ('_repeat'      , 0)       ,
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix', 'ilfm_cgs.iterations.25', 'immsb_cgs.iterations.150']),
        ('iterations', [25, 150]),
        ('testset_ratio', [40, 60, 80]),
        ('_format', '{model}_{corpus}_{K}_{hyper}_{homo}_{N}_{testset_ratio}'),
        ('_csv_typo', '_iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var'),

    ))

    PNAS3 = ExpTensor ((
        ('corpus', pnas_short),
        ('_refdir'        , 'pnas3') , # ign in gen
        ('_data_type'    , 'networks'),
        ('model'        , ['immsb_cgs', 'ilfm_cgs'])   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     , # ign in gen
        ('hyper'        , ['fix','auto'])    , # ign in gen
        ('homo'         , 0)         , # ign in gen
        ('_repeat'      , 0)       ,
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix', 'ilfm_cgs.iterations.25', 'immsb_cgs.iterations.150']),
        ('iterations', [25, 150]),
        ('testset_ratio', 20),
        ('_format', '{model}_{iterations}_{corpus}_{K}_{hyper}_{homo}_{N}_{testset_ratio}'),
        #('_csv_typo', '_iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var'),

    ))

    EXPE_ICDM_R = ExpTensor((
        ('_data_type', ('networks',)),
        #('corpus' , ('fb_uc', 'manufacturing')),
        ('corpus' , CORPUS_SYN_ICDM),
        #('_refdir'  , ('debug10', 'debug11')),
        ('_refdir'  , ('debug101010', 'debug111111')),
        ('model'  , ('immsb_cgs', 'ilfm_cgs')),
        ('K'      , (5, 10, 15, 20)),
        ('hyper'  , ('fix', 'auto')),
        ('homo'   , (0, 1, 2)),
        ('N'      , ('all',)),
        ('_repeat'   , list(range(10))),
        ('iterations', '200'),
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix']),
    ))

    EXPE_ICDM_R_R = ExpTensor((
        ('_data_type', ('networks',)),
        ('corpus' , ('fb_uc', 'manufacturing')),
        ('_refdir'  , ('debug101010', 'debug111111')),
        ('model'  , ('immsb_cgs', 'ilfm_cgs')),
        ('K'      , (5, 10, 15, 20)),
        ('hyper'  , ('fix', 'auto')),
        ('homo'   , (0, 1, 2)),
        ('N'      , ('all',)),
        ('_repeat'   , list(range(10))),
        ('iterations', '200'),
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix']),
    ))

    # Single Expe

    MODEL_FOR_CLUSTER_IBP = ExpSpace((
        ('_data_type'    , 'networks'),
        ('_refdir'        , 'debug11') ,
        ('model'        , 'ilfm_cgs')   ,
        ('K'            , 20)        ,
        ('N'            , 'all')     ,
        ('hyper'        , 'fix')     ,
        ('homo'         , 0)         ,
        #('_repeat'      , '*')       ,
    ))
    MODEL_FOR_CLUSTER_IMMSB = ExpSpace((
        ('_data_type'    , 'networks'),
        ('_refdir'        , 'debug11') ,
        ('model'        , 'immsb_cgs')   ,
        ('K'            , 20)        ,
        ('N'            , 'all')     ,
        ('hyper'        , 'auto')     ,
        ('homo'         , 0)         ,
        #('_repeat'      , '*')       ,
    ))

    default_gen = ExpTensor ((
        ('corpus', CORPUS_SYN_ICDM),
        ('_data_type'    , 'networks'),
        ('_refdir'        , 'debug111111') , # ign in gen
        #('model'        , 'mmsb_cgs')   ,
        ('model'        , ['immsb_cgs', 'ilfm_cgs'])   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     , # ign in gen
        ('hyper'        , ['auto', 'fix'])    , # ign in gen
        ('homo'         , 0)         , # ign in gen
        ('_repeat'      , 1)       ,
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix']),
        ('alpha', 1),
        ('gmma', 1),
        ('delta', [(1, 5)]),
    ))
    default_check = default_gen.copy()
    default_check['model'] = 'immsb_cgs'


    MODELS_GENERATE = ExpTensor ((
        ('_data_type'    , 'networks'),
        ('_refdir'        , 'debug11') ,
        ('model'        , ['immsb_cgs', 'ilfm_cgs'])   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     ,
        ('hyper'        , ['fix', 'auto'])     ,
        ('homo'         , 0)         ,
        #('_repeat'      , '*')       ,
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix']),
        ('_csv_typo ', ('_iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var',)),
    ))


#### Temp

    EXPE_ALL_ICDM_IBP = ExpTensor((
        ('_data_type', ('networks',)),
        ('_refdir'  , ('debug111111', 'debug101010')),
        ('corpus' , CORPUS_ALL_ICDM),
        ('model'  , ('ilfm_cgs',)),
        ('K'      , (5, 10, 15, 20)),
        ('N'      , ('all',)),
        ('hyper'  , ('fix',)),
        ('homo'   , (0,)),
        ('_repeat'   , (6, 7, 8, 9)),
        ('_csv_typo ', ('_iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var',)),

    ))
    EXPE_ALL_ICDM_IMMSB = ExpTensor((
        ('_data_type', ('networks',)),
        ('_refdir'  , ('debug111111', 'debug101010')),
        ('corpus' , CORPUS_ALL_ICDM),
        ('model'  , ('immsb_cgs',)),
        ('K'      , (5, 10, 15, 20)),
        ('N'      , ('all',)),
        ('hyper'  , ('auto',)),
        ('homo'   , (0,)),
        ('_repeat'   , (6, 7, 8, 9)),
        ('_csv_typo ', ('_iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var',)),
    ))



    RUN_DD = ExpTensor((
        ('_data_type', ('networks',)),
        #('corpus' , ('fb_uc', 'manufacturing')),
        ('_refdir' , ('test_temp',)),
        ('corpus' , ('generator1',)),
        ('model'  , ('immsb_cgs', 'ilfm_cgs')),
        ('K'      , (5,)),
        ('N'      , ('all',)),
        ('hyper'  , ('auto',)),
        ('homo'   , (0,)),
        ('hyper_prior', ('1 2 3 4', '20 2 10 2')),
        ('_repeat'   , (0, 1, 2, 4, 5)),
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix']),
        ('_csv_typo ', ('_iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var',)),
    ))

    EXPE_REAL_V2_IBP = ExpTensor((
        ('_data_type', ('networks',)),
        ('corpus' , ( 'propro', 'blogs', 'euroroad', 'emaileu')),
        ('_refdir'  , ('debug111111'),),
        ('model'  , ( 'ilfm_cgs',)),
        ('K'      , ( 10,)),
        ('hyper'  , ('fix',)),
        ('homo'   , (0,)),
        ('N'      , ('all',)),
        ('_repeat'   , list(range(5))),
        ('_csv_typo ', ('_iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var',)),
    ))

    EXPE_REAL_V2_IMMSB = ExpTensor((
        ('_data_type', ('networks',)),
        ('corpus' , ( 'propro', 'blogs', 'euroroad', 'emaileu')),
        ('_refdir'  , ('debug111111',),),
        ('model'  , ( 'immsb_cgs',)),
        ('K'      , ( 10,)),
        ('hyper'  , ('auto',)),
        ('homo'   , (0,)),
        ('N'      , ('all',)),
        ('_repeat'   , list(range(5))),
        ('_csv_typo ', ('_iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var',)),
    ))

    RAGNRK = ExpTensor(
        _data_type = ['networks'],
        corpus = ['propro', 'blogs', 'euroroad', 'emaileu'],
        _refdir  = ['ragnarok'],
        model  = ['immsb_cgs'],
        K      = [10],
        hyper  = ['auto'],
        homo   = [0],
        N      = [10],
        _repeat = list(range(2)),
        _csv_typo = '_iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var'
    )


    default_expe = ExpTensor(
        _data_type   = 'networks',
        corpus      = 'clique4',
        model       = 'immsb_cgs',
        hyper       = 'auto',
        _refdir      = 'debug',
        testset_ratio = 20,
        K           = 5,
        N           = 50,
        iterations  = 10,
        homo = 0,
        _format = '{model}_{corpus}_{K}_{hyper}_{homo}_{N}',
        #_csv_typo = '_iteration time_it _entropy _entropy_t _K _alpha _gmma alpha_mean delta_mean alpha_var delta_var'
    )

    debug = ExpTensor(
        _data_type   = 'networks',
        corpus      = 'clique4',
        model       = ['immsb_cgs', 'ilfm_cgs', 'immsb_scvb'],
        hyper       = 'auto',
        _refdir      = 'debug',
        testset_ratio = 20,
        K           = 4,
        N           = 42,
        iterations  = 3,
        homo = 0,
        _format = '{model}_{corpus}_{K}_{hyper}_{homo}_{N}',
        _csv_typo = '_iteration time_it _entropy _entropy_t _K'
    )



    model_all =  Model(['immsb_cgs', 'ilfm_cgs', 'immsb_scvb', 'rescal'])
    model_debug =  Model(['mmsb_vb'])



