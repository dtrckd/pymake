# -*- coding: utf-8 -*-

from pymake import ExpSpace, ExpTensor, Corpus, ExpDesign

class Netw(ExpDesign):

    _name = 'Networks Design'
    _package = {'model': ['pymake.model', 'mla', 'sklearn.decomposition']}

    # Use for Name on figure and table
    _mapname = dict((
        ('propro'   , 'Protein')  ,
        ('blogs'    , 'Blogs')    ,
        ('euroroad' , 'Euroroad') ,
        ('emaileu'  , 'Emaileu') ,
        ('manufacturing'  , 'Manufacturing'),
        ('fb_uc'          , 'UC Irvine' ),
        ('generator7'     , 'Network 1' ),
        ('generator12'    , 'Network 2' ),
        ('generator10'    , 'Network 3' ),
        ('generator4'     , 'Network 4' ),
        ('pmk.ilfm_cgs'     , 'ILFM' ),
        ('pmk.immsb_cgs'     , 'IMMSB' ),
    ))


    # Networks Data
    CORPUS_REAL_NET = Corpus(['manufacturing', 'fb_uc','blogs', 'emaileu', 'propro', 'euroroad'])

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
    pnas_rest = (CORPUS_REAL_NET + CORPUS_SYN_ICDM) - pnas_short

    # Text Corpus
    # intruder ?
    CORPUS_TEXT_ALL = Corpus(['kos', 'nips12', 'nips', 'reuter50', '20ngroups']) # lucene

    # Tensor Exp
    EXPE_ICDM = ExpTensor((
        ('data_type', ('networks',)),
        ('refdir'  , ('debug10', 'debug11')),
        #('corpus' , ('fb_uc', 'manufacturing')),
        ('corpus' , CORPUS_ALL_ICDM),
        ('model'  , ('immsb_cgs', 'ilfm_cgs')),
        ('K'      , (5,10,15,20)),
        ('N'      , ('all',)),
        ('hyper'  , ('fix', 'auto')),
        ('homo'   , (0,)),
        #('repeat'   , (0, 1, 2,3, 4, 5)),
    ))

    PNAS1 = ExpTensor ((
        ('corpus', CORPUS_ALL_PNAS),
        ('data_type'    , 'networks'),
        ('refdir'        , 'debug111111') , # ign in gen
        #('model'        , 'mmsb_cgs')   ,
        ('model'        , ['immsb_cgs', 'ilfm_cgs'])   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     , # ign in gen
        ('hyper'        , ['auto', 'fix'])    , # ign in gen
        ('homo'         , 0)         , # ign in gen
        ('repeat'      , 1)       ,
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix']),
        ('iterations', '200'),
        ('_format', '{model}_{K}_{hyper}_{homo}_{N}')
    ))

    PNAS2 = ExpTensor ((
        ('corpus', CORPUS_ALL_PNAS),
        ('data_type'    , 'networks'),
        ('refdir'        , 'pnas2') , # ign in gen
        #('model'        , 'mmsb_cgs')   ,
        ('model'        , ['immsb_cgs', 'ilfm_cgs'])   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     , # ign in gen
        ('hyper'        , ['fix','auto'])    , # ign in gen
        ('homo'         , 0)         , # ign in gen
        ('repeat'      , 0)       ,
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix', 'ilfm_cgs.iterations.25', 'immsb_cgs.iterations.150']),
        ('iterations', [25, 150]),
        ('testset_ratio', [40, 60, 80]),
        ('_format', '{model}_{K}_{hyper}_{homo}_{N}_{testset_ratio}')

    ))

    EXPE_ICDM_R = ExpTensor((
        ('data_type', ('networks',)),
        #('corpus' , ('fb_uc', 'manufacturing')),
        ('corpus' , CORPUS_SYN_ICDM),
        #('refdir'  , ('debug10', 'debug11')),
        ('refdir'  , ('debug101010', 'debug111111')),
        ('model'  , ('immsb_cgs', 'ilfm_cgs')),
        ('K'      , (5, 10, 15, 20)),
        ('hyper'  , ('fix', 'auto')),
        ('homo'   , (0, 1, 2)),
        ('N'      , ('all',)),
        ('repeat'   , list(range(10))),
        ('iterations', '200'),
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix']),
    ))

    EXPE_ICDM_R_R = ExpTensor((
        ('data_type', ('networks',)),
        ('corpus' , ('fb_uc', 'manufacturing')),
        ('refdir'  , ('debug101010', 'debug111111')),
        ('model'  , ('immsb_cgs', 'ilfm_cgs')),
        ('K'      , (5, 10, 15, 20)),
        ('hyper'  , ('fix', 'auto')),
        ('homo'   , (0, 1, 2)),
        ('N'      , ('all',)),
        ('repeat'   , list(range(10))),
        ('iterations', '200'),
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix']),
    ))

    # Single Expe

    MODEL_FOR_CLUSTER_IBP = ExpSpace((
        ('data_type'    , 'networks'),
        ('refdir'        , 'debug11') ,
        ('model'        , 'ilfm_cgs')   ,
        ('K'            , 20)        ,
        ('N'            , 'all')     ,
        ('hyper'        , 'fix')     ,
        ('homo'         , 0)         ,
        #('repeat'      , '*')       ,
    ))
    MODEL_FOR_CLUSTER_IMMSB = ExpSpace((
        ('data_type'    , 'networks'),
        ('refdir'        , 'debug11') ,
        ('model'        , 'immsb_cgs')   ,
        ('K'            , 20)        ,
        ('N'            , 'all')     ,
        ('hyper'        , 'auto')     ,
        ('homo'         , 0)         ,
        #('repeat'      , '*')       ,
    ))

    default_gen = ExpTensor ((
        ('corpus', CORPUS_SYN_ICDM),
        ('data_type'    , 'networks'),
        ('refdir'        , 'debug111111') , # ign in gen
        #('model'        , 'mmsb_cgs')   ,
        ('model'        , ['immsb_cgs', 'ilfm_cgs'])   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     , # ign in gen
        ('hyper'        , ['auto', 'fix'])    , # ign in gen
        ('homo'         , 0)         , # ign in gen
        ('repeat'      , 1)       ,
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix']),
        ('alpha', 1),
        ('gmma', 1),
        ('delta', [(1, 5)]),
    ))
    default_check = default_gen.copy()
    default_check['model'] = 'immsb_cgs'

    default_expe = ExpSpace(
        data_type   = 'networks',
        corpus      = 'clique2',
        model       = 'immsb_cgs',
        hyper       = 'auto',
        refdir      = 'debug',
        testset_ratio = 0.2,
        K           = 4,
        N           = 42,
        chunk       = 10000,
        iterations  = 3,
        homo        = 0, # learn W in IBP
    )


    MODELS_GENERATE = ExpTensor ((
        ('data_type'    , 'networks'),
        ('refdir'        , 'debug11') ,
        ('model'        , ['immsb_cgs', 'ilfm_cgs'])   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     ,
        ('hyper'        , ['fix', 'auto'])     ,
        ('homo'         , 0)         ,
        #('repeat'      , '*')       ,
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix']),
    ))


#### Temp

    EXPE_ALL_ICDM_IBP = ExpTensor((
        ('data_type', ('networks',)),
        ('refdir'  , ('debug111111', 'debug101010')),
        ('corpus' , CORPUS_ALL_ICDM),
        ('model'  , ('ilfm_cgs',)),
        ('K'      , (5, 10, 15, 20)),
        ('N'      , ('all',)),
        ('hyper'  , ('fix',)),
        ('homo'   , (0,)),
        ('repeat'   , (6, 7, 8, 9)),
    ))
    EXPE_ALL_ICDM_IMMSB = ExpTensor((
        ('data_type', ('networks',)),
        ('refdir'  , ('debug111111', 'debug101010')),
        ('corpus' , CORPUS_ALL_ICDM),
        ('model'  , ('immsb_cgs',)),
        ('K'      , (5, 10, 15, 20)),
        ('N'      , ('all',)),
        ('hyper'  , ('auto',)),
        ('homo'   , (0,)),
        ('repeat'   , (6, 7, 8, 9)),
    ))



    RUN_DD = ExpTensor((
        ('data_type', ('networks',)),
        #('corpus' , ('fb_uc', 'manufacturing')),
        ('refdir' , ('test_temp',)),
        ('corpus' , ('generator1',)),
        ('model'  , ('immsb_cgs', 'ilfm_cgs')),
        ('K'      , (5,)),
        ('N'      , ('all',)),
        ('hyper'  , ('auto',)),
        ('homo'   , (0,)),
        ('hyper_prior', ('1 2 3 4', '10 2 10 2')),
        ('repeat'   , (0, 1, 2, 4, 5)),
        ('_bind'    , ['immsb_cgs.auto', 'ilfm_cgs.fix']),
    ))

    EXPE_REAL_V2_IBP = ExpTensor((
        ('data_type', ('networks',)),
        ('corpus' , ( 'propro', 'blogs', 'euroroad', 'emaileu')),
        ('refdir'  , ('debug111111'),),
        ('model'  , ( 'ilfm_cgs',)),
        ('K'      , ( 10,)),
        ('hyper'  , ('fix',)),
        ('homo'   , (0,)),
        ('N'      , ('all',)),
        ('repeat'   , list(range(5))),
    ))

    EXPE_REAL_V2_IMMSB = ExpTensor((
        ('data_type', ('networks',)),
        ('corpus' , ( 'propro', 'blogs', 'euroroad', 'emaileu')),
        ('refdir'  , ('debug111111',),),
        ('model'  , ( 'immsb_cgs',)),
        ('K'      , ( 10,)),
        ('hyper'  , ('auto',)),
        ('homo'   , (0,)),
        ('N'      , ('all',)),
        ('repeat'   , list(range(5))),
    ))

    RAGNRK = ExpTensor(
        data_type = ['networks'],
        corpus = ['propro', 'blogs', 'euroroad', 'emaileu'],
        refdir  = ['ragnarok'],
        model  = ['immsb_cgs'],
        K      = [10],
        hyper  = ['auto'],
        homo   = [0],
        N      = [10],
        repeat = list(range(2)),
    )



