# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict, defaultdict
from pymake import basestring

class _spec_(object):
    """ Global Variable for experiments settgins.
        * Keep in mind that orderedDict here, are important, for tensor result construction
          in order to print table (tabulate) of results.

    """


    """
    =================
    === Mapping  Dictionary
    ================= """

    _trans = dict((
        ('propro'   , 'Protein')  ,
        ('blogs'    , 'Blogs')    ,
        ('euroroad' , 'Euroroad') ,
        ('emaileu'  , 'Emeaileu') ,
        ('manufacturing'  , 'Manufacturing'),
        ('fb_uc'          , 'UC Irvine' ),
        ('generator7'     , 'Network 1' ),
        ('generator12'    , 'Network 2' ),
        ('generator10'    , 'Network 3' ),
        ('generator4'     , 'Network 4' ),
        ('ibp'     , 'ilfm' ),
        ('mmsb'     , 'immsb' ),
    ))
    """
    =================
    === Networks Corpus
    ================= """

    ### Bursty
    CORPUS_BURST_1     = ( 'generator3', 'generator11', 'generator12', 'generator7', 'generator14',)

    ### Non Bursty
    CORPUS_NBURST_1    = ( 'generator4', 'generator5', 'generator6', 'generator9', 'generator10',)

    ### Expe ICDM
    CORPUS_SYN_ICDM_1  = ( 'generator7', 'generator12', 'generator10', 'generator4')
    CORPUS_REAL_ICDM_1 = ( 'manufacturing', 'fb_uc',)

    CORPUS_ALL_3 = CORPUS_SYN_ICDM_1 + CORPUS_REAL_ICDM_1

    CORPUS_REAL_V2 = ('blogs', 'emaileu', 'propro', 'euroroad')

    """
    =================
    === Text Corpus
    ================= """

    CORPUS_TEXT_ALL = ['kos', 'nips12', 'nips', 'reuter50', '20ngroups'],

    """
    =================
    === Expe Spec
    ================= """
    EXPE_ICDM = OrderedDict((
        ('data_type', ('networks',)),
        ('debug'  , ('debug10', 'debug11')),
        #('corpus' , ('fb_uc', 'manufacturing')),
        ('corpus' , CORPUS_ALL_3),
        ('model'  , ('immsb', 'ibp')),
        ('K'      , (5,10,15,20)),
        ('N'      , ('all',)),
        ('hyper'  , ('fix', 'auto')),
        ('homo'   , (0,)),
        #('repeat'   , (0, 1, 2,3, 4, 5)),
    ))

    EXPE_ICDM_R = OrderedDict((
        ('data_type', ('networks',)),
        #('corpus' , ('fb_uc', 'manufacturing')),
        ('corpus' , ('Graph7', 'Graph12', 'Graph10', 'Graph4')),
        #('debug'  , ('debug10', 'debug11')),
        ('debug'  , ('debug101010', 'debug111111')),
        ('model'  , ('immsb', 'ibp')),
        ('K'      , (5, 10, 15, 20)),
        ('hyper'  , ('fix', 'auto')),
        ('homo'   , (0, 1, 2)),
        ('N'      , ('all',)),
        ('repeat'   , list(range(10))),
    ))

    EXPE_ICDM_R_R = OrderedDict((
        ('data_type', ('networks',)),
        ('corpus' , ('fb_uc', 'manufacturing')),
        ('debug'  , ('debug101010', 'debug111111')),
        ('model'  , ('immsb', 'ibp')),
        ('K'      , (5, 10, 15, 20)),
        ('hyper'  , ('fix', 'auto')),
        ('homo'   , (0, 1, 2)),
        ('N'      , ('all',)),
        ('repeat'   , list(range(10))),
    ))


    MODEL_FOR_CLUSTER_IBP = dict ((
        ('data_type'    , 'networks'),
        ('debug'        , 'debug11') ,
        ('model'        , 'ibp')   ,
        ('K'            , 20)        ,
        ('N'            , 'all')     ,
        ('hyper'        , 'fix')     ,
        ('homo'         , 0)         ,
        #('repeat'      , '*')       ,
    ))
    MODEL_FOR_CLUSTER_IMMSB = dict ((
        ('data_type'    , 'networks'),
        ('debug'        , 'debug11') ,
        ('model'        , 'immsb')   ,
        ('K'            , 20)        ,
        ('N'            , 'all')     ,
        ('hyper'        , 'auto')     ,
        ('homo'         , 0)         ,
        #('repeat'      , '*')       ,
    ))

    NETWORKS_DD        = ('generator10', )
    MODELS_DD = [ dict ((
        ('data_type'    , 'networks'),
        ('debug'        , 'debug10') ,
        ('model'        , 'ibp')   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     ,
        ('hyper'        , 'auto')     ,
        ('homo'         , 0)         ,
        #('repeat'      , '*')       ,
    ))]

    MODELS_GENERATE_IBP = [dict ((
        ('data_type'    , 'networks'),
        ('debug'        , 'debug11') ,
        ('model'        , 'ibp')   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     ,
        ('hyper'        , 'fix')     ,
        ('homo'         , 0)         ,
        #('repeat'      , '*')       ,
    ))]
    MODELS_GENERATE_IMMSB = [dict ((
        ('data_type'    , 'networks'),
        ('debug'        , 'debug11') ,
        ('model'        , 'immsb')   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     ,
        ('hyper'        , 'auto')     ,
        ('homo'         , 0)         ,
        #('repeat'      , '*')       ,
    ))]
    MODELS_GENERATE = MODELS_GENERATE_IMMSB +  MODELS_GENERATE_IBP


#### Temp

    EXPE_ALL_3_IBP = dict((
        ('data_type', ('networks',)),
        ('debug'  , ('debug111111', 'debug101010')),
        ('corpus' , CORPUS_ALL_3),
        ('model'  , ('ibp',)),
        ('K'      , (5, 10, 15, 20)),
        ('N'      , ('all',)),
        ('hyper'  , ('fix',)),
        ('homo'   , (0,)),
        ('repeat'   , (6, 7, 8, 9)),
    ))
    EXPE_ALL_3_IMMSB = dict((
        ('data_type', ('networks',)),
        ('debug'  , ('debug111111', 'debug101010')),
        ('corpus' , CORPUS_ALL_3),
        ('model'  , ('immsb',)),
        ('K'      , (5, 10, 15, 20)),
        ('N'      , ('all',)),
        ('hyper'  , ('auto',)),
        ('homo'   , (0,)),
        ('repeat'   , (6, 7, 8, 9)),
    ))



    RUN_DD = dict((
        ('data_type', ('networks',)),
        #('corpus' , ('fb_uc', 'manufacturing')),
        ('debug' , ('test_temp',)),
        ('corpus' , ('generator1',)),
        ('model'  , ('immsb', 'ibp')),
        ('K'      , (5,)),
        ('N'      , ('all',)),
        ('hyper'  , ('auto',)),
        ('homo'   , (0,)),
        ('hyper_prior', ('1 2 3 4', '10 2 10 2')),
        ('repeat'   , (0, 1, 2, 4, 5)),
    ))

    EXPE_REAL_V2_IBP = dict((
        ('data_type', ('networks',)),
        ('corpus' , ( 'propro', 'blogs', 'euroroad', 'emaileu')),
        ('debug'  , ('debug111111'),),
        ('model'  , ( 'ibp',)),
        ('K'      , ( 10,)),
        ('hyper'  , ('fix',)),
        ('homo'   , (0,)),
        ('N'      , ('all',)),
        ('repeat'   , list(range(5))),
    ))

    EXPE_REAL_V2_IMMSB = dict((
        ('data_type', ('networks',)),
        ('corpus' , ( 'propro', 'blogs', 'euroroad', 'emaileu')),
        ('debug'  , ('debug111111',),),
        ('model'  , ( 'immsb',)),
        ('K'      , ( 10,)),
        ('hyper'  , ('auto',)),
        ('homo'   , (0,)),
        ('N'      , ('all',)),
        ('repeat'   , list(range(5))),
    ))

    RAGNRK = dict(
        data_type = ['networks'],
        corpus = ['propro', 'blogs', 'euroroad', 'emaileu'],
        debug  = ['ragnarok'],
        model  = ['immsb'],
        K      = [10],
        hyper  = ['auto'],
        homo   = [0],
        N      = [10],
        repeat = list(range(2)),
    )

    def __init__(self):
        pass

    def repr(self):
        return [d for d in dir(self) if not d.startswith('__')]


    def name(self, l):
        if isinstance(l, (set, list, tuple)):
            return [ self._trans[i] for i in l ]
        elif isinstance(l, basestring):
            try:
                return self._trans[l]
            except:
                return l
        else:
            print (l)
            print (type(l))
            raise NotImplementedError

