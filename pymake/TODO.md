# IN DOII NG
* @pymake related : confilt in -do when invoking script from zymake, solution?! 
    * lookup table for script (function) inside ExpeFormat !!!  (@whoosh related)
    * -l script
* build_corpus and build_networks are identical : wrap in DataBase
* expe_meas script
* ILFM creat sampler and proper inhheritance form GibbsSampler.
* @issue42


@pymake [CMD]
* init
    * if no spec 'print help'
        * pmk init [project]
        -> results
        -> data
        -> spec
        -> script
* update
    * update whoosh, model, corpus etc....

@MODEL Specification:
*  \_initialize  random init !!!! do cleaner init of class for empty model....
* ModelManager constructor from_model (object) (it calls \_load_model)
* print help/signature from command_line (model/corpus)

@Corpus/Dataset : 
* CorpusModules :load from package (sklearn) and disk (
* Whoosh integration !!!
* LDA on my own paper !

* @pymake : different example
    * zymake stats [spec]
    * zymake **search** [spec]
    * zymake check/gen [spec]
    * zymake runcmd [spec] | parallel #parralisation


@format_plot etc
* decorator for tabulate in ExpFormat ?
* plotManager (same scenario, axe, title, **kwargs**)
* OBJECT ! change frontendNetwork to Graph(Object), it is clear ?! (implement morphism in Object for example. (issue42)

@purge: 
* model/lda
* clean and homogeneize, the communities analysis framework. There is redoncdancy, and non consistent call to modularity...
* import models -> import pymake.models ?


@bigdata How should be the backend to make big scale learning with database and search engine interface...
* Spark/Hadoop interface... (hdfs = htable)
* MPI / pymake integration ?


## Expe missing
hyperprior testing
Model evolution with K as well as hyperprior nonparametric model
