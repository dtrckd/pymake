# Code Factoring

MODEL Specifivation:
*  \_initialize  random init !!!!  cleanerinit of class for empty model....
* ModelManager constructor from_model (object) (it calls \_load_model)

* @readme : different example
    * zymake stats [spec]
    * zymake check/generate [spec]
    * zymake runcmd [spec] | parallel #parralisation

## IN DOIING
* tabul | interface for tabulate (repeat /p2m)
* build_corpus and build_networks are identical : wrap in DataBase
* default VS const (argparser)???
    * settings object (AST)
* rename inference-\* file to \*.inf files
* ILFM creat sampler and proper inhheritance form GibbsSampler.
* @issue42


@structure
* plotManager (same scenario, axe, title, **kwargs**)
* OBJECT ! change frontendNetwork to Graph(Object), it is clear ?! (implement morphism in Object for example. (issue42)

@purge: 
* model/lda
* clean and homogeneize, the communities analysis framework. There is redoncdancy, and non consistent call to modularity...
* import models -> import pymake.models ?



# Baseline (to implement)

* Baseline
    * RESCAL -> prediction !?
    * logit rescal
    * m3F
    * gradient descent
    * neural networlk factorization

How should be the backend to make big scale learning with database and search engine interface...
* Spark/Hadoop interface... (hdfs = htable)

# Measure Centrality :
* proba que tout les lien à 0 soit liée ?  -- comparé à aux leader (ceux qui ont un haut degré)

* Mesure clulstering: 
    * contingency table / cluster%precision rappel F mesure ?
    * NMI (Mutuel Information Normalization)

### community
* equivalence struturelle (economie / partenariat)
* algo formelle (une communauté c'est une clique, k-clique / percolation)
* ensemble de neud fortement connecté, par rapport au reste du réseau.

-- voir **impossibilité theorem**

## Expe missing
hyperprior testing
Model evolution with K as well as hyperprior nonparametric model
