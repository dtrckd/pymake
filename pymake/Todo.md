# Code Factoring

use args...https://github.com/kennethreitz/args instead parser for fit.py
use arparser as standard parser !

@purge: 
* model/lda
* clean and homogeneize, the communities analysis framework. There is redoncdancy, and non consistent call to modularity...

@todo: 
* OBJECT ! change frontendNetwork to Graph(Object), it is clear ?! (implement morphism in Object for example. (issue42)
* @frontend: better print the status of what's going on in the beginning of fit.


@structure
* simpliy frontend:
    * check [dataset/model](source) [zipf|homo](measure)  -- plot figure / things
    * tabulate (tensor spec)  -- tabs
* ExpeManger (factorize the scripts)
* plotManager (same scenario, axe, title, **kwargs**)


@python3
* improve compatibility : checl map filter an zip (change zip to .T in numpy)

# Implement some baseline

* Baseline
    * RESCAL -> prediction !?
    * logit rescal
    * m3F
    * nonparam
    * MMSB ! (infinite - dynamic...)
    * IBP

How should be the backend to make big scale learning with database and search engine interface...
* Spark/Hadoop interface... (hdfs = htable)

## Mesure attachement prefarential:
* proba que tout les lien à 0 soit liée ?  -- comparé à aux leader (ceux qui ont un haut degré)

* Mesure clulstering: 
    * contingency table / cluster%precision rappel F mesure ?
    * NMI (Mutuel Information Normalization)

### communauté
* equivalence struturelle (economie / partenariat)
* algo formelle (une communauté c'est une clique, k-clique / percolation)
* ensemble de neud fortement connecté, par rapport au reste du réseau.

-- voir **impossibilité theorem**

## Expe missing
hyperprior testing
Model evolution with K as well as hyperprior nonparametric model
