## It start with an example
How to see what the 20 news group corpus looks like when it is a mixture of gaussian of the distribution of each documents ?
arr nooo


    Expe_ID = dict(model = 'kmeans++', 
            K = 1e6, 
            corpus = '20ngroups',
            vsm = 'tfidf',
            repeat = range(10))

(to complete)



## Mojo

* Do not reinvent the wheel, re-use good free library like:
    * nltk
    * scikit-learn
    * networkx
    * pyspark (to complete)
* Wrap things to add freedom of modelisation to existing library.

Architecture inspiration: Django.

## Logic

### Zymake
Script to manage Input and output of an experimental plateforms:

#### Launch a bunch of fitting jobs
    zymake runcmd EXPE_ID | pysync.py -- to make run a list of experience in parallel

#### Analyse results
    zymake path EXPE_ID json | some_matplotlib_script.py
    **or**
    expe_meas.py EXPE_ID


see also: [Directory Tree](#directory-tree).

### Directory Tree
data/ contains dataset and learning output results
results/ contains analysis and report
src/ code source of project

Data Lookup is coded in util/frontend*.py and:
* Output Data are parse in directory:  [bdir/-d]/[type]/[corpus/-c]/[subdir/--refdir]/[model/-m]
* ouput result lookup according to: -m model wich depend on [-k #topics -n #size -i #iterations --homo int --hyper str
    * create: inference-model_K_hyper_homo_n. (as weel as pickle file and json predict result)


## Models


#### Indian Buffet Process

Collapsed Gibbs sampling:
Uncollapsed Gibbs sampling:
Variational Bayes:

##### Networks applications
ILFRM

#### Hierarchical Dirichlet Process 

Collapsed Gibbs sampling:


##### Text Applications
LDA

##### Networks Application
MMSB

### Parametric Applications:
It is prooved that Nonparametric bayesian can be view as the infinite limit of parametric ones. Hence, by fixing K we fall down on parametric models. it avoids learning part for model dimension. So it could speed up inference and reduce memory complexity.


