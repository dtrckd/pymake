### Directory Logic
data/ contains dataset and learning output results
results/ contains analysis and report
src/ code source of project

Data Lookup is coded in util/frontend*.py and:
* Output Data are parse in directory:  [bdir/-d]/[type]/[corpus/-c]/[subdir/--refdir]/[model/-m]
* ouput result lookup according to: -m model wich depend on [-k #topics -n #size -i #iterations --homo int --hyper str
    * create: inference-model_K_hyper_homo_n. (as weel as pickle file and json predict result)

### Models

#### iIndian Buffet Process

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
It prooved that Nonparametric bayesian are the infinite limite of parametric ones. Hence, by fixing K we fall down on parametric models. it avoid kearning part for model dimension. So it could speed up inference and reduce memory complexity.



