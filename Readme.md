Pymake helps making reproducible research by providing tools adapted for creating **complex and traceable design of experiments** and **Models for data analysis**.

This code is in a living devellopment and yet unstable.

## Zen

If a good (scientific) library do the job, wrap it:
* numpy
* networkx
* scikit-learn
* nltk
* tensorflow (to complete)
* pyspark (to complete)

## Logic

Once an experiments is designed, we deploy it using two script :

*  `zymake` to create the uniq experiment command for each of them.
*  `pysinc` to parralelize the jobs.

### Usage
###### Zymake
Script to create the path and command for a design of experiments:

###### Run experiments
    zymake runcmd EXPE_ID | parralle [opt]  # to make run a list of experience in parallel

or equivalently :

      pyrallel.sh EXPE_ID [nb CORES]

######  Results Analysis
    zymake path EXPE_ID json | some_matplotlib_script.py  # plot some results
    expe_meas.py EXPE_ID    #  Create a table of results


see also: [Directory Tree](#directory-tree).

### Directory Tree

* `data/` contains dataset and learning output results
* `results/` contains analysis and report
* `src/` code source of project

Data Lookup is coded in `util/frontend_io.py` and Output Data path specification depend on the following arguments :

    [script_name] [bdir/-d]/[type]/[corpus/-c]/[subdir/--refdir]/[model/-m][iterations/-i]

Results of experiments are stored in data/[specification].

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

## Example

Fit a Gaussian mixtures on a text corpus...

    Expe_ID = dict(model = 'kmeans++',
            K = 1e6,
            corpus = '20ngroups',
            vsm = 'tfidf',
            repeat = range(10))

(to complete)
