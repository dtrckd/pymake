# PYMAKE

Pymake is machine-friendly platform for making reproducible research. It provides tools adapted for the creation of :
* Complex and traceable design of experiments, as a **command-line** interface.
* Models and workflows for Machine Learning, as a **framework**.

This code is in a living development stage and yet unstable.

### Features
* Experience design specification by simple grammar.
* support experience rules filtering
* commandline toolkit for quick design and experience testing
* browse design and test several models and corpus find in the
literrature
Ongoing development :
* A database to share (push/fetch/search) Design of experimentation.
* Better specification of the formal grammar of design of experimentation.
* Better documentation.

### Usage
Script to create the path and commands for a design of experiments :

    from pymake import Gramexp, Zymake
    # Fit a Gaussian mixtures on a text corpus...

    Expe_ID = dict(model = 'kmeans++',
            K = 1e6,
            corpus = '20ngroups',
            vsm = 'tfidf',
            repeat = range(10))

    # Only the first epoch/repeat here.
    gram = Gramexp(Expe_ID)
    model.fit(data)
    model.predict()

List design of experiments :

    pymake list

Show one experiments :

    pymake show ExpeID


###### Run experiments
With gnu-parallel :

    pymake cmd ExpeID --script ./fit.py | parallel [opt]  # to make run a list of experience in parallel


######  Results Analysis
    pymake --script ExpeDesign


### Directory Tree

* `data/` contains dataset and learning output results,
* `results/` contains analysis and report,
* `pymake/` code source of the models and frontends.

Data Lookup is coded in `util/frontend_io.py` and Output Data path specification are automatically adaptated from the design of experiments. Note that you can specify the format of the results of each expetiments with `--format options`, see examples.

The Results of experiments are stored in data/[specification].


### Inference Scheme

* Gibbs Samplers
* Variationnal Bayes

### Implemented Models
* LDA (text)
* MMSB (networks)
* ILFM (networks)

Notes : it includes nonparametric version using Hierarchical Dirichlet
Process (HDP) and Beta Process (IBP).

### Examples

(to complete)
