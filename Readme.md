# PYMAKE

Pymake is a machine-friendly platform for making reproducible research. It provides tools adapted for the creation of :
* Complex and traceable design of experiments, as a **command-line** interface.
* Models and workflows for Machine Learning, as a **framework**.

This code is in a living development stage and yet unstable.

### Features
* Specification of design of experimentations with simple grammar,
* commandline toolkit for quick design and experience testing,
* Support experience rules filtering,
* Pipe experience for parallelization (see pymake cmd [expe]),
* browse, design and test several models and corpus find in the literrature.

Ongoing development :

* A database to share (push/fetch/search) Design of experimentations,
* Better specification of the formal grammar of design of experimentation,
* Better documentation.

### Install
    make install
    # try it
    pymake --help

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

    pymake -l        # Current design
    pymake -l script # Available scripts
    pymake -l atom   # Available models

Show one experiments :

    pymake show ExpeID


###### Run experiments
With gnu-parallel :

    pymake cmd ExpeID --script fit | parallel [opt]  # to make run a list of experience in parallel


######  Results Analysis
    pymake exec ExpeID --script plot


### Directory Tree

By default, pymake will use the configuration in the ~/.pymake directory. To create your own project use ```pymake init```. It is design to makes easy the creation and sharing of models and design of experimentation. The pymake arborescence has the following directory :

* `model/` contains models -- selection with the ```-m``` options,
* `data/` contains datasets (and saved results) -- selection with the ```-c``` options,
* `script/` contains scripts for action, -- selection with the ```--script``` options
* `results/` contains analysis and report,


Data Lookup and Output Data path specification are automatically adaptated from the design of experiments. Note that you can specify the format of the results of each expetiments with `--format options`, see examples.

The Results of experiments are stored in data/[specification].

If new models or script are added in the project, you'll need to update the pymake index : ```pymake update```.


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
