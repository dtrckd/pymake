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
    git clone https://github.com/dtrckd/pymake
    cd pymake && make install

### Learn by examples

We provide an example of a design workflow with pymake
As an example of usage, we provide an **Search Engine** experience.

In a pymake project there is 4 main components : 

* The data : The input of any experience,
* A model : It represents our understanfing of the data,
* A script : Code that operate with the data and models,
* A Specification (spec/design) : It is the specicification of the context of an experiment. In order words, the parameters of an experience.


### Documentation

(to complete)

* pymake workflow
* pymake cmd and grammar
* ExpSpace and ExpTensor
* pymake.cfg
* Search and indexation


### Directory Structure

By default, pymake will use the configuration in the ~/.pymake directory. To create your own project use `pymake init`. 
It is designed to makes easy the creation and sharing of models and design of experimentations.
The pymake arborescence has the following directory :

* `model/` contains models -- selection with the `-m` options,
* `script/` contains scripts for action, -- selection with the `--script` options
* `spec/` contains specification of (design) experiments, -- can be precised as an argument after the second argument of pymake.
* `data/` contains datasets (and saved results) -- selection with the `-c` options,


Data Lookup and Output Data path specification are automatically adaptated from the design of experiments. Note that you can specify the format of the results of each expetiments with `--format options`, see examples.

The Results of experiments are stored in data/[specification].

If new models or script are added in the project, you'll need to update the pymake index : ```pymake update```.


