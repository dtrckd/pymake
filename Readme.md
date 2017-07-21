# PYMAKE

Pymake is a machine-friendly platform for making reproducible research. It provides tools adapted for the creation of :
* Complex and traceable design of experiments, as a **command-line** interface.
* Models and workflows for Machine Learning, as a **framework**.

## Features
* Specification of design of experimentations with a simple grammar,
* commandline toolkit for quick design and experience testing,
* Support experience rules filtering,
* Pipe experience for parallelization (see pymake cmd [expe]),
* browse, design and test several models and corpus find in the literature.

Ongoing development :

* A database to share (push/fetch/search) Design of experimentations,
* Better specification of the formal grammar of design of experimentation,
* Better documentation (or just a documentation).


## Install

```bash
git clone https://github.com/dtrckd/pymake
cd pymake && make install
```

## Learn by examples

We provide an example of a design workflow with pymake by providing a **Search Engine** experience.

The context of the experience is has follows :
* **Data** : documents to search in, here it will be pdf documents (like articles for example),
* **Model** : A bm25 model, that assumes a information model of bag of word representation.
* **Script** : basically two scripts :
    + a fit script that build  the index,
    + a search script that return relevant documents.
* Experience Parameters: A default **specification** is in  script.a-script.\_default_expe

Setup the experience (needed just once) :

```bash
cd examples/docsearch/
make setup
```

Then a typical pymake usage :

```bash
pymake run --script fit --path path/to/your/pdfs/   # index your pdf documents, take a coffe
pymake run --script search "your text search request"  # show relevant information
```
Or show only the first match :  `pymake run --script search "your text search request" --limit 1`

To add new model, a new script, you need to write it in the dedicated folder following the base class implementations.

Then you can list some informations about pymake :

* What model are there: `pymake -l atom`
* What experience are there: `pymake -l expe`
* What script are there: `pymake -l script`
* Show signatures of methods in a script ('ir' script): `pymake -l --script ir`


Note that pymake provides  mechanisms to save and track results for designing, analysing and reproducing complex experiments.
This will be documented soon.


## Documentation


1. Workflow / directory structure
2. pymake commands
3. pymake.cfg
3. ExpSpace and ExpTensor
5. Track your data and results
6. Search and indexation

(to be completed)

----

###### Workflow / Directory Structure

In a pymake project there is 4 main components, associated to 4 directories :

* `data/`: The input of any experience,
    + contains datasets (and saved results) <!--  selection with the `-c` options and see frontendManager -->,
* `model/`: It represents our understanding of the data,
    + contains models -- every class with a `fit` method <!-- selection with the `-m` options and see ModelManager -->,
* `script/`: Code that operate with the data and models,
    + contains scripts for actions, -- every class that inherit `pymake.ExpeFormat` <!-- selection with the `-r` options -->
* `spec/` : It is the specifications of the context of an experiment. In order words, the parameters of an experience.
    + contains specification of (design) experiments, -- can be precised as an argument after the second argument of pymake.

Along with those directory there is two system files :
* pymake.cfg : at the root of a project (basically define a project) specify the default and contrib : data | model | script | spec, and other global options, <!-- document each entry -->
* gramarg.py : define the command-line options for a project. <!-- explaine the exp_append type -->


###### Pyamke Commands

Create your own project:
    pymake init

If new models or script are added in the project, you'll need to update the pymake index : 
    pymake update

Show/list things : 
    pymake -l atom   # show available models (don't ask me why is not -l model, we'll see that later!)
    pymake -l script # show available scripts
    pymake -l expe   # show available designs of experimentation                                                                                                                             
    pymake show expe_name # equivalent to: pymake expe_name --simulate|-s

Run a experience :
    pymake run [expe_name] --script script_name [script options...]

###### Track your data and results

Pymake provide a mechanism in order to track data from a specification to another.

Data Lookup and Output Data path specification are automatically adapted from the design of experiments. Note that you can specify the format of the results of each experiments with `--format options`, see examples.

The Results of experiments are stored in data/[specification].
