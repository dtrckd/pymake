# PYMAKE

Pymake is a platform for making reproducible research. It provides tools adapted to ease the creation, maintenance, tracking and sharing of experiments. It has two main paradigms :

* Manage and navigate in your experiences, as a **command-line** interface.
* Models and workflows for Machine Learning experiments, as a **framework**.

## Features
* Specification of design of experimentations with a simple grammar,
* command-line toolkit for quick design and experience testing,
* Support experience rules filtering,
* Pipe experience for parallelization (see pymake cmd [expe]),
* browse, design and test several models and corpus find in the literature.

Perspectives :

* A online repo to push/fetch/search in {design of experimentations || models || scripts},
* Better documentation (or just a documentation, needs feedback!).


## Install


System dependencies : `apt-get install python3-setuptools python3-pip pyhton3-tk`

Numpy/scipy dependencies : `apt-get install libopenblas-dev gfortran`

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

* What model are there: `pymake -l model`
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

##### Workflow / Directory Structure

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


##### Pymake Commands

Create your own project:

    pymake init

If new models or script are added in the project, you'll need to update the pymake index :

    pymake update

Show/list things :

    pymake -l model   # show available models
    pymake -l script # show available scripts
    pymake -l expe   # show available designs of experimentation                                                                                                                             
    pymake show expe_name # equivalent to: pymake expe_name --simulate|-s

Run a experience :

    pymake run [expe_name] --script script_name [script options...]

##### Track your data and results

In order t  save and analyse your results, each unique experience need to be identified in a file. To do so we propose a mechanism to map settings/specification of an unique experience to a <filename>. Depending on what the developer want to write, the extension of this file can be modified. Pymake use three conventions : 

* <filename>.inf : csv file where each line contains the state of iterative process of an experiment,
* <filename>.pk : to save complex object usually at the end of an experiments, to load it after for analysis/visualisation,
* <filename>.json : to save information of an experiments ("database friendly").

###### formatting the filename -- _format

The choice of the filename will depends on the settings of the experiments. In order to specify the format of the filename, there is the special settings `--format  str_fmt`. `str_fmt` is a string template for the filename, with braces delimiter to specify what settings will be replaced, example : 

Suppose we have the following settings : 
```
settings = dict(name = 'myexpe',
                size = 42,
                key1 = 100,
                key2 = 'johndoe'
                _format = '{name}-{size}-{key1}_-_{key2}'
        )
```

The filename for this unique experiment will be 'myexpe-42-100_-_johndoe'

###### settings the path -- _refdir

By default all experiments results files will be written in the same directory (specify in `pymake.cfg`). In order to give a special subdirectory name for an experiment, the settings `--refdir str` is a string that represents the subdirectory for results files of the experiments.

###### Specifying what data to sage in .inf files.

to complete...



