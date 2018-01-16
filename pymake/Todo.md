# IN DOIING


### @Debug
* there is a random effect when doing checknetworks zipf. Class re-ordering is stochastik f. why ?!

# @v0.3
* ipython builder helper
* web server interface and visualization  (flask...)

@CLI
* pmk diff expe1 expe2 # show diff between expe...
* pmk push [spec] [opts] # push expe in spec !!! (Update MAN)
* pmk expe_id1 expe_id2 # merge exp => We need expGroup for that

### Bash Completion
echo /etc/bash_completion.d/pymake
    _pymake() 
    {
        local cur prev opts
        COMPREPLY=()
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
        opts="--help --verbose --version"

        if [[ ${cur} == -* ]] ; then
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
        fi
    }
    complete -F _pymake pymake
    complete -F _pymake pmk

* complete opts and args (in spec) (and -x and -m and -c) depending on where iam on the pymak.cfg to get the autompletion file that generate pymake update yeah !

* feature --> autocompletion yeah !!!

@Core
* print warning on duplicate class name duplicate for Scipt, Model, Spec or Copus, but especially for Script and Spec class définition wich is more probable to happen


@Frontend
* merge vocabulary and frontendtext
* build_corpus and build_networks are identical : wrap in DataBase
* @issue42

@Model
* move entropy() to modelBase. 
    * review _reduce_latent vs get_params...!

@MODEL Specification:
* -> catch type error in manager.get_model._model.__init,:
        1. first find the number of required argument andt cut the expe surplus
        2. find a way to pass the argument  from signature **kwargs to command-line. (index the signature)
*  \_initialize  random init !!!! do cleaner init of class for empty model....
* ModelManager constructor from_model (object) (it calls \_load_model)
* print help/signature from command_line (model/corpus)
* ILFM create sampler and proper inhheritance form GibbsSampler.

@Corpus/Dataset : 
* expe_meas script ("*" specification obsolete ???? )
* CorpusModules :load from package (sklearn) and disk (
* Whoosh integration !!!
* LDA on my own paper !


@format_plot etc
* decorator for tabulate in ExpFormat ?
* plotManager (same scenario, axe, title, **kwargs**)
* OBJECT ! change frontendNetwork to Graph(Object), it is clear ?! (implement morphism in Object for example. (issue42)

@purge: 
* model/lda
* clean and homogeneize, the communities analysis framework. There is redoncdancy, and non consistent call to modularity...
* import models -> import pymake.models ?


@bigdata 
* How should be the backend to make big scale learning with database and search engine interface...
* Spark/Hadoop interface... (hdfs = htable)
* MPI / pymake integration ?

@Examples
* [docsearch] own pdftotext ? quickest ? multi platform !?

## Expe missing
* isolate my expe (but keep general model AND frontend, and manager...)....hyperprior testing
Model evolution with K as well as hyperprior nonparametric model
