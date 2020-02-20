# 0.43
* improve spec type. change cvs_typo to measures
* import modelbase and improve model RandomGraphModel
* add script integration (load_model, load_data, compute_measure, dump_results)
* separate pmk core and repo (ml/, docsearch/, ...)

#  0.42.3
* auto make input_path when loading a corpus from a frontend (graph-tool,.)
* add load_data method in expeFormat
* add loguru log formating
* fix pmk-db tempfile for environment managing
* fix script homonimy with ExpeFormat module name and script
* improve test exposition
* use configparser and tempfile for env and settings management.
* add notebook and data directory
* add username and project name config keys

# 0.41.5

* solve local import in pymake repo (import from the root repo)
* process script : pmk -x killall # kill all pmk process.
* improve "plot tab" action and syntax (repeat automatic detection, value selection, etc)

# 0.41
* graph-tool driver integration completed + wmms model.
* improve frontend and model baseclasse, toward templating.
* fix print format for returns of action-scripts in sandbox.
* fix exp_append\* type for grammarg
* ouptput_path formatting will convert float required from \_format to .2f.
* add a test target to the makefile => functional test
* return a exit status positive if an expe failed.
* fix i/o auto compression filesystem.

# 0.4

* data path are simplified and trainind and results are now in {data}/.pymake/ .
* Integration of ModelSkl for skleanr model like integration.
* allow {\_spec} to set default spec to use in \_default_expe.

* (autocompletion) todo ~/.bash_completion.d/ (autmatically use it ?)
* \_alias maping dict inside expDesign to map/format output.
* object return by endpoint actions (scipts) are printed to stdout.
* pmk diff spec1 spec2 -- diff between two spec

# 0.3
first alpha stable release of pymake.
