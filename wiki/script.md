
# Script

In the DMSA model, what we call actions are endpoint methods that can be called from the command line or a Rest request.

* Actions are defines as method of any class that inherit the `ExpeFormat` class.
* Those classe lives in python files that we call *scripts* and which are locate in the script/ dir by default.

## Special method

Any actions will be visible in pymake scope for execution, except if they start by an underscore `_`.

ExpeFormat class provides a set of special attribute, method and decorator:

### Attribute
* self.expe: the dict of the current expe.
* self.expe_size: how many expe in the sandbox
* self.\_it: a compteur that identify the current expe (from 0 to self.expe_size-1)
* self.output_path: a path basename that uniquely identify the ecurrent expe.
* self.D: a global container to store variables and share across all the expe/runs.

### method
* self.\_preprocess: a method that will be exectuted before each script. useful to load some variable shared by each actions.
* self.log_expe: should return a string to custom the headers message prompt by pymake for each expe/run.
* self.get_data_path: the data path related to the current [data_type], where data can be load or stored 
* self.load_model(data, [init=False]): Return a model initialized for the current experience. If init is True, the model is load from pickle file.

### Decorator
* @plot: 
* @table



