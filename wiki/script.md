
# Script

In the DMSA model, what we call actions are endpoint methods that can be called from the command line or a Rest request.

* Actions are defines as method of any class that inherit the `ExpeFormat` class.
* Those classe lives in python files that we call *scripts* and which are locate in the script/ dir by default.

## Special method

Any actions will be visible in pymake scope for execution, except if they start by an underscore `_`.

ExpeFormat class provides a set of special attribute, method and decorator:

### Attribute
* self.expe: the dict of the current expe.

### method
* self.\_preprocess: a method that will be exectuted before each script. useful to load some variable shared by each actions.
* self.get_data_path: the data path related to the current [data_type], where data can be load or stored 

### Decorator
* @plot: 
* @table



