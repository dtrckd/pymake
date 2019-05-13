import sys, os, re
import pickle, json
import pkgutil, pyclbr, inspect, ast
import traceback
from importlib import import_module
from collections import OrderedDict
from itertools import groupby
import zlib
import numpy as np

from pymake import ExpDesign, ExpeFormat, get_pymake_settings

import logging
lgg = logging.getLogger('root')




''' Python I/O Parsing Tools. '''



class PyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(PyEncoder, self).default(obj)


def resolve_filename(fn, ext='pk'):
    splt = fn.split('.')
    if splt[-1] not in ['pk', 'json', 'gt']:
        fn += '.' + ext
    return fn

def load(fn, ext='pk', silent=False, driver=None):
    fn = resolve_filename(fn, ext)
    ext = fn.split('.')[-1]

    # Seek for compressed data
    if os.path.exists(fn+'.gz'):
        compressed = True
        fn += '.gz'
    else:
        compressed = False
        lgg.debug('Data is not compressed: %s' % fn)

    if not silent:
        lgg.info('Loading data : %s' % fn)

    if driver:
        return driver(fn)
    elif ext == 'pk':
        if compressed:
            with open(fn, 'rb') as _f:
                return pickle.loads(zlib.decompress(_f.read()))
        else:
            with open(fn, 'rb') as _f:
                return pickle.load(_f)
        #except:
        #    # python 2to3 bug
        #    _f.seek(0)
        #    try:
        #        model =  pickle.load(_f, encoding='latin1')
        #    except OSError as e:
        #        cls.log.critical("Unknonw error while opening  while `_load_model' at file: %s" % (fn))
        #        cls.log.error(e)
        #            return
    elif ext == 'json':
        if compressed:
            with open(fn, 'rb') as _f:
                return json.loads(zlib.decompress(_f.read()))
        else:
            with open(fn) as _f:
                return json.load(_f)
    else:
        raise ValueError('File format unknown: %s' % ext)


def save(fn, data, ext='pk', silent=False, compress=None, driver=None,
         compressed_pk=True, compressed_json=False):

    if compress is not None:
        compressed_pk = compressed_json = compressed

    fn = resolve_filename(fn, ext)
    ext = fn.split('.')[-1]

    if not silent:
        lgg.info('Saving data : %s' % fn)

    if driver:
        if data is None:
            return driver(fn)
        else:
            return driver(fn, data)
    elif ext == 'pk':
        if compressed_pk:
            fn += '.gz'
            obj = zlib.compress(pickle.dumps(data))
            with open(fn, 'wb') as _f:
                return _f.write(obj)
        else:
            with open(fn, 'wb') as _f:
                return pickle.dump(data, _f, protocol=pickle.HIGHEST_PROTOCOL)

    elif ext == 'json':
        # @Todo: Option to update if file exists:
        #res = json.load(open(fn,'r'))
        #res.update(data)
        #self.log.info('Updating json data: %s' % fn)
        if compressed_json:
            fn += '.gz'
            obj = zlib.compress(json.dumps(data))
            with open(fn, 'wb') as _f:
                return _f.write(obj)
        else:
            with open(fn, 'wb') as _f:
                return json.dump(data, _f, cls=PyEncoder)
    else:
        raise ValueError('File format unknown: %s' % ext)




def is_abstract(cls):
    ABCFLAG = '__abstractmethods__'
    isabc = hasattr(cls, ABCFLAG) or inspect.isabstract(cls)
    return isabc

def is_empty_file(filen):

    if os.path.exists(filen+'.gz'):
        filen = filen+'.gz'

    if not os.path.isfile(filen) or os.stat(filen).st_size == 0:
        return True

    try:
        with open(filen, 'r') as f: first_line = f.readline()
    except UnicodeDecodeError:
        first_line = '...'

    if first_line[0] in ('#', '%') and sum(1 for line in open(filen)) <= 1:
        # empy file
        return True
    else:
       return False

# untested!
def get_decorators(cls):
    target = cls
    decorators = {}

    def visit_FunctionDef(node):
        decorators[node.name] = []
        for n in node.decorator_list:
            name = ''
            if isinstance(n, ast.Call):
                name = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
            else:
                name = n.attr if isinstance(n, ast.Attribute) else n.id

            decorators[node.name].append(name)

    node_iter = ast.NodeVisitor()
    node_iter.visit_FunctionDef = visit_FunctionDef
    node_iter.visit(ast.parse(inspect.getsource(target)))
    return decorators


class PackageWalker(object):
    ''' Import all submodules of a module, recursively.

        Module walker implements this class by overloading the methods 'submodule_hook',
        that should call one of the following methods :
            * submodul_hook_by_attr : to reach/get module if an attribut is {defined}
            * submodul_hook_by_class : to reach/get a module if a class is of a {defined} type.

        The {defined} value should be defined in the {_default_attr} with a class inherit PackageWalker is defining the class.
        (see the *Loader for .e.g).

        Attributes
        ----------
        packages : OrederedDict
            contains the module by name
        _cls_browse : pyclbr
            contains module informations
    '''

    _default_attr = dict(
        attr_filter = None,
        class_filter = None,
        max_depth = 1,
        prefix = False,
        shrink_module_name = True,
    )

    def __init__(self, module_name, **kwargs):
        [setattr(self, k, kwargs.get(k, v)) for k,v in self._default_attr.items()]

        self._cls_browse = {}
        self._unavailable_modules = []

        ### Core function
        self.packages = self._get_packages(module_name)

        if len(self._unavailable_modules) > 0:
            lgg.warning('There is some unavailable modules : %s' % self._unavailable_modules)

    def __repr__(self):
        return self.packages

    def _get_packages(self, module_name,  _depth=0):
        ''' recursive call with regards to depth parameter '''

        argv = sys.argv

        if self.prefix is True:
            prefix = module_name
        elif self.prefix is False:
            prefix = ''
        elif isinstance(self.prefix, str):
            prefix = self.prefix

        packages = OrderedDict()
        try:
            module = import_module(module_name)
        except ImportError as e:
            lgg.warning('package unavailable (%s) : %s' % (e, module_name))
            if '-v' in argv:
                print(traceback.format_exc())
            return packages

        for loader, name, is_pkg in pkgutil.walk_packages(module.__path__):
            submodule_name = module_name + '.' + name
            if is_pkg and _depth < self.max_depth:
                next_depth = _depth + 1
                packages.update(self._get_packages(submodule_name, next_depth))
                continue
            try:
                submodule = import_module(submodule_name)
            except ImportError as e:
                lgg.debug('submodule unavailable (%s) : %s'%(e, submodule_name))
                if '-v' in argv:
                    print(traceback.format_exc())
                self._unavailable_modules.append(submodule_name)
                if 'algo' in str(e):
                    lgg.warning('Please report this error: pymake.io algo condition ?')
                    submodule = import_module(submodule_name)
                continue
            except Exception as e:
                lgg.critical('Module Error: %s' % e)
                if '-v' in argv:
                    print(traceback.format_exc())
                continue

            spck = self.submodule_hook(submodule, prefix, name=name)
            if spck:
                packages.update(spck)

        return packages

    def submodule_hook(self, *args, **kwargs):
        raise NotImplementedError

    def submodule_hook_by_attr(self, submodule, prefix, name=None):
        try:
            _packages = pyclbr.readmodule(submodule.__name__)
        except:
            # @DEBUG : use shelve to get it
            sys.path.append(os.getenv('PWD'))
            #_packages = pyclbr.readmodule(submodule.__name__)
            _packages = pyclbr.readmodule('.'.join(submodule.__name__.split('.')[1:]))
        self._cls_browse.update(_packages)
        packages = {}
        for cls_name, cls  in _packages.items():
            obj = getattr(submodule, cls_name)
            if not hasattr(obj, self.attr_filter) or is_abstract(obj):
            #if not any([hasattr(obj, attr) for attr in self.attr_filter]) or is_abstract(obj):
                continue
            if self.shrink_module_name:
                prefix = prefix.split('.')[0]
            if prefix:
                name = prefix +'.'+ cls_name.lower()
            else:
                name = cls_name.lower()

            packages[name] = obj

        return packages

    def submodule_hook_by_class(self, submodule, prefix, name=None):
        try:
            _packages = pyclbr.readmodule(submodule.__name__)
        except:
            # @DEBUG : use shelve to get it
            sys.path.append(os.getenv('PWD'))
            #_packages = pyclbr.readmodule(submodule.__name__)
            _packages = pyclbr.readmodule('.'.join(submodule.__name__.split('.')[1:]))
        self._cls_browse.update(_packages)
        packages = {}
        for cls_name, cls  in _packages.items():
            obj = getattr(submodule, cls_name)
            if not issubclass(obj, self.class_filter) or is_abstract(obj):
            # ExpDesign don't load without type() ?
            #if not issubclass(type(obj), self.class_filter) or is_abstract(obj):
                continue

            if prefix:
                name = prefix +'.'+ cls_name.lower()
            else:
                name = cls_name.lower()
            packages[name] = obj

        return packages

class ModelsLoader(PackageWalker):
    ''' Load models in a modules :
        * lookup for models inside a module by fuzzing the name of the module.
        * Need either a :
            * `fit' method to be identified as model,
            * a `module' attribute name => that should have a fit method.
    '''
    def submodule_hook(self, *args,  **kwargs):
        return super(ModelsLoader, self).submodule_hook_by_attr(*args,  **kwargs)

    @classmethod
    def get_packages(cls, module_name, **kwargs):
        if not 'attr_filter' in kwargs:
            kwargs['attr_filter'] = 'fit'

        # need of not needs ?
        #if isinstance(module_name, list):
        #    packs = {}
        #    for m in module_name:
        #        packs.update(cls(m, **kwargs).packages)
        #    return packs
        #else:
        #    return cls(module_name, **kwargs).packages

        return cls(module_name, **kwargs).packages

    @staticmethod
    def _fuzz_funcname(name):
        ''' Fuzzing search '''
        # use regexp ?
        # @Obsolete: use attr_method only...
        nsplit = name.split('_')
        nsplit1 = ''.join([nsplit[0].upper()] + nsplit[1:])
        nsplit2 = ''.join(list(map(str.title, nsplit)))
        return (name, name.lower(), name.upper(), name.title(),
                name.lower().title(), nsplit1, nsplit2)

    def _submodule_hook_by_fuzz(self, submodule, prefix, name):
        obj_list = dir(submodule)
        idx = None
        packages = OrderedDict()
        for n in self._fuzz_funcname(name):
            # Search for the object inside the submodule
            try:
                idx = obj_list.index(n)
                break
            except:
                continue
        if idx is None: return

        obj = getattr(submodule, obj_list[idx])
        if hasattr(obj, self.attr_filter):
            if prefix:
                name = prefix +'.'+ name
            # remove duplicate name
            name = '.'.join([x[0] for x in groupby(name.split('.'))])
            packages[name] = obj

        return packages

    @classmethod
    def get_atoms(cls,  _type='short'):
        if _type == 'short':
            shrink_module_name = True
        elif _type == 'topos':
            shrink_module_name = False

        packages = get_pymake_settings('_model')
        atoms = OrderedDict()
        for pkg in packages:
            if len(pkg) > 8:
                prefix = pkg[:3]
                if '.' in pkg:
                    prefix  += ''.join(map(lambda x:x[0], pkg.split('.')[1:]))
            else:
                prefix = True
            atoms.update(ModelsLoader.get_packages(pkg,  prefix=prefix, max_depth=3, shrink_module_name=shrink_module_name))
        return atoms

class CorpusLoader(PackageWalker):
    def submodule_hook(self, *args, **kwargs):
        raise NotImplemented('todo :scikit-learn loohup')

    #@classmethod
    #def get_atoms(cls_type=None):
    #    # get some information about what package to use in _spec ...
    #    atoms = cls.get_packages('pymake.data')
    #    return atoms

class ScriptsLoader(PackageWalker):

    module = ExpeFormat

    def submodule_hook(self, *args,  **kwargs):
        return super(ScriptsLoader, self).submodule_hook_by_class(*args, **kwargs)

    @classmethod
    def get_packages(cls, **kwargs):
        module_name = get_pymake_settings('_script')
        if not 'class_filter' in kwargs:
            kwargs['class_filter'] = cls.module

        if isinstance(module_name, list):
            packs = {}
            for m in module_name:
                packs.update(cls(m, **kwargs).packages)
            return packs
        else:
            return cls(module_name, **kwargs).packages

    @classmethod
    def get_atoms(cls):
        atoms = dict()
        modules = get_pymake_settings('_script')
        modules = [modules] if type(modules) is str else modules
        for module in modules:

            s = cls(module, class_filter=cls.module)

            ## get decorator for each class
            #class2met2dec = {}
            #for method, _class in classs.packages.items():
            #    append decoratpr information to filter @atpymake

            for surname, _module in s.packages.items():
                name = _module.__name__
                module = s._cls_browse[name]
                methods = list(module.methods.keys())
                for m in methods.copy():
                    _m = getattr(s.packages[name.lower()], m)
                    if not inspect.isfunction(_m) and m != '__call__':
                        methods.remove(m)
                    elif '__call__' == m:
                        methods.remove('__call__')
                        methods.append(name.lower())
                    elif m.startswith('_'):
                        methods.remove(m)
                    elif m in dir(cls.module):
                        methods.remove(m)

                content = {}
                content['scriptname'] = name
                content['scriptsurname'] = surname
                content['module_file'] = module.file
                content['module'] = _module.__module__
                content['_module'] = _module
                #content['module_name'] = '.'.join((module.name, module.module))
                content['module_super'] = module.super
                content['methods'] = set(methods)
                atoms[name] = content

        return atoms

class SpecLoader(PackageWalker):
    ''' Load specification of design experimentation :
        * Lookup for ExpDesign subclass in the target module.
    '''

    module = ExpDesign

    def submodule_hook(self, *args, **kwargs):
        return super(SpecLoader, self).submodule_hook_by_class(*args, **kwargs)

    @staticmethod
    def default_spec():
        import pymake
        return getattr(pymake, '__spec')

    #@staticmethod
    #def _default_spec():
    #    #module_name = get_pymake_settings('_spec')
    #    module_name = 'pymake.spec.netw'
    #    spec = import_module(module_name)
    #    for m in dir(spec):
    #        o = getattr(spec,m)
    #        if issubclass(o, cls.module) and not o is cls.module:
    #            return o()

    @classmethod
    def get_packages(cls, **kwargs):
        module_name = get_pymake_settings('_spec')
        if not 'class_filter' in kwargs:
            kwargs['class_filter'] = cls.module

        if isinstance(module_name, list):
            packs = {}
            for m in module_name:
                packs.update(cls(m, **kwargs).packages)
            return packs
        else:
            return cls(module_name, **kwargs).packages

    @classmethod
    def get_atoms(cls):
        expe_designs = []
        atoms = dict()

        modules = get_pymake_settings('_spec')
        modules = [modules] if type(modules) is str else modules
        for module in modules:

            s = cls(module, class_filter=cls.module)

            for surname, _module in s.packages.items():
                name = _module.__name__
                module = s._cls_browse[name]

                expd = getattr(import_module(module.module), name)()

                content = {}
                content['script_name'] = surname
                content['module_name'] = '.'.join((_module.__module__, _module.__name__))
                content['_module'] = module
                content['exp'] = expd._specs()
                atoms[name] = content

        return atoms














#
# @Obsolete Code:
#   Altough, there is some idea to re-build spec/paramas
#   by path traversing. It coul also be used to discover expe
#   from path traversing. But re-implementing it is advised.
#
#


### directory/file tree reference
# Default and New values
# @filename to rebuild file
_MASTERKEYS = OrderedDict((
    ('corpus'      , None),
    ('repeat'      , None),
    ('model'       , None),
    ('K'           , None),
    ('hyper'       , None),
    ('homo'        , None),
    ('N'           , 'all'),
))


# Debug put K_end inside the json
#_Key_measures = [ 'g_precision', 'Precision', 'Recall', 'K',
#                 'homo_dot_o', 'homo_dot_e', 'homo_model_o', 'homo_model_e',
#                ]
_Key_measures = [ 'g_precision', 'Precision', 'Recall', 'K',
                 'homo_dot_o', 'homo_dot_e', 'homo_model_o', 'homo_model_e',
                 'f1'
                ]

_New_Dims = [{'measure':len(_Key_measures)}]






def tree_hook(key, value):
    hook = False
    if key == 'corpus':
        if value in ('generator', ):
            hook = True
    return hook


# Warning to split('_')
def get_conf_from_file(target, mp):
    """ Return dictionary of property for an expe file.
        @mp: map parameters
        format model_K_hyper_N
        @template_file order important to align the dictionnary.
        """
    masterkeys = _MASTERKEYS.copy()
    template_file = masterkeys.keys()
    ##template_file = 'networks/generator/Graph13/debug11/immsb_10_auto_0_all.*'

    data_path = get_pymake_settings('project_data')
    # Relative path ignore
    if target.startswith(data_path):
        target.replace(data_path, '')

    path = target.lstrip('/').split('/')

    _prop = os.path.splitext(path.pop())[0]
    _prop = path + _prop.split('_')

    prop = {}
    cpt_hook_master = 0
    cpt_hook_user = 0
    # @Debug/Improve the nasty Hook here
    def update_pt(cur, master, user):
        return cur - master + user

    #prop = {k: _prop[i] for i, k in enumerate(template_file) if k in mp}
    for i, k in enumerate(template_file):
        if not k in mp:
            cpt_hook_master += 1
            continue
        pt = update_pt(i, cpt_hook_master, cpt_hook_user)
        hook = tree_hook(k, _prop[pt])
        if hook:
            cpt_hook_user += 1
            pt = update_pt(i, cpt_hook_master, cpt_hook_user)
        prop[k] = _prop[pt]

    return prop

def get_conf_dim_from_files(targets, mp):
    """ Return the sizes of proporties in a list for expe files
        @mp: map parameters """
    c = []
    for t in targets:
        c.append(get_conf_from_file(t, mp))

    sets = {}
    keys_name = mp.keys()
    for p in keys_name:
        sets[p] = len(set([ _p[p] for _p in c ]))

    return sets

def forest_tensor(target_files, map_parameters):
    """ It has to be ordered the same way than the file properties.
        Fuze directory to find available files then construct the tensor
        according the set space fomed by object found.
        @in target_files has to be orderedDict to align the the tensor access.
    """
    # Expe analyser / Tabulyze It

    # res shape ([expe], [model], [measure]
    # =================================================================================
    # Expe: [debug, corpus] -- from the dirname
    # Model: [name, K, hyper, homo] -- from the expe filename
    # measure:
    #   * 0: global precision,
    #   * 1: local precision,
    #   * 2: recall

    ### Output: rez.shape rez_map_l rez_map
    if not target_files:
        lgg.info('Target Files empty')
        return None

    #dim = get_conf_dim_from_files(target_files, map_parameters) # Rely on Expe...
    dim = dict( (k, len(v)) if isinstance(v, (list, tuple)) else (k, len([v])) for k, v in map_parameters.items() )

    rez_map = map_parameters.keys() # order !
    # Expert knowledge value
    new_dims = _New_Dims
    # Update Mapping
    [dim.update(d) for d in new_dims]
    [rez_map.append(n.keys()[0]) for n in new_dims]

    # Create the shape of the Ananisys/Resulst Tensor
    #rez_map = dict(zip(rez_map_l, range(len(rez_map_l))))
    shape = []
    for n in rez_map:
        shape.append(dim[n])

    # Create the numpy array to store all experience values, whith various setings
    rez = np.zeros(shape) * np.nan

    not_finished = []
    info_file = []
    for _f in target_files:
        prop = get_conf_from_file(_f, map_parameters)
        pt = np.empty(rez.ndim)

        assert(len(pt) - len(new_dims) == len(prop))
        for k, v in prop.items():
            try:
                v = int(v)
            except:
                pass
            try:
                idx = map_parameters[k].index(v)
            except Exception as e:
                lgg.error(prop)
                lgg.error('key:value error --  %s, %s'% (k, v))
                raise ValueError
            pt[rez_map.index(k)] = idx

        f = os.path.join(get_pymake_settings('project_data'), _f)
        d = load(f)
        if not d:
            not_finished.append( '%s not finish...\n' % _f)
            continue

        try:
            pt = list(pt.astype(int))
            for i, v in enumerate(_Key_measures):
                pt[-1] = i
                ### HOOK
                # v:  is the measure name
                # json_v: the value of the measure
                if v == 'homo_model_e':
                    try:
                        json_v =  d.get('homo_model_o') - d.get(v)
                    except: pass
                elif v == 'f1':
                    precision = d.get('Precision')
                    try:
                        recall = d.get('Recall')
                        recall*2
                    except:
                        # future remove
                        recall = d.get('Rappel')
                    json_v = 2*precision*recall / (precision+recall)
                else:
                    if v == 'Recall':
                        try:
                            v * 2
                        except:
                            v = 'Rappel'

                    json_v = d.get(v)
                rez[zip(pt)] = json_v

        except IndexError as e:
            lgg.error(e)
            lgg.error('Index Error: Files are probably missing here to complete the results...\n')

        #info_file.append( '%s %s; \t K=%s\n' % (corpus_type, f, K) )

    lgg.debug(''.join(not_finished))
    #lgg.debug(''.join(info_file))
    rez = np.ma.masked_array(rez, np.isnan(rez))
    return rez

def clean_extra_expe(expe, map_parameters):
    for k in expe:
        if k not in map_parameters and k not in [ k for d in _New_Dims for k in d.keys() ] :
            del expe[k]
    return expe

def make_tensor_expe_index(expe, map_parameters):
    ptx = []
    expe = clean_extra_expe(expe, map_parameters)
    for i, o in enumerate(expe.items()):
        k, v = o[0], o[1]
        if v in ( '*',): #wildcar / indexing ...
            ptx.append(slice(None))
        elif k in map_parameters:
            ptx.append(map_parameters[k].index(v))
        elif type(v) is int:
            ptx.append(v)
        elif type(v) is str and ':' in v: #wildcar / indexing ...
            sls = v.split(':')
            sls = [None if not u else int(u) for u in sls]
            ptx.append(slice(*sls))
        else:
            raise ValueError('Unknow data type for tensor forest')

    ptx = tuple(ptx)
    return ptx

