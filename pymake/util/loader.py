# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict
from itertools import groupby
import sys, os, re, importlib, pkgutil, pyclbr, inspect

from pymake.util.utils import parse_file_conf
from pymake import ExpDesign, ExpeFormat

import logging
lgg = logging.getLogger('root')

def is_abstract(cls):
    ABCFLAG = '__abstractmethods__'
    isabc = hasattr(cls, ABCFLAG) or inspect.isabstract(cls)
    return isabc

def get_global_settings():
    dir =  os.path.dirname(os.path.realpath(__file__))
    fn = os.path.join(dir, '..', 'pymake.cfg')
    return parse_file_conf(fn, sep='=')



class PackageWalker(object):
    ''' Import all submodules of a module, recursively. '''

    _default_attr = dict(
        attr_filter = None,
        class_filter = None,
        max_depth = 1,
        prefix=False,
        shrink_module_name=True
    )

    def __init__(self, module_name, **kwargs):
        self.module_name = module_name
        [setattr(self, k, kwargs.get(k, v)) for k,v in self._default_attr.items()]

        self.packages = self._get_packages()

    def __repr__(self):
        return self.packages


    def _get_packages(self, module_name=None,  _depth=0):
        ''' recursive call with regards to depth parameter '''
        if module_name is None:
            module_name = self.module_name
        if self.prefix is True:
            prefix = module_name
        elif self.prefix is False:
            prefix = ''
        elif isinstance(self.prefix, str):
            prefix = self.prefix

        packages = OrderedDict()
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            lgg.debug('package unavailable (%s) : %s' % (e, module_name))
            return packages

        for loader, name, is_pkg in pkgutil.walk_packages(module.__path__):
            submodule_name = module_name + '.' + name
            if is_pkg and _depth < self.max_depth:
                next_depth = _depth + 1
                packages.update(self._get_packages(submodule_name, next_depth))
                continue
            try:
                submodule = importlib.import_module(submodule_name)
            except ImportError as e:
                lgg.debug('package unavailable : %s'%(submodule_name))
                continue

            spck = self.submodule_hook(submodule, prefix, name=name)
            if spck:
                packages.update(spck)

        return packages

    def submodule_hook(self, *args, **kwargs):
        raise NotImplementedError

    def submodule_hook_by_attr(self, submodule, prefix, name=None):
        _packages = pyclbr.readmodule(submodule.__name__)
        packages = {}
        for cls_name, cls  in _packages.items():
            obj = getattr(submodule, cls_name)
            if not hasattr(obj, self.attr_filter) or is_abstract(obj):
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
        _packages = pyclbr.readmodule(submodule.__name__)
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

class SpecLoader(PackageWalker):
    ''' Load specification of design experimentation :
        * Lookup for ExpDesign subclass in the target module.
    '''
    def __init__(self):
        module_name = get_global_settings()['default_spec']
        class_filter = ExpDesign
        super(SpecLoader, self).__init__(module_name, class_filter=class_filter)

    def submodule_hook(self, *args, **kwargs):
        return super(SpecLoader, self).submodule_hook_by_class(*args, **kwargs)

    @staticmethod
    def _default_spec():
        spec = importlib.import_module('pymake.spec.netw')
        for m in dir(spec):
            o = getattr(spec,m)
            if issubclass(o, ExpDesign) and not o is ExpDesign:
                return o()

    @staticmethod
    def default_spec():
        import pymake
        return getattr(pymake, '__spec')


class ModelsLoader(PackageWalker):
    ''' Load models in a modules :
        * Need a `fit' method to be identified as model,
        * lookup for models inside a module by fuzzing the name of the module.
    '''

    @classmethod
    def get_packages(cls, module_name, **kwargs):
        if not 'attr_filter' in kwargs:
            kwargs['attr_filter'] = 'fit'
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

    def submodule_hook(self, *args,  **kwargs):
        return super(ModelsLoader, self).submodule_hook_by_attr(*args,  **kwargs)

class CorpusLoader(PackageWalker):
    def submodule_hook(self, *args, **kwargs):
        raise NotImplemented('todo :scikit-learn loohup')

class ScriptsLoader(PackageWalker):
    def submodule_hook(self, *args,  **kwargs):
        return super(ScriptsLoader, self).submodule_hook_by_class(*args, **kwargs)

    @classmethod
    def get_packages(cls, **kwargs):
        module_name = get_global_settings()['default_scripts']
        if not 'class_filter' in kwargs:
            kwargs['class_filter'] = ExpeFormat
        return cls(module_name, **kwargs).packages


