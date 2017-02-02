# -*- coding: utf-8 -*-

# __future__
try:
    basestring = basestring
except NameError:
    #python3
    basestring = (str, bytes)


#from zymake import Zymake
