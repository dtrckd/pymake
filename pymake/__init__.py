
try:
    basestring = basestring
except NameError:
    #python3
    basestring = (str, bytes)
