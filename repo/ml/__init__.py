import sys

try:
    # backwards compatibility
    from ml import model
    sys.modules['pymake.model'] = model
except:
    pass
