import  os
if os.environ.get('DISPLAY') is None:
    # Plot in nil/void
    import matplotlib; matplotlib.use('Agg')
    print("==> Warning : Unable to load DISPLAY, try : `export DISPLAY=:0.0'")
else:
    # Plot config
    __plot_font_size = 14
    import matplotlib.pyplot as plt
    plt.rc('font', size=__plot_font_size)  # controls default text sizes


from pymake.core.types import Spec, Corpus, Model, Script, ExpSpace, ExpVector, ExpTensor, ExpDesign, ExpGroup
from pymake.core.format import ExpeFormat

# Without whoosh fashion...
#from pymake.frontend.io import SpecLoader
#__spec = SpecLoader.get_atoms()

from pymake.core.gramexp import GramExp

