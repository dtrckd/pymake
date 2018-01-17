import  os
if os.environ.get('DISPLAY') is None:
    # plot in nil/void
    import matplotlib; matplotlib.use('Agg')
    print('==> Warning : Unable to load DISPLAY')
    print("To force a display try : `export DISPLAY=:0.0")
else:
    __plot_font_size = 14

    ### PLOTLIB CONFIG
    import matplotlib.pyplot as plt
    plt.rc('font', size=__plot_font_size)  # controls default text sizes


from pymake.core.format import Spec, Corpus, Model, Script, ExpSpace, ExpVector, ExpTensor, ExpeFormat, ExpDesign, ExpGroup

# Without whoosh fashion...
#from pymake.frontend.frontend_io import SpecLoader
#__spec = SpecLoader.get_atoms()

from pymake.core.gramexp import GramExp

