matplot_imported = False
try:
    import matplotlib.pyplot as plt
    matplot_imported = True
except ImportError:
    pass
if matplot_imported:
    import plot_procedure
import ptb
from word_vecs import WordVecsHelper
from options import Options
