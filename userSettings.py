# Edit these settings as necessary. You may wish to run the command
# `git update-index --skip-worktree userSettings.py` to prevent your changes
# being tracked by git.
#
# Location of raw data file
datafile = './Davies_etal_NGeo_2019_Datasets/hoggard/spot_shiptrack.dat'
# If you have just obtained this package from github you may need to run the
# command `git submodule update --init --recursive` to also pull down a copy
# of the data.

# Base location for outputs
# The code will create subdirectories under this for each data subset and
# for the resulting figures
outputdir = '.'

# Run analysis for the subset of 1160 high-accuracy spot points?
DO_HIGH_ACCURACY_SPOT = True

# Run analysis for the subset of all 2030 spot points?
DO_ALL_SPOT = False

# Run analysis for complete database?
# N.B. This requires significant computational resources
#      (~36 hours on quad-core 3.2GHz Intel i5 with 16Gb RAM)
#      Running this on a machine with less than 16Gb RAM is
#      not recommended.
DO_SPOT_SHIP = False

# Test predictive performance against a synthetic dataset?
DO_SYNTHETIC = False
# Location of data file containing the 'observations' (assumed to match
# a 'high-accuracy' dataset, i.e. only first 1160 points used)
syn_file_sampled = './Davies_etal_NGeo_2019_Datasets/models/with_shallow_structure/with_shallow_structure_spot.dat'
# Data file against which to test predictive performance
syn_file_full = './oceans.dat' #'./Davies_etal_NGeo_2019_Datasets/models/with_shallow_structure/with_shallow_structure_full.dat'


#How many random samples should be used for computing statistics? (Larger = slower)
N_RANDOM_SAMPLES = 100000

# Should maps be plotted with continuous or quantised colour scales?
CONTINUOUS_COLOURSCALES = False

# Which figures should be generated? List of figure numbers. Note that this selection
# may not be fully honoured: figs 6 & 7 cannot be generated independently, and
# setting TABLE_DATA = True below will force generation of Figs. 2, 6 & 7.
FIGURES = [9]

# When generating figures, should plt.show() be called?
# Note that this may cause the interpreter to block and
# wait for the figure to be closed
PLT_SHOW = False

# During figure generation, should the information presented in tables within
# the paper also be calculated and displayed? (may be slow)
TABLE_DATA = False

# Generate the data files provided as 'supplementary material'?
SUPPLEMENTARY_DATA = False
