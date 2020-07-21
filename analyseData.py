import dynamicTopoGP
import os
import sys
import userSettings
########################################
## Andrew Valentine                   ##
## Research School of Earth Sciences  ##
## The Australian National University ##
## June 2020                          ##
########################################


Lmax = 30
outputdir = os.path.abspath(userSettings.outputdir)
if not os.path.exists(outputdir):
    print("Requested output directory: appears not to exist. Please create")
    print(outputdir)
    print("or modify 'outputdir' in userSettings.py as necessary.")
    sys.exit(1)
datafile = os.path.abspath(userSettings.datafile)
if not os.path.exists(datafile):
    print("Data file not found at")
    print(datafile)
    print("Please check and modify 'datafile' in userSettings.py if necessary.")
    sys.exit(1)
if userSettings.DO_SYNTHETIC:
    synfile_sampled = os.path.abspath(userSettings.syn_file_sampled)
    synfile_full = os.path.abspath(userSettings.syn_file_full)
    if not os.path.exists(synfile_sampled):
        print("Synthetic dataset not found at")
        print(synfile_sampled)
        print("Please check and modify 'syn_file_sampled' in userSettings.py if necessary.")
        sys.exit(1)
    if not os.path.exists(synfile_full):
        print("Synthetic dataset not found at")
        print(synfile_full)
        print("Please check and modify 'syn_file_full' in userSettings.py if necessary.")
        sys.exit(1)
if userSettings.DO_HIGH_ACCURACY_SPOT:
    print("Analysing high-accuracy spot data")
    subset = 'high_accuracy_spot'
    outdir = os.path.join(outputdir,subset)
    if not os.path.exists(outdir): os.mkdir(outdir)
    paramfile = os.path.join(outdir,'optimal_params.pickle')
    covfile = os.path.join(outdir,'inverseCov.pickle')
    specfile = os.path.join(outdir,'sphcoeff.pickle')
    mapdatafile = os.path.join(outdir,'mapdata.pickle')
    samplemask = os.path.join(outdir,'sampling_%i.pickle')


    dynamicTopoGP.determineOptimalParams(subset,datafile,paramfile,False)
    dynamicTopoGP.obtainInverse(subset,datafile,paramfile,covfile)
    dynamicTopoGP.obtainSpectrum(subset,datafile,paramfile,specfile,Lmax)
    dynamicTopoGP.calculateMapData(subset,datafile,paramfile,covfile,mapdatafile)
    # We only make maps of where to sample for high accuracy spot.
    dynamicTopoGP.calculateWhereToSample(subset,datafile,paramfile,covfile,specfile,mapdatafile,samplemask)
else:
    print("Analysis of high-accuracy spot data is switched off in userSettings.py")
if userSettings.DO_ALL_SPOT:
    print("Analysing all spot data")
    subset = 'all_spot'
    outdir = os.path.join(outputdir,subset)
    if not os.path.exists(outdir): os.mkdir(outdir)
    paramfile = os.path.join(outdir,'optimal_params.pickle')
    covfile = os.path.join(outdir,'inverseCov.pickle')
    specfile = os.path.join(outdir,'sphcoeff.pickle')
    mapdatafile = os.path.join(outdir,'mapdata.pickle')
    tradeofffile = os.path.join(outdir,'likelihood.pickle')
    samplemask = os.path.join(outdir,'sampling_%i.pickle')


    dynamicTopoGP.determineOptimalParams(subset,datafile,paramfile,True)
    dynamicTopoGP.obtainInverse(subset,datafile,paramfile,covfile)
    dynamicTopoGP.obtainSpectrum(subset,datafile,paramfile,specfile,Lmax)
    dynamicTopoGP.calculateMapData(subset,datafile,paramfile,covfile,mapdatafile)
    if userSettings.DO_TRADEOFF_ANALYSIS: dynamicTopoGP.generateLikelihoodGrids(subset,datafile,paramfile,[.05,.05,.05,.3,.08],tradeofffile)
else:
    print("Analysis of all spot data is switched off in userSettings.py")
if userSettings.DO_SPOT_SHIP:
    print("Analysing spot and shiptrack data")
    subset = 'spot_shiptrack'
    outdir = os.path.join(outputdir,subset)
    if not os.path.exists(outdir): os.mkdir(outdir)
    paramfile = os.path.join(outdir,'optimal_params.pickle')
    covfile = os.path.join(outdir,'inverseCov.pickle')
    specfile = os.path.join(outdir,'sphcoeff.pickle')
    mapdatafile = os.path.join(outdir,'mapdata.pickle')
    samplemask = os.path.join(outdir,'sampling_%i.pickle')


    dynamicTopoGP.determineOptimalParams(subset,datafile,paramfile,True)
    dynamicTopoGP.obtainInverse(subset,datafile,paramfile,covfile)
    dynamicTopoGP.obtainSpectrum(subset,datafile,paramfile,specfile,Lmax)
    dynamicTopoGP.calculateMapData(subset,datafile,paramfile,covfile,mapdatafile)
else:
    print("Analysis of spot and shiptrack data is switched off in userSettings.py")
if userSettings.DO_SYNTHETIC:
    print("Analysing synthetic dataset")

    subset = 'high_accuracy_spot' # Only use the first 1160 points in synthetic file
    outdir = os.path.join(outputdir,'synthetic')
    subsetdir = os.path.join(outputdir,subset)
    if not os.path.exists(outdir): os.mkdir(outdir)
    paramfile = os.path.join(subsetdir,'optimal_params.pickle')
    covfile = os.path.join(subsetdir,'inverseCov.pickle')
    perffile = os.path.join(outdir,'performance.pickle')
    if not os.path.exists(paramfile):
        print("File not found: %s"%paramfile)
        print("Analysis for %s subset must be run before attempting synthetic test."%subset)
        sys.exit(1)
    if not os.path.exists(covfile):
        print("File not found: %s"%covfile)
        print("Analysis for %s subset must be run before attempting synthetic test."%subset)
        sys.exit(1)
    dynamicTopoGP.testPerformance(subset,synfile_sampled,paramfile,covfile,synfile_full,perffile)
else:
    print("Analysis of synthetic dataset is switched off in userSettings.py")
print("Analysis complete.")
