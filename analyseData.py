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
    samplemask = os.path.join(outdir,'sampling_%i.pickle')


    dynamicTopoGP.determineOptimalParams(subset,datafile,paramfile,True)
    dynamicTopoGP.obtainInverse(subset,datafile,paramfile,covfile)
    dynamicTopoGP.obtainSpectrum(subset,datafile,paramfile,specfile,Lmax)
    dynamicTopoGP.calculateMapData(subset,datafile,paramfile,covfile,mapdatafile)
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
