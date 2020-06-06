import pickle
import numpy as np
import scipy.special as spec
import scipy.integrate as integ
import scipy.optimize as optim
import scipy.linalg as slinalg
import tqdm
import time

########################################
## Andrew Valentine                   ##
## Research School of Earth Sciences  ##
## The Australian National University ##
## June 2020                          ##
########################################


datafile_path = '../Davies_etal_NGeo_2019_Datasets/hoggard/spot_shiptrack.dat'
N_HIGH_ACCURACY = 1160 # Number of 'high-accuracy' spot measurements at the start of Hoggard's datafile
N_SPOT = 2030 # Next 870 points are 'low-accuracy' spot

def dist(lat, lon,latp, lonp):
    '''Compute epicentral angle between (lat,lon) and (lat',lon').
    Inputs:
    lat, lon -- coordinates of point 1 (degrees)
    latp, lonp -- coordinates of point 2 (degrees). May be arrays of shape (N,) containing multiple points.

    Result:
    float or array, epicentral angle (degrees)
    '''
    #if lat==latp and lon==lonp: return 0.
    clat = np.deg2rad(90 - lat)
    clatp = np.deg2rad(90 - latp)
    lon = np.deg2rad(lon)
    lonp= np.deg2rad(lonp)
    arg = (np.cos(clat)*np.cos(clatp) + np.sin(clat)*np.sin(clatp)*np.cos(lon-lonp)).clip(-1,1)
    v = np.where((lat==latp)*(lon==lonp),0.,np.arccos(arg))
    return v

def matern_full(d,params):
    '''Evaluate Matern covariance function.
    Inputs:
    d -- float or array, shape (N, N), distance matrix
    params -- array, shape (3,), hyperparameter vector -> [amplitude , lengthscale, order ]

    Result:
    float or array, shape (N,N), matern function corresponding to each distance.
    '''
    s,rho,v = params
    mask = np.where(d==0,False,True) # Avoids warnings
    out = np.zeros_like(d)
    try:
        out[mask] = s**2 * (2**(1-v)) * (np.sqrt(2*v)*d[mask]/rho)**v * spec.kv(v,np.sqrt(2*v)*d[mask]/rho)/spec.gamma(v)
        out[~mask] = s**2
    except TypeError:
        if d==0:
            out = s**2
        else:
            out = s**2 * (2**(1-v)) * (np.sqrt(2*v)*d/rho)**v * spec.kv(v,np.sqrt(2*v)*d/rho)/spec.gamma(v)
    return out

def real_sph_harm(m,l,theta,phi,radians=True):
    '''Real surface spherical harmonics
    Inputs:
    m,l - integers, spherical harmonic order and degree
    theta - longitudnal coordinate (rad or deg)
    phi - latitudnal coordinate (rad or deg)
    radians - True if angles are given in radians; False if in degrees.

    Result:
    float - the value of the selected spherical harmonic at the point.
    '''
    if not radians:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
    csph = spec.sph_harm(abs(m),l,theta,np.pi/2 - phi)
    return ((-1)**abs(m))*np.where(m==0,1, np.sqrt(2))*np.where(m<=0,np.real(csph),np.imag(csph))

def logLikelihood(dists,data,stds,params):
    '''Evaluate the log-likelihood of a data vector. Assumes a multidimensional Gaussian
    distribution with Matern covariance.
    Inputs:
    dists -- Array, shape (N,N), distances between sample points
    data -- Array, shape (N,), observations at sample points
    stds -- Array, shape (N,), observational uncertainty, as standard deviation of Gaussian at each sample points
    params -- Array, shape (3,), parameters for Matern covariance function

    Result:
    float, log Likelihood.
    '''
    C = matern_full(dists,params)+np.diag(stds**2)
    N = stds.shape[0]
    s,detC = np.linalg.slogdet(C)
    if s<0: return -np.Inf
    return -0.5*(N*np.log(2*np.pi)+detC + data.dot(np.linalg.solve(C,data)))

def loadOptimalParams(file):
    '''Read hyperparameter file'''
    with open(file,'rb') as fp:
        opt_out = pickle.load(fp)
    return opt_out.x

def modelToPower(model):
    '''Convert array of model coefficients into an array of power-per-degree'''
    Lmax = int(np.sqrt(model.shape[-1]+1))-1
    M = np.zeros([model.shape[-1],Lmax])
    i=0
    for lm1 in range(Lmax):
        for m in range(-lm1-1,lm1+2):
            M[i,lm1] = 1
            i+=1
    return (model**2).dot(M)


def loadSpectrum(file):
    '''Read mean and covariance of spherical harmonic coefficients from file'''
    with open(file,'rb') as fp:
        y = pickle.load(fp)
        Sigma = pickle.load(fp)
    return y,Sigma


def Il(l,optimal_params):
    leg = spec.legendre(l)
    return integ.quad(lambda u:matern_full(np.arccos(u),optimal_params)*leg(u),-1,1)
#flm = lambda m,l,th, ph,params: 2*np.pi*Il(l,params)[0]*real_sph_harm(m,l,th,ph,radians=False)#Nl = 1 given definition of spherical harmonics
#glm = lambda m,l,params:2*np.pi*Il(l,params)[0]

def flm_arr_gl(l,theta,phi,params,radians=False):
    out = np.zeros([theta.shape[0],2*l+1])
    integral = Il(l,params)
    for m in range(-l,l+1):
        out[:,m+l] = 2*np.pi*integral[0]*real_sph_harm(m,l,theta,phi,radians)
    return out,2*np.pi*integral[0]

def determineOptimalParams(dataset_type,datafile,outfile,determine_err_correction=False):
    '''
    Perform hyperparameter optimisation for the residual topography dataset.

    Inputs:
    dataset_type - str, "high_accuracy_spot","all_spot", or "spot_shiptrack". Select subset for analysis.
    datafile - Path to residual topography dataset
    outfile - Path to file for output
    determine_err_correction - True/False, determine correction to uncertainty for points without crustal information
    '''
    print("Loading file...")
    data = np.loadtxt(datafile)
    if dataset_type == 'high_accuracy_spot':
        data = data[:N_HIGH_ACCURACY,:] # Select only the high accuracy points
    elif dataset_type == 'all_spot':
        data = data[:N_SPOT,:] # Select all spot points
        if determine_err_correction: data[N_HIGH_ACCURACY:,3]-=0.2 # Undo the hard-coded correction
    elif dataset_type == 'spot_shiptrack':
        if determine_err_correction: data[N_HIGH_ACCURACY:,3]-=0.2 # Undo the hard-coded correction
    else:
        raise ValueError("Unrecognised dataset type")
    print("...done!\n")
    ndata = data.shape[0]
    print("Computing distances...")
    dists = np.zeros([ndata,ndata])
    for i in tqdm.tqdm(range(ndata)):
        dists[:,i] = dist(data[i,0],data[i,1],data[:,0],data[:,1])
    print("...done!\n")

    if determine_err_correction and dataset_type!='high_accuracy_spot': #No error correction to determine for high accuracy spot data
        # We solve for five hyperparameters: (matern amplitude, matern lengthscale, matern order, dataset mean, correction to uncertainties)
        def opfunc(p,dataset):
            value = -logLikelihood(dists,dataset[:,2]-p[3],np.where(np.arange(dataset.shape[0])<N_HIGH_ACCURACY,dataset[:,3],dataset[:,3]+p[4]),p[0:3])
            #print(p,value)
            return value

        print("Determinining optimal parameters...")
        fmin_out = optim.minimize(opfunc,np.array([0.6,0.2,0.5,0.,0.1])
                                     ,args=(data,)
                                     ,bounds=[(0.05,None),(0.01,3.0),(0.01,10),(None,None),(0.,None)]
                                     ,options={'iprint':101})
        print("...done!\n")
        print("Determined parameters:")
        print("     mu = %.5f"%fmin_out.x[3])
        print("  sigma = %.5f"%fmin_out.x[0])
        print("    rho = %.5f"%fmin_out.x[1])
        print("      v = %.5f"%fmin_out.x[2])
        print(" Eshift = %.5f"%fmin_out.x[4])
    else:
        # We solve for four hyperparameters: (matern amplitude, matern lengthscale, matern order, dataset mean)
        def opfunc(p,dataset):
            value = -logLikelihood(dists,dataset[:,2]-p[3],dataset[:,3],p[0:3])
            #print(p,value)
            return value

        print("Determinining optimal parameters...")
        fmin_out = optim.minimize(opfunc,np.array([0.5,0.2,0.5,0.])
                                     ,args=(data,)
                                     ,bounds=[(0.05,None),(0.01,3.0),(0.01,10),(None,None)]
                                     ,options={'iprint':101})
        print("...done!\n")
        print("Determined parameters:")
        print("     mu = %.5f"%fmin_out.x[3])
        print("  sigma = %.5f"%fmin_out.x[0])
        print("    rho = %.5f"%fmin_out.x[1])
        print("      v = %.5f"%fmin_out.x[2])
    with open(outfile,'wb') as fp:
        pickle.dump(fmin_out,fp)
    print("Output stored in: %s"%outfile)

def obtainInverse(dataset_type,datafile,paramfile, outfile):
    '''
    Compute inverse covariance matrix for GP based on observed data and chosen
    hyperparameters.

    Inputs:
    dataset_type - str, "high_accuracy_spot","all_spot" or "spot_shiptrack"
    datafile - Path to residual topography dataset
    paramfile - File containing hyperparameters
    outfile - File to store inverse covariance matrix
    '''
    print("Loading files...")
    data = np.loadtxt(datafile)
    if dataset_type == 'high_accuracy_spot':
        data = data[:N_HIGH_ACCURACY,:]
    elif dataset_type == 'all_spot':
        data = data[:N_SPOT,:]
    elif dataset_type == 'spot_shiptrack':
        pass
    else:
        raise ValueError("Unrecognised dataset type")
    ndata = data.shape[0]
    opt_out=loadOptimalParams(paramfile)
    print("...done!\n")

    optimal_params = opt_out[0:3]
    data[:,2] -= opt_out[3]
    if len(opt_out)==5: # Params include error correction
        print("Revised shiptrack error correction: %.3f"%opt_out[4])
        data[N_HIGH_ACCURACY:,3]+=opt_out[4]-0.2 # Apply new correction instead of hard-coded.
    print("Computing distances...")
    dists = np.zeros([ndata,ndata])
    for i in tqdm.tqdm(range(ndata)):
        dists[:,i] = dist(data[i,0],data[i,1],data[:,0],data[:,1])
    print("...done!\n")

    print("Building covariance matrix...")
    C = matern_full(dists,optimal_params)+np.diag(data[:,3]**2)
    print("...done!\n")
    del( dists) #Save a bit of space

    iC = np.linalg.inv(C)



    with open(outfile,'wb') as fp:
        pickle.dump(iC,fp)

def obtainSpectrum(dataset_type,datafile,paramfile,outfile,Lmax):
    '''
    Compute inverse covariance matrix for GP based on observed data and chosen
    hyperparameters.

    Inputs:
    dataset_type - str, "high_accuracy_spot","all_spot" or "spot_shiptrack"
    datafile - Path to residual topography dataset
    paramfile - File containing hyperparameters
    outfile - File to store spherical harmonic coefficients
    Lmax - maximum spherical harmonic degree to compute
    '''
    print("Loading files...")
    data = np.loadtxt(datafile)
    if dataset_type == 'high_accuracy_spot':
        data = data[:N_HIGH_ACCURACY,:]
    elif dataset_type == 'all_spot':
        data = data[:N_SPOT,:]
    elif dataset_type == 'spot_shiptrack':
        pass
    else:
        raise ValueError("Unrecognised dataset type")
    ndata = data.shape[0]
    opt_out = loadOptimalParams(paramfile)
    print("...done!\n")

    optimal_params = opt_out[0:3]
    data[:,2] -= opt_out[3]
    if len(opt_out)==5:
        print("Revised shiptrack error correction: %.3f"%opt_out[4])
        data[N_HIGH_ACCURACY:,3]+=opt_out[4]-0.2
    print("Computing distances...")
    dists = np.zeros([ndata,ndata])
    for i in tqdm.tqdm(range(ndata)):
        dists[:,i] = dist(data[i,0],data[i,1],data[:,0],data[:,1])
    print("...done!\n")

    print("Building covariance matrix...")
    C = matern_full(dists,optimal_params)+np.diag(data[:,3]**2)
    print("...done!\n")
    del( dists) #Save a bit of space
    ncomp = (Lmax+1)**2
    ll = np.zeros(ncomp,dtype='int')
    mm = np.zeros(ncomp,dtype='int')
    i = 0
    for l in range(Lmax+1):
        for m in range(-l,l+1):
            ll[i] = l
            mm[i] = m
            i+=1
    print("Computing Legendre expansion...")
    flms = np.zeros([ndata,ncomp])
    gls = np.zeros([Lmax+1])
    i = 0
    for l in range(Lmax+1):
        flms[:,i:i+(2*l)+1],gls[l] = flm_arr_gl(l,data[:,1],data[:,0],optimal_params)
        i+=(2*l)+1
    flms[:,0] = data[:,2]
    print("...done!\n")

    print("Solving linear system...")

    t = time.time()
    sol = slinalg.solve(C,flms,assume_a='pos')
    solvetime = time.time()-t
    print("...done!\n")
    coeff = flms[:,1:].T.dot(sol)
    with open(outfile,'wb') as fp:
        pickle.dump(coeff[:,0],fp)
        pickle.dump(np.diag([gls[l] for l in ll[1:]])-coeff[:,1:],fp)
        pickle.dump(solvetime,fp)


def interp(lat,lon,dataset,iK,optimal_params):
    k = matern_full(dist(lat,lon,dataset[:,0],dataset[:,1]),optimal_params)
    return k.dot(iK).dot(dataset[:,2]),optimal_params[0]**2 - k.dot(iK).dot(k)

def calculateMapData(dataset_type,datafile,paramfile,covfile,outfile,nlats=90,nlons=180):
    '''
    Evaluate GP interpolation on regular grid to allow generation of a map.

    Inputs:
    dataset_type - str, "high_accuracy_spot","all_spot" or "spot_shiptrack"
    datafile - Path to residual topography dataset
    paramfile - File containing hyperparameters
    covfile - File containing inverse covariance matrix
    outfile - File to store map data
    nlats,nlons - Number of latitude/longitude points to compute
    '''
    print("Loading files...")
    data = np.loadtxt(datafile)
    if dataset_type == 'high_accuracy_spot':
        data = data[:N_HIGH_ACCURACY,:]
    elif dataset_type == 'all_spot':
        data = data[:N_SPOT,:]
    elif dataset_type == 'spot_shiptrack':
        pass
    else:
        raise ValueError("Unrecognised dataset type")
    opt_out = loadOptimalParams(paramfile)
    optimalParams = opt_out[0:3]
    data[:,2] -= opt_out[3]
    if len(opt_out)==5:
        print("Revised shiptrack error correction: %.3f"%opt_out[4])
        data[N_HIGH_ACCURACY:,3]+=opt_out[4]-0.2
    with open(covfile,'rb') as fp:
        iK = pickle.load(fp)
    print("...done!")
    lats = np.linspace(-90,90,nlats)
    lons = np.linspace(-180,180,nlons)
    mean = np.zeros([nlats,nlons])
    variance = np.zeros([nlats,nlons])

    t = tqdm.tqdm(total=nlats*nlons)
    for i,lat in enumerate(lats):
        for j,lon in enumerate(lons):
            mean[i,j],variance[i,j] = interp(lat,lon,data,iK,optimalParams)
            t.update(1)
    t.close()
    dkl = 0.5 * (mean**2/optimalParams[0]**2 + variance/optimalParams[0]**2 - np.log(variance/optimalParams[0]**2)-1)
    with open(outfile,'wb') as fp:
        pickle.dump(mean,fp)
        pickle.dump(variance,fp)
        pickle.dump(dkl,fp)
        pickle.dump(lats,fp)
        pickle.dump(lons,fp)
def calculateWhereToSample(dataset_type,datafile,paramfile,covfile,sphfile,mapdatafile,outfile_mask,llist=[2,5,10,15,20,30],obsErr=0.1):
    '''
    Evaluate GP interpolation on regular grid to allow generation of a map.

    Inputs:
    dataset_type - str, "high_accuracy_spot","all_spot" or "spot_shiptrack"
    datafile - Path to residual topography dataset
    paramfile - File containing hyperparameters
    covfile - File containing inverse covariance matrix
    sphfile - File containing spherical harmonic coefficients
    mapdatafile - File containing map dataset
    outfile_mask - String with %i placeholder to store map at degree l
    llist - List of spherical harmonic degrees of interest
    obsErr - Assumed standard error in putative observation
    '''
    print("Loading files...")
    data = np.loadtxt(datafile)
    if dataset_type == 'high_accuracy_spot':
        data = data[:N_HIGH_ACCURACY,:]
    elif dataset_type == 'all_spot':
        data = data[:N_SPOT,:]
    elif dataset_type == 'spot_shiptrack':
        pass
    else:
        raise ValueError("Unrecognised dataset type")
    opt_out = loadOptimalParams(paramfile)
    optimalParams = opt_out[0:3]
    data[:,2] -= opt_out[3]
    if len(opt_out)==5:
        print("Revised shiptrack error correction: %.3f"%opt_out[4])
        data[N_HIGH_ACCURACY:,3]+=opt_out[4]-0.2
    ndata = data.shape[0]

    print("Loading spherical harmonic distribution...")
    with open(sphfile,'rb') as fp:
        yFull = pickle.load(fp)
        SigmaFull = pickle.load(fp)
    print("...done!\n")

    print("Loading map data...")
    with open(mapdatafile,'rb') as fp:
        mean = pickle.load(fp)
        variance = pickle.load(fp)
        dkl = pickle.load(fp)
        lats = pickle.load(fp)
        lons = pickle.load(fp)
    nlats = lats.shape[0]
    nlons = lons.shape[0]
    print("...done!\n")
    print("Loading inverse matrix...")
    with open(covfile,'rb') as fp:
        Q = pickle.load(fp)
    print("...done!")
    t = tqdm.tqdm(total=nlats*nlons*len(llist))
    for l in llist:
        integral = 2*np.pi*Il(l,optimalParams)[0]
        il = sum([2*ll+1 for ll in range(1,l)])
        y = yFull[il:il+2*l+1]
        Sigma = SigmaFull[il:il+2*l+1,il:il+2*l+1]

        fl = lambda lat,lon:integral*real_sph_harm(np.arange(-l,l+1),l,lon,lat,radians=False)
        F = np.zeros([2*l+1,ndata])
        for i in range(ndata):
            F[:,i] = fl(data[i,0],data[i,1])
        FQ = F.dot(Q)
        zl = lambda lat, lon: fl(lat,lon)-FQ.dot(matern_full(dist(lat,lon,data[:,0],data[:,1]),optimalParams))


        dkl_min = np.zeros_like(dkl)

        for i,lat in enumerate(lats):
            for j,lon in enumerate(lons):
                alpha = 1/(variance[i,j]+obsErr**2)
                z = zl(lat,lon)
                SigmaP = Sigma - alpha*np.outer(z,z)
                invSigmaP = np.linalg.inv(SigmaP)
                sgnSigma, ldSigma = np.linalg.slogdet(Sigma)
                sgnSigmaP,ldSigmaP = np.linalg.slogdet(SigmaP)
                if sgnSigma<0 or sgnSigmaP<0: raise ValueError('Negative determinant?')
                dkl_min[i,j] = 0.5*(np.trace(invSigmaP.dot(Sigma))-(2*l+1) +(ldSigmaP-ldSigma))
                dkl[i,j] = dkl_min[i,j]+0.5*alpha**2 * variance[i,j]*z.dot(invSigmaP).dot(z)
                t.update(1)
        with open(outfile_mask%l,'wb') as fp:
            pickle.dump(dkl_min,fp)
            pickle.dump(dkl,fp)
    t.close()
