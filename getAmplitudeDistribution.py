import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pickle
import scipy.special as spec
import tqdm
from dynamicTopoGP import dist, matern_full, loadOptimalParams,loadSpectrum,real_sph_harm
def interp(lat,lon,dataset,iK,optimal_params):
    k = matern_full(dist(lat,lon,dataset[:,0],dataset[:,1]),optimal_params)
    return k.dot(iK).dot(dataset[:,2]),optimal_params[0]**2 - k.dot(iK).dot(k)
def sph_map_vec(nlats,nlons,lmax=30):
    ngrid = nlats*nlons
    i=0
    ll = np.zeros([ngrid,2])
    for ilat,lat in enumerate(np.linspace(-90,90,nlats)):
        for ilon,lon in enumerate(np.linspace(-180,180,nlons)):
            ll[i,:] = lat,lon
            i+=1
    out = []
    t = tqdm.tqdm(total=(lmax+1)**2 - 1)
    for l in range(1,lmax+1):
        for m in range(-l,l+1):
            out+=[real_sph_harm(m,l,ll[:,1].reshape(nlats,nlons),ll[:,0].reshape(nlats,nlons),radians=False)]
            t.update(1)
    t.close()
    return ll,np.array(out).reshape((-1,ngrid))

N_HIGH_ACCURACY = 1160
# HIGH_ACCURACY_ONLY=False
# type = 'spot'
# filename_modifier='_determine_uncertainty'
# dataFile = '../Davies_etal_NGeo_2019_Datasets/hoggard/%s.dat'%type
# inverseFile = 'inverseCov_%s%s.pickle'%(type,filename_modifier)
# paramFile = 'optimal_params_%s%s.pickle'%(type,filename_modifier)
# outputFile = 'map_%s%s.pdf'%(type,filename_modifier)
# CALCULATE_MAP_DATA = False
# mapdataFile = 'mapdata_%s%s.pickle'%(type,filename_modifier)
# data = np.loadtxt(dataFile)
# if HIGH_ACCURACY_ONLY: data = data[:N_HIGH_ACCURACY,:]
#
#
# opt_out = loadOptimalParams(paramFile)
# optimalParams = opt_out[0:3]
# data[:,2] -= opt_out[3]
# if len(opt_out)==5:
#     print("Revised shiptrack error correction: %.3f"%opt_out[4])
#     data[N_HIGH_ACCURACY:,3]+=opt_out[4]-0.2
nlats = 45
nlons = 90
ngrid = nlats*nlons
nrand= 100000
if False:
    ll,smv = sph_map_vec(nlats,nlons)
    plt.rcParams['font.size']=12
    fig = plt.figure(figsize=(8,10))
    x0=0.05
    y0=0.05
    cy=0.03
    ysp=0.04


    i0 = 0
    i1 = 15
    my=(1-(2*y0)-cy)/3 - ysp
    print("")
    for i,(type,filename_modifier) in enumerate([['spot','_high_accuracy'],
                                                 ['spot','_determine_uncertainty'],
                                                 ['spot_shiptrack','_determine_uncertainty']]):
        y,Sigma = loadSpectrum('sphcoeff_%s%s.pickle'%(type,filename_modifier))
        mapdata = y[i0:i1].dot(smv[i0:i1,:]).reshape(nlats,nlons)
        d = abs(np.random.multivariate_normal(y[i0:i1],Sigma[i0:i1,i0:i1],size=nrand).dot(smv[i0:i1,:])).max(1)
        srt = np.argsort(d)
        print(type,filename_modifier,d.mean(),d.std(),d[srt[int(nrand/200)]],d[srt[int(199*nrand/200)]],d[srt[int(25*nrand/100)]],d[srt[int(75*nrand/100)]])

        ax = fig.add_axes((x0,y0+cy+(2-i)*(ysp+my)+ysp,1-2*x0,my),projection=ccrs.Robinson())
        pc = ax.pcolor(ll[:,1].reshape(nlats,nlons),ll[:,0].reshape(nlats,nlons),mapdata,cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree())
        pc.set_edgecolor('face')
        co = ax.contour(ll[:,1].reshape(nlats,nlons),ll[:,0].reshape(nlats,nlons),y[i0:i1].dot(smv[i0:i1,:]).reshape(nlats,nlons),levels=np.arange(-1.4,1.5,0.2),transform=ccrs.PlateCarree(),colors='k')
        pc.set_clim(-.8,.8)
        ax.coastlines()
        ax.text(0.9,0,'%.2f/%.2f'%(mapdata.min(),mapdata.max()),transform=ax.transAxes)
        ax.text(0.0,0.95,'(%s)'%(['a','b','c'][i]),transform=ax.transAxes)
    ax = fig.add_axes((x0+0.25,y0,1-2*(x0+0.25),cy))
    plt.colorbar(pc,cax=ax,orientation='horizontal',ticks=[-.8,-.4,0,.4,.8],label="Residual topography at degrees 1-3, km")
    ax.xaxis.set_label_position('top')
    plt.savefig('1to3.pdf')
    plt.show()
if True:
    def bands(x):
        s = np.argsort(x)
        n=x.shape[0]
        return x[s[int(n/200)]],x[s[int(199*n/200)]],x[s[int(n/4)]],x[s[int(3*n/4)]]
    HIGH_ACCURACY_ONLY=False
    type = 'spot_shiptrack'
    filename_modifier='_determine_uncertainty'
    dataFile = '../Davies_etal_NGeo_2019_Datasets/hoggard/%s.dat'%type
    inverseFile = 'inverseCov_%s%s.pickle'%(type,filename_modifier)
    paramFile = 'optimal_params_%s%s.pickle'%(type,filename_modifier)
    outputFile = 'map_%s%s.pdf'%(type,filename_modifier)
    CALCULATE_MAP_DATA = False
    mapdataFile = 'mapdata_%s%s.pickle'%(type,filename_modifier)
    data = np.loadtxt(dataFile)
    if HIGH_ACCURACY_ONLY: data = data[:N_HIGH_ACCURACY,:]


    opt_out = loadOptimalParams(paramFile)
    optimalParams = opt_out[0:3]
    data[:,2] -= opt_out[3]
    if len(opt_out)==5:
        print("Revised shiptrack error correction: %.3f"%opt_out[4])
        data[N_HIGH_ACCURACY:,3]+=opt_out[4]-0.2


    with open(inverseFile,'rb') as fp:
        iK = pickle.load(fp)


    k1 = np.zeros([ngrid,ngrid])
    k2 = np.zeros([ngrid,data.shape[0]])
    ll = np.zeros([ngrid,2])
    i = 0
    print("Generating locations")
    for ilat,lat in tqdm.tqdm(enumerate(np.linspace(-90,90,nlats))):
        for ilon,lon in enumerate(np.linspace(-180,180,nlons)):
            ll[i,:] = lat,lon
            i+=1
    print("Computing covariance")
    for i in tqdm.tqdm(range(ngrid)):
        k1[:,i] = matern_full(dist(ll[i,0],ll[i,1],ll[:,0],ll[:,1]),optimalParams)
        k2[i,:] = matern_full(dist(ll[i,0],ll[i,1],data[:,0],data[:,1]),optimalParams)



    print("Generating random maps...")
    rmaps = np.random.multivariate_normal(k2.dot(iK).dot(data[:,2]),k1 - k2.dot(iK).dot(k2.T),size=nrand)
    print (bands(abs(rmaps).max(1)))
