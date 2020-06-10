import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pickle
from dynamicTopoGP import N_SPOT,N_HIGH_ACCURACY,loadOptimalParams,matern_full,loadSpectrum,modelToPower,interp,real_sph_harm,loadData,dist
import os
import sys
import userSettings

########################################
## Andrew Valentine                   ##
## Research School of Earth Sciences  ##
## The Australian National University ##
## June 2020                          ##
########################################


color_spot_1 = 'darkorange'
color_spot_2 = 'gold'
color_all_1 = 'firebrick'
color_all_2 = 'salmon'
color_spot_ship_1 = 'royalblue'
color_spot_ship_2 ='lightblue'

# Settings overriden from userSettings.py if run as script
SHOWFIGS=True
SHOWTABLES=True

def toggleFigure(n):
    try:
        userSettings.FIGURES.index(n)
        return True
    except ValueError:
        return False

def bands(powers):
    '''Report median and central 99% and 50% ranges of dataset'''
    N,M = powers.shape
    N1 = int(N/200)
    N25 = int(N/4)
    N50 = int(N/2)
    N75 = int(3*N/4)
    N99 = int(199*N/200)
    bands = np.zeros([5,M])
    for i in range(M):
        order = np.argsort(powers[:,i])
        bands[0,i] = powers[order[N1],i]
        bands[1,i] = powers[order[N25],i]
        bands[2,i] = powers[order[N50],i]
        bands[3,i] = powers[order[N75],i]
        bands[4,i] = powers[order[N99],i]
    return bands
def sph_map_vec(nlats,nlons,lmax=30):
    ngrid = nlats*nlons
    i=0
    ll = np.zeros([ngrid,2])
    for ilat,lat in enumerate(np.linspace(-90,90,nlats)):
        for ilon,lon in enumerate(np.linspace(-180,180,nlons)):
            ll[i,:] = lat,lon
            i+=1
    out = []
    for l in range(1,lmax+1):
        for m in range(-l,l+1):
            out+=[real_sph_harm(m,l,ll[:,1].reshape(nlats,nlons),ll[:,0].reshape(nlats,nlons),radians=False)]
    return ll,np.array(out).reshape((-1,ngrid))

def plotDatasets(datafile,outfile):
    plt.rcParams['font.size']=14
    data = np.loadtxt(datafile)
    data[N_HIGH_ACCURACY:,3]-=0.2
    max_abs_data=2.6
    max_err_data = 0.6
    fig = plt.figure(figsize=(11.5,10))

    c1 = 0.025
    csp = 0.025
    cw = (1 - (2*c1) - csp)/2
    c2 = c1+cw+csp
    cbhw=0.15

    r0 = 0.05
    r1 = 0.05
    cbh = 0.03
    rsp = 0.025
    rh = (1 - r0 - cbh -5*rsp)/3
    r1 = r0+cbh+2*rsp
    r2 = r1+rh+rsp
    r3 = r1+2*(rh+rsp)

    ax_data_ha = fig.add_axes((c1,r3,cw,rh),projection=ccrs.Robinson())
    ax_err_ha = fig.add_axes((c2,r3,cw,rh),projection=ccrs.Robinson())
    ax_data_spot = fig.add_axes((c1,r2,cw,rh),projection=ccrs.Robinson())
    ax_err_spot = fig.add_axes((c2,r2,cw,rh),projection=ccrs.Robinson())
    ax_data_ship = fig.add_axes((c1,r1,cw,rh),projection=ccrs.Robinson())
    ax_err_ship = fig.add_axes((c2,r1,cw,rh),projection=ccrs.Robinson())
    ax_cb_data = fig.add_axes((0.25-cbhw,r0,2*cbhw,cbh))
    ax_cb_err = fig.add_axes((0.75-cbhw,r0,2*cbhw,cbh))
    for ax in [ax_data_ha,ax_err_ha,ax_data_spot,ax_err_spot,ax_data_ship,ax_err_ship]:
        ax.set_global()
        ax.coastlines()
    s = ax_data_ha.scatter(data[:N_HIGH_ACCURACY,1],data[:N_HIGH_ACCURACY,0],c=data[:N_HIGH_ACCURACY,2],s=0.5,transform=ccrs.PlateCarree(),cmap=plt.cm.coolwarm)
    s.set_clim(-max_abs_data,max_abs_data)
    ax_data_ha.text(0,0.95,'(a)',transform=ax_data_ha.transAxes)
    s = ax_err_ha.scatter(data[:N_HIGH_ACCURACY,1],data[:N_HIGH_ACCURACY,0],c=data[:N_HIGH_ACCURACY,3],s=0.5,transform=ccrs.PlateCarree(),cmap=plt.cm.cubehelix_r)
    s.set_clim(0,max_err_data)
    ax_err_ha.text(0,0.95,'(b)',transform=ax_err_ha.transAxes)

    s = ax_data_spot.scatter(data[N_HIGH_ACCURACY:N_SPOT,1],data[N_HIGH_ACCURACY:N_SPOT,0],c=data[N_HIGH_ACCURACY:N_SPOT,2],s=0.5,transform=ccrs.PlateCarree(),cmap=plt.cm.coolwarm)
    s.set_clim(-max_abs_data,max_abs_data)
    ax_data_spot.text(0,0.95,'(c)',transform=ax_data_spot.transAxes)
    s = ax_err_spot.scatter(data[N_HIGH_ACCURACY:N_SPOT,1],data[N_HIGH_ACCURACY:N_SPOT,0],c=data[N_HIGH_ACCURACY:N_SPOT,3],s=0.5,transform=ccrs.PlateCarree(),cmap=plt.cm.cubehelix_r)
    s.set_clim(0,max_err_data)
    ax_err_spot.text(0,0.95,'(d)',transform=ax_err_spot.transAxes)

    s = ax_data_ship.scatter(data[N_SPOT:,1],data[N_SPOT:,0],c=data[N_SPOT:,2],s=0.5,transform=ccrs.PlateCarree(),cmap=plt.cm.coolwarm)
    s.set_clim(-max_abs_data,max_abs_data)
    fig.colorbar(s,cax=ax_cb_data,orientation='horizontal')
    ax_data_ship.text(0,0.95,'(e)',transform=ax_data_ship.transAxes)
    ax_cb_data.set_xlabel("Measured residual topography, km")
    ax_cb_data.xaxis.set_label_position('top')
    ax_err_ship.text(0,0.95,'(f)',transform=ax_err_ship.transAxes)
    s = ax_err_ship.scatter(data[N_SPOT:,1],data[N_SPOT:,0],c=data[N_SPOT:,3],s=0.5,transform=ccrs.PlateCarree(),cmap=plt.cm.cubehelix_r)
    s.set_clim(0,max_err_data)
    fig.colorbar(s,cax=ax_cb_err,orientation='horizontal')
    ax_cb_err.set_xlabel("Measurement uncertainty, km")
    ax_cb_err.xaxis.set_label_position('top')
    plt.savefig(outfile)
    if SHOWFIGS: plt.show()


def plotMap(mapdatafile,outfile):
    plt.rcParams['font.size']=12
    with open(mapdatafile,'rb') as fp:
        mean = pickle.load(fp)
        variance = pickle.load(fp)
        dkl = pickle.load(fp)
        lats = pickle.load(fp)
        lons = pickle.load(fp)
    nlats = lats.shape[0]
    nlons = lons.shape[0]

    nlevels=100
    max_mean = 2.3 #2.1
    mean_levels = np.linspace(-max_mean,max_mean,nlevels)

    max_std = 0.65#max((variance**0.5).max(),(vg_ns**0.5).max(),(vg_ws**0.5).max())
    std_levels = np.linspace(0,max_std,nlevels)

    max_dkl = 4#3 #max(dkl_hs.max(),dkl_ns.max(),dkl_ws.max())
    dkl_levels = np.linspace(0,max_dkl,nlevels)


    fig = plt.figure(figsize=(8,10))
    left = 0.01
    cb = 0.85
    cbw=0.05
    cbh = 0.2
    row1 = 0.65
    row2 = 0.35
    row3 = 0.05
    width=0.85
    height=0.27
    ax_mean = fig.add_axes((left,row1,width,height),projection=ccrs.Robinson())
    ax_std = fig.add_axes((left,row2,width,height),projection=ccrs.Robinson())
    ax_dkl = fig.add_axes((left,row3,width,height),projection=ccrs.Robinson())

    for ax in [ax_mean,ax_std]:
        ax.set_global()
        ax.coastlines()
    for ax in [ax_dkl]:
        ax.set_global()
        ax.coastlines(color='white')

    ax_mean_cb = fig.add_axes((cb,row1+(height-cbh)/2,cbw,cbh))
    ax_std_cb = fig.add_axes((cb,row2+(height-cbh)/2,cbw,cbh))
    ax_dkl_cb = fig.add_axes((cb,row3+(height-cbh)/2,cbw,cbh))

    sc_mean = ax_mean.contourf(lons,lats,mean,levels=mean_levels,cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree())
    for c in sc_mean.collections:
        c.set_edgecolor('face')
    sc_std = ax_std.contourf(lons,lats,variance**0.5,levels=std_levels,cmap=plt.cm.cubehelix_r,transform=ccrs.PlateCarree())
    for c in sc_std.collections:
        c.set_edgecolor('face')
    sc_dkl = ax_dkl.contourf(lons,lats,dkl,levels=dkl_levels,cmap=plt.cm.cubehelix,transform=ccrs.PlateCarree(),extend='max')
    for c in sc_dkl.collections:
        c.set_edgecolor('face')

    cb_mean = plt.colorbar(sc_mean,cax=ax_mean_cb,ticks=[-max_mean,0,max_mean])
    cb_std = plt.colorbar(sc_std,cax=ax_std_cb,ticks=[0,max_std/2,max_std])
    cb_dkl = plt.colorbar(sc_dkl,cax=ax_dkl_cb,extend='max',ticks=[0,max_dkl/2,max_dkl])
    ax_mean_cb.yaxis.set_label_position('left')
    ax_std_cb.yaxis.set_label_position('left')
    ax_dkl_cb.yaxis.set_label_position('left')
    cb_mean.set_label("Inferred residual topo., km")
    cb_std.set_label("Standard deviation, km")
    cb_dkl.set_label(r"$D_{KL}$, nats")
    #ax_mean.text(0,0,'%.1f/+%.1fkm'%(mean.min(),mean.max()),transform=ax_mean.transAxes)
    ax_mean.text(0,0.95,'(a)',transform=ax_mean.transAxes)
    ax_std.text(0,0.95,'(b)',transform=ax_std.transAxes)
    ax_dkl.text(0,0.95,'(c)',transform=ax_dkl.transAxes)
    ax_mean.text(0.85,0,'%.2f/%.2f'%(mean.min(),mean.max()),transform=ax_mean.transAxes)
    ax_std.text(0.85,0,'%.2f'%(variance.max()**0.5),transform=ax_std.transAxes)
    ax_dkl.text(0.85,0,'%.2f'%(dkl.max()),transform=ax_dkl.transAxes)
    plt.savefig(outfile,dpi=300)
    if SHOWFIGS: plt.show()


def plotCovariance(paramfile_ha,paramfile_all,paramfile_ship,outfile):
    plt.rcParams['font.size']=12
    kmMax = 10000
    radMax = kmMax/6371.
    nRad = 5001
    dd = np.linspace(-radMax,radMax,nRad)
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    if SHOWTABLES:
        print("  Table 1:")
        print("          mu   delta   sig1   sig2     v")
    try:
        params_spot_ha = loadOptimalParams(paramfile_ha)
        if SHOWTABLES: print("    HA: %5.2f          %4.2f   %4.2f   %4.2f"%(params_spot_ha[3],*params_spot_ha[0:3]))
        ax.plot(6371*dd,matern_full(abs(dd),params_spot_ha[0:3]),color=color_spot_1,label='High accuracy spot only',zorder=3)
    except FileNotFoundError:
        print ("  Unable to load optimal parameters for high-accuracy spot data; continuing...")
    try:
        params_spot_all = loadOptimalParams(paramfile_all)
        if SHOWTABLES: print("   All: %5.2f   %4.2f   %4.2f   %4.2f   %4.2f"%(params_spot_all[3],params_spot_all[4],*params_spot_all[0:3]))
        ax.plot(6371*dd,matern_full(abs(dd),params_spot_all[0:3]),color=color_all_1,linestyle='--',label="All spot",zorder=2)
    except FileNotFoundError:
        print ("  Unable to load optimal parameters for all spot data; continuing...")
    try:
        params_spot_ship = loadOptimalParams(paramfile_ship)
        if SHOWTABLES: print("  Ship: %5.2f   %4.2f   %4.2f   %4.2f   %4.2f"%(params_spot_ship[3],params_spot_ship[4],*params_spot_ship[0:3]))
        ax.plot(6371*dd,matern_full(abs(dd),params_spot_ship[0:3]),color=color_spot_ship_1,linestyle='--',label="All spot and shiptrack",zorder=1)
    except FileNotFoundError:
        print ("  Unable to load optimal parameters for shiptrack data; continuing...")
    ax.set_xlim(-kmMax,kmMax)
    ax.set_xticks([-10000,-5000,0,5000,10000])
    ax.set_xlabel(r"Distance, $d(\mathbf{x},\mathbf{x^\prime})$, km")
    ax.set_ylim(0,0.425)
    ax.set_yticks([0,0.2,0.4])
    ax.set_ylabel(r"Covariance, $k(\mathbf{x},\mathbf{x^\prime})$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    if SHOWFIGS: plt.show()
def plotSpectra(sphfile_ha,sphfile_all,sphfile_ship,outfile_spec,outfile_hist,n_sample=100000):
    plt.rcParams['font.size']=13
    np.random.seed(42) # Seed random number generator for repeatability
    fig = plt.figure(figsize=(12,4))
    # Grid for finding the maximum absolute topographic height
    nlats = 45
    nlons = 90
    if SHOWTABLES: ll,smv = sph_map_vec(nlats,nlons)
    # Load ARD results from Davies et al. 2019
    try:
        davies_ARD = np.loadtxt('Davies_ARD.dat')
    except FileNotFoundError:
        print("  Unable to load ARD results; omitting...")
    try:
        ax = fig.add_subplot(131)
        y_spot_ha,Sigma_spot_ha = loadSpectrum(sphfile_ha)
        pow_spot_ha = modelToPower(y_spot_ha)
        posterior_samples_ha = np.random.multivariate_normal(y_spot_ha,Sigma_spot_ha,size=n_sample)
        random_spot_ha = modelToPower(posterior_samples_ha)
        b = bands(random_spot_ha)
        if SHOWTABLES:
            print("  High accuracy spot data: power of %i random samples from posterior (Table 2)"%n_sample)
            for l in range(1,pow_spot_ha.shape[0]+1):
                print("    l=%2i:  MP = %.2f  Mean = %.2f  Std = %.2f Median = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(l,pow_spot_ha[l-1],random_spot_ha[:,l-1].mean(),random_spot_ha[:,l-1].std(),
                                                                                                    b[2,l-1],b[0,l-1],b[4,l-1],b[1,l-1],b[3,l-1]))
            print("  High accuracy spot data: maximum absolute topographic heights across random samples (Table 3)")
            d = abs(posterior_samples_ha[:,0:15].dot(smv[0:15,:])).max(1)
            srt = np.argsort(d)
            print("    Degrees 1--3   - MP = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(abs(smv[0:15,:].T.dot(y_spot_ha[0:15])).max(),d[srt[int(n_sample/200)]],d[srt[int(199*n_sample/200)]],d[srt[int(25*n_sample/100)]],d[srt[int(75*n_sample/100)]]))
            d = abs(posterior_samples_ha[:,15:120].dot(smv[15:120,:])).max(1)
            srt = np.argsort(d)
            print("    Degrees 3--10  - MP = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(abs(smv[15:120,:].T.dot(y_spot_ha[15:120])).max(),d[srt[int(n_sample/200)]],d[srt[int(199*n_sample/200)]],d[srt[int(25*n_sample/100)]],d[srt[int(75*n_sample/100)]]))
            d = abs(posterior_samples_ha[:,120:].dot(smv[120:,:])).max(1)
            srt = np.argsort(d)
            print("    Degrees 11--30 - MP = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(abs(smv[120:,:].T.dot(y_spot_ha[120:])).max(),d[srt[int(n_sample/200)]],d[srt[int(199*n_sample/200)]],d[srt[int(25*n_sample/100)]],d[srt[int(75*n_sample/100)]]))
        ax.plot(np.arange(1,pow_spot_ha.shape[0]+1),pow_spot_ha,color=color_spot_1)
        try:
            ax.fill_between(np.arange(1,pow_spot_ha.shape[0]+1),davies_ARD[:pow_spot_ha.shape[0],2],davies_ARD[:pow_spot_ha.shape[0],3],color='darkgrey',alpha=0.5)
            ax.fill_between(np.arange(1,pow_spot_ha.shape[0]+1),davies_ARD[:pow_spot_ha.shape[0],4],davies_ARD[:pow_spot_ha.shape[0],5],color='grey',alpha=0.5,label="Davies et al. (2019), ARD")
        except NameError:
            pass
        ax.fill_between(np.arange(1,pow_spot_ha.shape[0]+1),b[0,:],b[4,:],color=color_spot_2,alpha=0.6)
        ax.fill_between(np.arange(1,pow_spot_ha.shape[0]+1),b[1,:],b[3,:],color=color_spot_1,alpha=0.6,label="High accuracy spot only")
        ax.text(0.025,0.925,'(a)',transform=ax.transAxes)
        ax.legend(loc="lower left")
        ax.set_yscale('log')
        ax.set_ylabel(r"Power, km${}^2$")
        ax.set_ylim(1e-3,2.)
        ax.set_xlim(1,30)
        ax.set_xticks([1,10,20,30])
        ax.set_xlabel("Spherical harmonic degree")
        ax.grid()
    except FileNotFoundError:
        print("  Unable to load spherical harmonic coefficients for high-accuracy spot data; continuing...")
    try:
        ax = fig.add_subplot(132)
        y_spot_all,Sigma_spot_all = loadSpectrum(sphfile_all)
        pow_spot_all = modelToPower(y_spot_all)
        posterior_samples_all = np.random.multivariate_normal(y_spot_all,Sigma_spot_all,size=n_sample)
        random_spot_all = modelToPower(posterior_samples_all)
        b = bands(random_spot_all)
        if SHOWTABLES:
            print("  All spot data: power of %i random samples from posterior (Table 2)"%n_sample)
            for l in range(1,pow_spot_all.shape[0]+1):
                print("    l=%2i: MP = %.2f  Mean = %.2f  Std = %.2f Median = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(l,pow_spot_all[l-1],random_spot_all[:,l-1].mean(),random_spot_all[:,l-1].std(),
                                                                                                            b[2,l-1],b[0,l-1],b[4,l-1],b[1,l-1],b[3,l-1]))
            print("  All spot data: maximum absolute topographic heights across random samples (Table 3)")
            d = abs(posterior_samples_all[:,0:15].dot(smv[0:15,:])).max(1)
            srt = np.argsort(d)
            print("    Degrees 1--3   - MP = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(abs(smv[0:15,:].T.dot(y_spot_all[0:15])).max(),d[srt[int(n_sample/200)]],d[srt[int(199*n_sample/200)]],d[srt[int(25*n_sample/100)]],d[srt[int(75*n_sample/100)]]))
            d = abs(posterior_samples_all[:,15:120].dot(smv[15:120,:])).max(1)
            srt = np.argsort(d)
            print("    Degrees 3--10  - MP = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(abs(smv[15:120,:].T.dot(y_spot_all[15:120])).max(),d[srt[int(n_sample/200)]],d[srt[int(199*n_sample/200)]],d[srt[int(25*n_sample/100)]],d[srt[int(75*n_sample/100)]]))
            d = abs(posterior_samples_all[:,120:].dot(smv[120:,:])).max(1)
            srt = np.argsort(d)
            print("    Degrees 11--30 - MP = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(abs(smv[120:,:].T.dot(y_spot_all[120:])).max(),d[srt[int(n_sample/200)]],d[srt[int(199*n_sample/200)]],d[srt[int(25*n_sample/100)]],d[srt[int(75*n_sample/100)]]))
        ax.plot(np.arange(1,pow_spot_all.shape[0]+1),pow_spot_all,color=color_all_1)
        try:
            ax.fill_between(np.arange(1,pow_spot_all.shape[0]+1),davies_ARD[:pow_spot_all.shape[0],2],davies_ARD[:pow_spot_all.shape[0],3],color='darkgrey',alpha=0.5)
            ax.fill_between(np.arange(1,pow_spot_all.shape[0]+1),davies_ARD[:pow_spot_all.shape[0],4],davies_ARD[:pow_spot_all.shape[0],5],color='grey',alpha=0.5,label="Davies et al. (2019), ARD")
        except NameError:
            pass
        ax.fill_between(np.arange(1,pow_spot_all.shape[0]+1),b[0,:],b[4,:],color=color_all_2,alpha=0.6)
        ax.fill_between(np.arange(1,pow_spot_all.shape[0]+1),b[1,:],b[3,:],color=color_all_1,alpha=0.6,label="All spot")
        ax.text(0.025,0.925,'(b)',transform=ax.transAxes)
        ax.legend(loc="lower left")
        ax.set_yscale('log')
        ax.set_ylim(1e-3,2.)
        ax.set_yticklabels([])
        ax.set_xlim(1,30)
        ax.set_xticks([1,10,20,30])
        ax.set_xlabel("Spherical harmonic degree")
        ax.grid()
    except FileNotFoundError:
        print("  Unable to load spherical harmonic coefficients for all spot data; continuing...")
    try:
        ax = fig.add_subplot(133)
        y_spot_ship,Sigma_spot_ship = loadSpectrum(sphfile_ship)
        pow_spot_ship = modelToPower(y_spot_ship)
        posterior_samples_ship = np.random.multivariate_normal(y_spot_ship,Sigma_spot_ship,size=n_sample)
        random_spot_ship = modelToPower(posterior_samples_ship)
        b = bands(random_spot_ship)
        if SHOWTABLES:
            print("  All spot and shiptrack data: power of %i random samples from posterior (Table 2)"%n_sample)
            for l in range(1,pow_spot_ship.shape[0]+1):
                print("    l=%2i:  MP = %.2f  Mean = %.2f  Std = %.2f Median = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(l,pow_spot_ship[l-1],random_spot_ship[:,l-1].mean(),random_spot_ship[:,l-1].std(),
                                                                                                        b[2,l-1],b[0,l-1],b[4,l-1],b[1,l-1],b[3,l-1]))
            print("  All spot and shiptrack data: maximum absolute topographic heights across random samples (Table 3)")
            d = abs(posterior_samples_ship[:,0:15].dot(smv[0:15,:])).max(1)
            srt = np.argsort(d)
            print("    Degrees 1--3   - MP = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(abs(smv[0:15,:].T.dot(y_spot_ship[0:15])).max(),d[srt[int(n_sample/200)]],d[srt[int(199*n_sample/200)]],d[srt[int(25*n_sample/100)]],d[srt[int(75*n_sample/100)]]))
            d = abs(posterior_samples_ship[:,15:120].dot(smv[15:120,:])).max(1)
            srt = np.argsort(d)
            print("    Degrees 3--10  - MP = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(abs(smv[15:120,:].T.dot(y_spot_ship[15:120])).max(),d[srt[int(n_sample/200)]],d[srt[int(199*n_sample/200)]],d[srt[int(25*n_sample/100)]],d[srt[int(75*n_sample/100)]]))
            d = abs(posterior_samples_ship[:,120:].dot(smv[120:,:])).max(1)
            srt = np.argsort(d)
            print("    Degrees 11--30 - MP = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(abs(smv[120:,:].T.dot(y_spot_ship[120:])).max(),d[srt[int(n_sample/200)]],d[srt[int(199*n_sample/200)]],d[srt[int(25*n_sample/100)]],d[srt[int(75*n_sample/100)]]))
        ax.plot(np.arange(1,pow_spot_ship.shape[0]+1),pow_spot_ship,color=color_spot_ship_1)
        try:
            ax.fill_between(np.arange(1,pow_spot_ship.shape[0]+1),davies_ARD[:pow_spot_ship.shape[0],2],davies_ARD[:pow_spot_ship.shape[0],3],color='darkgrey',alpha=0.5)
            ax.fill_between(np.arange(1,pow_spot_ship.shape[0]+1),davies_ARD[:pow_spot_ship.shape[0],4],davies_ARD[:pow_spot_ship.shape[0],5],color='grey',alpha=0.5,label="Davies et al. (2019), ARD")
        except NameError:
            pass
        ax.fill_between(np.arange(1,pow_spot_ship.shape[0]+1),b[0,:],b[4,:],color=color_spot_ship_2,alpha=0.6)
        ax.fill_between(np.arange(1,pow_spot_ship.shape[0]+1),b[1,:],b[3,:],color=color_spot_ship_1,alpha=0.6,label="All spot and shiptrack")
        ax.text(0.025,0.925,'(c)',transform=ax.transAxes)
        ax.legend(loc="lower left")
        ax.set_yscale('log')
        ax.set_ylim(1e-3,2.)
        ax.set_yticklabels([])
        ax.set_xlim(1,30)
        ax.set_xticks([1,10,20,30])
        ax.set_xlabel("Spherical harmonic degree")
        ax.grid()
    except FileNotFoundError:
        print("  Unable to load spherical harmonic coefficients for shiptrack data; continuing...")
    plt.tight_layout()
    plt.savefig(outfile_spec)
    if SHOWFIGS: plt.show()

    nrows = 3
    ncols = 8
    plt.rcParams['font.size']=16
    fig = plt.figure(figsize=(15,8))
    bins = np.linspace(0,1,100)
    for l in range(1,23):
        ax = fig.add_subplot(nrows,ncols,l)
        if l==22:
            label_spot_ha = "High accuracy spot only"
            label_spot_all = "All spot"
            label_spot_ship = "All spot and shiptrack"
        else:
            label_spot_ha = None
            label_spot_all = None
            label_spot_ship = None
        try:
            ax.hist(random_spot_ha[:,l-1],bins,orientation='horizontal',color=color_spot_1,label=label_spot_ha,alpha=0.6,zorder=3)
        except NameError:
            pass
        try:
            ax.hist(random_spot_all[:,l-1],bins,orientation='horizontal',color=color_all_1,label=label_spot_all,alpha=0.6,zorder=2)
        except NameError:
            pass
        try:
            ax.hist(random_spot_ship[:,l-1],bins,orientation='horizontal',color=color_spot_ship_1,alpha=1,label=label_spot_ship,zorder=1)
        except NameError:
            pass
        ax.set_xticks([])
        ax.set_yticks([0,.25,0.5,.75,1])
        ax.set_ylim(0,1)
        ax.text(0.5,0.87,'l = %i'%l,transform=ax.transAxes)
        if l == 1 or l==9 or l==17:
            ax.set_yticklabels(['0','','0.5','','1'])
            ax.set_ylabel(r"Power, km${}^2$")
        # elif l == 17:
        #     ax.set_yticklabels(['0','','0.5','','1'])
        #     ax.set_ylabel(r"Power, km${}^2$")
        else:
            ax.set_yticklabels([])
        if l==22: fig.legend(loc=(.75,0.15))
    plt.tight_layout()
    plt.savefig(outfile_hist)
    if SHOWFIGS: plt.show()
def plotWhereToSample(mapdatafile,samplemask,outfile,llist=[2,5,10,15,20,30]):
    plt.rcParams['font.size']=12
    with open(mapdatafile,'rb') as fp:
        mean = pickle.load(fp)
        variance = pickle.load(fp)
        dkl = pickle.load(fp)
        lats = pickle.load(fp)
        lons = pickle.load(fp)
    nlats = lats.shape[0]
    nlons = lons.shape[0]
    fig_dkl = plt.figure(figsize=(8,7))

    c1 = 0.05
    cshift = 0.475
    width=0.425
    r1 = 0.16
    rowshift = 0.28
    height=0.25
    for i,l in enumerate(llist):
        #ax = fig_dkl.add_subplot(3,2,i+1,projection=ccrs.Robinson())
        ax = fig_dkl.add_axes((c1 +(i%2)*cshift,r1+(2-int(i/2))*rowshift,width,height),projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(color='white')
        with open(samplemask%l,'rb') as fp:
            dkl_min = pickle.load(fp)
            dkl = pickle.load(fp)
        sc = ax.contourf(lons,lats,dkl/dkl.max(),levels=np.linspace(0,1,100),cmap=plt.cm.cubehelix,transform=ccrs.PlateCarree())
        for c in sc.collections:
            c.set_edgecolor('face')
        ax.text(0,0.95,"l = %i"%l,transform=ax.transAxes)
        ax.text(0.87,0,"%.2f"%dkl.max(),transform=ax.transAxes)
    ax = fig_dkl.add_axes((0.4,0.05,0.2,0.05))
    plt.colorbar(sc,cax=ax,orientation='horizontal',ticks=[0,0.5,1])
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels(['0','Max/2','Max'])
    ax.set_xlabel("Value of one additional sample")
    plt.savefig(outfile)
    if SHOWFIGS:plt.show()
def plotLowDegrees(sph_ha,sph_all,sph_ship,outfile):
    nlats = 90
    nlons = 180
    ll,smv = sph_map_vec(nlats,nlons)
    plt.rcParams['font.size']=12
    fig = plt.figure(figsize=(8,10))
    x0=0.05
    y0=0.05
    cy=0.03
    ysp=0.04
    my=(1-(2*y0)-cy)/3 - ysp
    for i,file in enumerate([sph_ha,sph_all,sph_ship]):
        ax = fig.add_axes((x0,y0+cy+(2-i)*(ysp+my)+ysp,1-2*x0,my),projection=ccrs.Robinson())
        ax.text(0.0,0.95,'(%s)'%(['a','b','c'][i]),transform=ax.transAxes)
        ax.coastlines()
        try:
            y,Sigma = loadSpectrum(file)
        except FileNotFoundError:
            print("  Unable to load %s; continuing..."%file)
            continue
        mapdata = y[0:15].dot(smv[0:15,:]).reshape(nlats,nlons)
        pc = ax.pcolor(ll[:,1].reshape(nlats,nlons),ll[:,0].reshape(nlats,nlons),mapdata,cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree())
        pc.set_edgecolor('face')
        co = ax.contour(ll[:,1].reshape(nlats,nlons),ll[:,0].reshape(nlats,nlons),y[0:15].dot(smv[0:15,:]).reshape(nlats,nlons),levels=[-1.4,-1.2,-1,-.8,-.6,-.4,-.2,.2,.4,.6,.8,1,1.2,1.4],transform=ccrs.PlateCarree(),colors='k')
        co = ax.contour(ll[:,1].reshape(nlats,nlons),ll[:,0].reshape(nlats,nlons),y[0:15].dot(smv[0:15,:]).reshape(nlats,nlons),levels=[0],transform=ccrs.PlateCarree(),colors='grey')
        pc.set_clim(-.8,.8)
        ax.text(0.9,0,'%.2f/%.2f'%(mapdata.min(),mapdata.max()),transform=ax.transAxes)
    ax = fig.add_axes((x0+0.25,y0,1-2*(x0+0.25),cy))
    plt.colorbar(pc,cax=ax,orientation='horizontal',ticks=[-.8,-.4,0,.4,.8],label="Residual topography at degrees 1-3, km")
    ax.xaxis.set_label_position('top')
    plt.savefig(outfile)
    if SHOWFIGS:plt.show()
def calculateModelRange(datafile,dataset_type,paramfile,inversefile,nlats = 45,nlons = 90,n_sample=100000):
    # Seed RNG for repeatability
    np.random.seed(42)
    try:
        data,ndata = loadData(datafile,dataset_type)
        optimal_params = loadOptimalParams(paramfile)
        ngrid = nlats*nlons
        data[:,2] -= optimal_params[3]
        with open(inversefile,'rb') as fp:
            iK = pickle.load(fp)
    except FileNotFoundError:
        print("    Data files not found; skipping...")
        return
    k1 = np.zeros([ngrid,ngrid])
    k2 = np.zeros([ngrid,data.shape[0]])
    ll = np.zeros([ngrid,2])
    i = 0
    for ilat,lat in enumerate(np.linspace(-90,90,nlats)):
        for ilon,lon in enumerate(np.linspace(-180,180,nlons)):
            ll[i,:] = lat,lon
            i+=1
    for i in range(ngrid):
        k1[:,i] = matern_full(dist(ll[i,0],ll[i,1],ll[:,0],ll[:,1]),optimal_params[0:3])
        k2[i,:] = matern_full(dist(ll[i,0],ll[i,1],data[:,0],data[:,1]),optimal_params[0:3])
    rmaps = np.random.multivariate_normal(k2.dot(iK).dot(data[:,2]),k1 - k2.dot(iK).dot(k2.T),size=n_sample)
    maxs = abs(rmaps).max(1)
    srt = np.argsort(maxs)
    print('    Median: %.2f  99: %.2f--%.2f  50: %.2f--%.2f'%(maxs[srt[int(n_sample/2)]],maxs[srt[int(n_sample/200)]],maxs[srt[int(199*n_sample/200)]],maxs[srt[int(n_sample/4)]],maxs[srt[int(3*n_sample/4)]]))

if __name__ == '__main__':
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
    SHOWFIGS = userSettings.PLT_SHOW
    SHOWTABLES = userSettings.TABLE_DATA
    figdir=os.path.join(outputdir,'figures')
    if not os.path.exists(figdir): os.mkdir(figdir)
    # Create some functions to handle paths neatly
    figpath = lambda f: os.path.join(figdir,f)
    ha_spot = lambda f: os.path.join(outputdir,'high_accuracy_spot',f)
    all_spot = lambda f: os.path.join(outputdir,'all_spot',f)
    spot_ship = lambda f: os.path.join(outputdir,'spot_shiptrack',f)


    # Map of raw data -- Fig. 1
    if toggleFigure(1):
        print("Making figure 1: Map of raw data")
        plotDatasets(datafile,figpath('data.pdf'))
        print("Figure 1 complete; file saved at %s\n"%figpath('data.pdf'))
    # Plot of covariance functions -- Fig. 2; Part of Table 1
    if toggleFigure(2) or SHOWTABLES:
    print("Making figure 2: plot of covariance functions")
        plotCovariance(ha_spot('optimal_params.pickle'),
                       all_spot('optimal_params.pickle'),
                       spot_ship('optimal_params.pickle'),
                       figpath('k_comparison.pdf'))
        print("Figure 2 complete; file saved at %s\n"%figpath('k_comparison.pdf'))
    # Maps for all three models -- Figs. 3--5
    if toggleFigure(3):
        try:
            print("Making figure 3: Map of GP model from high-accuracy spot data")
            plotMap(ha_spot('mapdata.pickle'),figpath('map_high_accuracy_spot.pdf'))
            print("Figure 3 complete; file saved at %s\n"%figpath('map_high_accuracy_spot.pdf'))
        except FileNotFoundError:
            print("Unable to find necessary data files. Skipping...\n")
    if toggleFigure(4):
        try:
            print("Making figure 4: Map of GP model from all spot data")
            plotMap(all_spot('mapdata.pickle'),figpath('map_all_spot.pdf'))
            print("Figure 4 complete; file saved at %s\n"%figpath('map_all_spot.pdf'))
        except FileNotFoundError:
            print("Unable to find necessary data files. Skipping...\n")
    if toggleFigure(5):
        try:
            print("Making figure 5: Map of GP model from spot and shiptrack data")
            plotMap(spot_ship('mapdata.pickle'),figpath('map_spot_shiptrack.pdf'))
            print("Figure 5 complete; file saved at %s\n"%figpath('map_spot_shiptrack.pdf'))
        except FileNotFoundError:
            print("Unable to find necessary data files. Skipping...\n")
    if SHOWTABLES: # Part of Table 3
        print("Computing maximum absolute amplitudes for full models (Table 3)")
        print("  (For values associated with most-probable model refer to Figs. 3--5)")
        print("  High accuracy spot data (%i random samples)"%userSettings.N_RANDOM_SAMPLES)
        calculateModelRange(datafile,'high_accuracy_spot',ha_spot('optimal_params.pickle'),ha_spot('inverseCov.pickle'),n_sample = userSettings.N_RANDOM_SAMPLES)
        print("  All spot data (%i random samples)"%userSettings.N_RANDOM_SAMPLES)
        calculateModelRange(datafile,'all_spot',all_spot('optimal_params.pickle'),all_spot('inverseCov.pickle'),n_sample = userSettings.N_RANDOM_SAMPLES)
        print("  Spot and shiptrack data (%i random samples)"%userSettings.N_RANDOM_SAMPLES)
        calculateModelRange(datafile,'spot_shiptrack',spot_ship('optimal_params.pickle'),spot_ship('inverseCov.pickle'),n_sample = userSettings.N_RANDOM_SAMPLES)
        print("")
    # Plot of spectra -- Fig. 6-7; Table 2; Part of Table 3
    if toggleFigure(6) or toggleFigure(7) or SHOWTABLES:
        print("Making figures 6 & 7: Plots of spectra")
        plotSpectra(ha_spot('sphcoeff.pickle'),
                     all_spot('sphcoeff.pickle'),
                     spot_ship('sphcoeff.pickle'),
                     figpath('spectra.pdf'),
                     figpath('histograms.pdf'),
                     n_sample = userSettings.N_RANDOM_SAMPLES)
        print("Figure 6 complete; file saved at %s"%figpath('spectra.pdf'))
        print("Figure 7 complete; file saved at %s\n"%figpath('histograms.pdf'))
    if toggleFigure(8):
        try:
            # Map of where to sample -- Fig. 8
            print("Making figure 8: Map of value of one additional sample")
            plotWhereToSample(ha_spot('mapdata.pickle'),ha_spot('sampling_%i.pickle'),figpath('wheretosample.pdf'))
            print("Figure 8 complete; file saved at %s\n"%figpath('wheretosample.pdf'))
        except FileNotFoundError:
            print("Unable to find necessary data files. Skipping...")
    if toggleFigure(9):
        print("Making figure 9: Map of low-degree residual topography")
        plotLowDegrees(ha_spot('sphcoeff.pickle'),all_spot('sphcoeff.pickle'),spot_ship('sphcoeff.pickle'),figpath('1to3.pdf'))
        print("Figure 9 complete; file saved at %s\n"%figpath('1to3.pdf'))
    print("All figures generated.")
