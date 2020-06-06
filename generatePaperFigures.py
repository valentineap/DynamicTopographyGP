import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pickle
from dynamicTopoGP import N_SPOT,N_HIGH_ACCURACY,loadOptimalParams,matern_full,loadSpectrum,modelToPower
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


def bands(powers):
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
def plotDatasets(datafile,outfile):
    plt.rcParams['font.size']=12
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
    print("Table 1:")
    print("        mu   delta   sig1   sig2     v")
    try:
        params_spot_ha = loadOptimalParams(paramfile_ha)
        print("  HA: %5.2f          %4.2f   %4.2f   %4.2f"%(params_spot_ha[3],*params_spot_ha[0:3]))
        ax.plot(6371*dd,matern_full(abs(dd),params_spot_ha[0:3]),color=color_spot_1,label='High accuracy spot only',zorder=3)
    except FileNotFoundError:
        print ("Unable to load optimal parameters for high-accuracy spot data; continuing...")
    try:
        params_spot_all = loadOptimalParams(paramfile_all)
        print(" All: %5.2f   %4.2f   %4.2f   %4.2f   %4.2f"%(params_spot_all[3],params_spot_all[4],*params_spot_all[0:3]))
        ax.plot(6371*dd,matern_full(abs(dd),params_spot_all[0:3]),color=color_all_1,linestyle='--',label="All spot",zorder=2)
    except FileNotFoundError:
        print ("Unable to load optimal parameters for all spot data; continuing...")
    try:
        params_spot_ship = loadOptimalParams(paramfile_ship)
        print("Ship: %5.2f   %4.2f   %4.2f   %4.2f   %4.2f"%(params_spot_ship[3],params_spot_ship[4],*params_spot_ship[0:3]))
        ax.plot(6371*dd,matern_full(abs(dd),params_spot_ship[0:3]),color=color_spot_ship_1,linestyle='--',label="All spot and shiptrack",zorder=1)
    except FileNotFoundError:
        print ("Unable to load optimal parameters for shiptrack data; continuing...")
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
    plt.rcParams['font.size']=12
    np.random.seed(42) # Seed random number generator for repeatability
    try:
        y_spot_ha,Sigma_spot_ha = loadSpectrum(sphfile_ha)
        pow_spot_ha = modelToPower(y_spot_ha)
        random_spot_ha = modelToPower(np.random.multivariate_normal(y_spot_ha,Sigma_spot_ha,size=n_sample))
    except FileNotFoundError:
        print("Unable to load spherical harmonic coefficients for high-accuracy spot data; continuing...")
    try:
        y_spot_all,Sigma_spot_all = loadSpectrum(sphfile_all)
        pow_spot_all = modelToPower(y_spot_all)
        random_spot_all = modelToPower(np.random.multivariate_normal(y_spot_all,Sigma_spot_all,size=n_sample))
    except FileNotFoundError:
        print("Unable to load spherical harmonic coefficients for all spot data; continuing...")
    try:
        y_spot_ship,Sigma_spot_ship = loadSpectrum(sphfile_ship)
        pow_spot_ship = modelToPower(y_spot_ship)
        random_spot_ship = modelToPower(np.random.multivariate_normal(y_spot_ship,Sigma_spot_ship,size=n_sample))
    except FileNotFoundError:
        print("Unable to load spherical harmonic coefficients for shiptrack data; continuing...")
    try:
        davies_ARD = np.loadtxt('Davies_ARD.dat')
    except FileNotFoundError:
        print("Unable to load ARD results; continuing...")

    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(131)
    # High-accuracy spot
    try:
        b = bands(random_spot_ha)
        print("High accuracy spot data: %i random samples from posterior (Table 2)"%n_sample)
        for l in range(1,pow_spot_ha.shape[0]+1):
            print("l=%2i:  MP = %.2f  Mean = %.2f  Std = %.2f Median = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(l,pow_spot_ha[l-1],random_spot_ha[:,l-1].mean(),random_spot_ha[:,l-1].std(),
                                                                                                b[2,l-1],b[0,l-1],b[4,l-1],b[1,l-1],b[3,l-1]))
        ax.plot(np.arange(1,pow_spot_ha.shape[0]+1),pow_spot_ha,color=color_spot_1)
    except NameError:
        pass
    try:
        ax.fill_between(np.arange(1,pow_spot_ha.shape[0]+1),davies_ARD[:pow_spot_ha.shape[0],2],davies_ARD[:pow_spot_ha.shape[0],3],color='darkgrey',alpha=0.5)
        ax.fill_between(np.arange(1,pow_spot_ha.shape[0]+1),davies_ARD[:pow_spot_ha.shape[0],4],davies_ARD[:pow_spot_ha.shape[0],5],color='grey',alpha=0.5,label="Davies et al. (2019), ARD")
    except NameError:
        pass
    try:
        ax.fill_between(np.arange(1,pow_spot_ha.shape[0]+1),b[0,:],b[4,:],color=color_spot_2,alpha=0.6)
        ax.fill_between(np.arange(1,pow_spot_ha.shape[0]+1),b[1,:],b[3,:],color=color_spot_1,alpha=0.6,label="High accuracy spot only")
    except NameError:
        pass
    ax.text(0.025,0.925,'(a)',transform=ax.transAxes)
    ax.legend(loc="lower left")
    ax.set_yscale('log')
    ax.set_ylabel(r"Power, km${}^2$")
    ax.set_ylim(1e-3,2.)
    ax.set_xlim(1,30)
    ax.set_xticks([1,10,20,30])
    ax.set_xlabel("Spherical harmonic degree")
    ax.grid()
    # All spot data
    ax = fig.add_subplot(132)
    try:
        b = bands(random_spot_all)
        print("All spot data: %i random samples from posterior (Table 2)"%n_sample)
        for l in range(1,pow_spot_all.shape[0]+1):
            print("l=%2i: MP = %.2f  Mean = %.2f  Std = %.2f Median = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(l,pow_spot_all[l-1],random_spot_all[:,l-1].mean(),random_spot_all[:,l-1].std(),
                                                                                                        b[2,l-1],b[0,l-1],b[4,l-1],b[1,l-1],b[3,l-1]))
        #ax.plot(np.arange(1,pow_spot_ship.shape[0]+1),davies_ARD[:pow_spot.shape[0],1],color='darkgrey')
        ax.plot(np.arange(1,pow_spot_all.shape[0]+1),pow_spot_all,color=color_all_1)
    except NameError:
        pass
    try:
        ax.fill_between(np.arange(1,pow_spot_all.shape[0]+1),davies_ARD[:pow_spot_all.shape[0],2],davies_ARD[:pow_spot_all.shape[0],3],color='darkgrey',alpha=0.5)
        ax.fill_between(np.arange(1,pow_spot_all.shape[0]+1),davies_ARD[:pow_spot_all.shape[0],4],davies_ARD[:pow_spot_all.shape[0],5],color='grey',alpha=0.5,label="Davies et al. (2019), ARD")
    except NameError:
        pass
    try:
        ax.fill_between(np.arange(1,pow_spot_all.shape[0]+1),b[0,:],b[4,:],color=color_all_2,alpha=0.6)
        ax.fill_between(np.arange(1,pow_spot_all.shape[0]+1),b[1,:],b[3,:],color=color_all_1,alpha=0.6,label="All spot")
    except NameError:
        pass
    ax.text(0.025,0.925,'(b)',transform=ax.transAxes)
    ax.legend(loc="lower left")
    ax.set_yscale('log')
    ax.set_ylim(1e-3,2.)
    ax.set_yticklabels([])
    ax.set_xlim(1,30)
    ax.set_xticks([1,10,20,30])
    ax.set_xlabel("Spherical harmonic degree")
    ax.grid()

    # All spot + shiptrack
    ax = fig.add_subplot(133)
    try:
        b = bands(random_spot_ship)
        print("All spot and shiptrack data: %i random samples from posterior (Table 2)"%n_sample)
        for l in range(1,pow_spot_ship.shape[0]+1):
            print("l=%2i:  MP = %.2f  Mean = %.2f  Std = %.2f Median = %.2f  99: %.2f--%.2f  50: %.2f--%.2f"%(l,pow_spot_ship[l-1],random_spot_ship[:,l-1].mean(),random_spot_ship[:,l-1].std(),
                                                                                                    b[2,l-1],b[0,l-1],b[4,l-1],b[1,l-1],b[3,l-1]))
        #ax.plot(np.arange(1,pow_spot_ship.shape[0]+1),davies_ARD[:pow_spot.shape[0],1],color='darkgrey')
        ax.plot(np.arange(1,pow_spot_ship.shape[0]+1),pow_spot_ship,color=color_spot_ship_1)
    except NameError:
        pass
    try:
        ax.fill_between(np.arange(1,pow_spot_ship.shape[0]+1),davies_ARD[:pow_spot_ship.shape[0],2],davies_ARD[:pow_spot_ship.shape[0],3],color='darkgrey',alpha=0.5)
        ax.fill_between(np.arange(1,pow_spot_ship.shape[0]+1),davies_ARD[:pow_spot_ship.shape[0],4],davies_ARD[:pow_spot_ship.shape[0],5],color='grey',alpha=0.5,label="Davies et al. (2019), ARD")
    except NameError:
        pass
    try:
        ax.fill_between(np.arange(1,pow_spot_ship.shape[0]+1),b[0,:],b[4,:],color=color_spot_ship_2,alpha=0.6)
        ax.fill_between(np.arange(1,pow_spot_ship.shape[0]+1),b[1,:],b[3,:],color=color_spot_ship_1,alpha=0.6,label="All spot and shiptrack")
    except NameError:
        pass
    ax.text(0.025,0.925,'(c)',transform=ax.transAxes)
    ax.legend(loc="lower left")
    ax.set_yscale('log')
    ax.set_ylim(1e-3,2.)
    ax.set_yticklabels([])
    ax.set_xlim(1,30)
    ax.set_xticks([1,10,20,30])
    ax.set_xlabel("Spherical harmonic degree")
    ax.grid()

    plt.tight_layout()
    plt.savefig(outfile_spec)
    if SHOWFIGS: plt.show()

    nrows = 3
    ncols = 8
    plt.rcParams['font.size']=14
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
        ax.text(0.55,0.9,'l = %i'%l,transform=ax.transAxes)
        if l == 1 or l==9 or l==17:
            ax.set_yticklabels(['0','','0.5','','1'])
            ax.set_ylabel(r"Power, km${}^2$")
        # elif l == 17:
        #     ax.set_yticklabels(['0','','0.5','','1'])
        #     ax.set_ylabel(r"Power, km${}^2$")
        else:
            ax.set_yticklabels([])
        if l==22: fig.legend(loc=(.76,0.15))
    plt.tight_layout()
    plt.savefig(outfile_hist)
    if SHOWFIGS: plt.show()
def plotWhereToSample(mapdatafile,samplemask,outfile,llist=[2,5,10,15,20,30]):
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
        print("Working on panel %i of %i..."%(i+1,len(llist)))
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
    figdir=os.path.join(outputdir,'figures')
    if not os.path.exists(figdir): os.mkdir(figdir)
    # Create some functions to handle paths neatly
    figpath = lambda f: os.path.join(figdir,f)
    ha_spot = lambda f: os.path.join(outputdir,'high_accuracy_spot',f)
    all_spot = lambda f: os.path.join(outputdir,'all_spot',f)
    spot_ship = lambda f: os.path.join(outputdir,'spot_shiptrack',f)


    # Map of raw data -- Fig. 1
    plotDatasets(datafile,figpath('data.pdf'))
    # Maps for all three models -- Figs. 2--4
    try:
        plotMap(ha_spot('mapdata.pickle'),figpath('map_high_accuracy_spot.pdf'))
    except FileNotFoundError:
        print("Unable to find data files for map of high-accuracy spot data. Continuing...")
    try:
        plotMap(all_spot('mapdata.pickle'),figpath('map_all_spot.pdf'))
    except FileNotFoundError:
        print("Unable to find data files for map of all spot data. Continuing...")
    try:
        plotMap(spot_ship('mapdata.pickle'),figpath('map_spot_shiptrack.pdf'))
    except FileNotFoundError:
        print("Unable to find data files for map of spot and shiptrack data. Continuing...")
    # Plot of covariance functions -- Fig. 5
    plotCovariance(ha_spot('optimal_params.pickle'),
                   all_spot('optimal_params.pickle'),
                   spot_ship('optimal_params.pickle'),
                   figpath('k_comparison.pdf'))
    # Plot of spectra -- Fig. 6-7
    plotSpectra(ha_spot('sphcoeff.pickle'),
                 all_spot('sphcoeff.pickle'),
                 spot_ship('sphcoeff.pickle'),
                 figpath('spectra.pdf'),
                 figpath('histograms.pdf'),
                 100000)
    try:
        # Map of where to sample -- Fig. 8
        plotWhereToSample(ha_spot('mapdata.pickle'),ha_spot('sampling_%i.pickle'),figpath('wheretosample.pdf'))
    except FileNotFoundError:
        print("Unable to find data files for map of where to sample.")
