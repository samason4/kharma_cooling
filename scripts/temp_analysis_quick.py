######################################################################
#
# Simple analysis and plotting script to process problem output
# Plots mhdmodes, bondi and torus problem (2D and 3D)
#
######################################################################

# python3 ./scripts/temp_analysis_quick.py

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os,psutil,sys
import pyharm.coordinates as coords
import pyharm.fluid_dump as fluid_dump
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp

# Parallelize analysis by spawning several processes using multiprocessing's Pool object
def run_parallel(function,dlist,nthreads):
    pool = mp.Pool(nthreads)
    pool.map_async(function,dlist).get(720000)
    pool.close()
    pool.join()

# Initialize global variables
globalvars_keys = ['PROB','NDIMS','DUMPSDIR_heat','DUMPSDIR_heat_and_cool','PLOTSDIR']
globalvars = {}
grid ={}


# Take poloidal slice of 2D array
def xz_slice(var, dump, patch_pole=False):
  xz_var = np.zeros((dump['n1'], dump['n2']))
  for i in range(dump['n1']):
    xz_var[i,:] = var[dump['n1']-1-i,:]
  if patch_pole:
    xz_var[:,0] = xz_var[:,-1] = 0
  return xz_var

def xz_slice_iharm(var, patch_pole=False, average=False):
	xz_var = np.zeros((2*grid['n1'],grid['n2']))
	if average:
		var = np.mean(var,axis=2)
		for i in range(grid['n1']):
			xz_var[i,:] = var[grid['n1']-1-i,:]
			xz_var[i+grid['n1'],:] = var[i,:]
	else:
		angle = 0.; ind = 0
		for i in range(grid['n1']):
			xz_var[i,:] = var[grid['n1']-1-i,:,ind+grid['n3']//2]
			xz_var[i+grid['n1'],:] = var[i,:,ind]
	if patch_pole:
		xz_var[:,0] = xz_var[:,-1] = 0
	return xz_var

def analysis_torus2d(dumpval, cmap='jet', vmin=5, vmax=12, domain = [-50,0,-50,50], bh=True, shading='gouraud'):
    plt.clf()
    print("Analyzing {0:04d} dump".format(dumpval))
    dump_iharm = h5py.File(os.path.join('./dumps_iharm_3d_without','dump_0000{0:04d}.h5'.format(dumpval)),'r')
    
    rho = dump_iharm['prims'][()][Ellipsis,0]
    uu = dump_iharm['prims'][()][Ellipsis,1]
    game = np.array(dump_iharm['header/gam_e'][()])
    KEL0 = np.array(dump_iharm['prims'][()][Ellipsis,9])
    KEL1 = np.array(dump_iharm['prims'][()][Ellipsis,10])
    KEL2 = np.array(dump_iharm['prims'][()][Ellipsis,11])
    KEL3 = np.array(dump_iharm['prims'][()][Ellipsis,12])
    uel0 = rho**game*KEL0/(game-1.)
    uel1 = rho**game*KEL1/(game-1.)
    uel2 = rho**game*KEL2/(game-1.)
    uel3 = rho**game*KEL3/(game-1.)
    #I would define these at Tel (temperature of the electrons), but at first I tried plotting the normalized temperature, theta,
    #which ended up being way too small, and I didn't want to go through and change theta to Tel, just for the sake of time
    ME = 9.1093826e-28 #Electron mass
    MP = 1.67262171e-24 #Proton mass
    CL = 2.99792458e10 # Speed of light
    GNEWT = 6.6742e-8 # Gravitational constant
    MSUN = 1.989e33 # grams per solar mass
    Kbol = 1.380649e-16 # boltzmann constant
    M_bh = 6.5e9 #mass of M87* in solar masses
    M_unit = 1.e28 #arbitrary
    M_bh_cgs = M_bh * MSUN
    L_unit = GNEWT*M_bh_cgs/pow(CL, 2.)
    RHO_unit = M_unit*pow(L_unit, -3.)
    Ne_unit = RHO_unit/(MP + ME)
    U_unit = RHO_unit*CL*CL
    theta0 = np.log10((game-1.)*uel0*U_unit/(rho*Ne_unit*Kbol))
    theta1 = np.log10((game-1.)*uel1*U_unit/(rho*Ne_unit*Kbol))
    theta2 = np.log10((game-1.)*uel2*U_unit/(rho*Ne_unit*Kbol))
    theta3 = np.log10((game-1.)*uel3*U_unit/(rho*Ne_unit*Kbol))
    t = dump_iharm['t'][()]
    dump_iharm.close()
    t = "{:.3f}".format(t)

    xp = xz_slice_iharm(grid['x'], patch_pole=True)
    zp = xz_slice_iharm(grid['z'])
    theta0p = xz_slice_iharm(theta0)
    theta1p = xz_slice_iharm(theta1)
    theta2p = xz_slice_iharm(theta2)
    theta3p = xz_slice_iharm(theta3)
    rhop = xz_slice_iharm(np.log10(rho))
    uup = xz_slice_iharm(np.log10(uu))

#**********************************************************************************************************************

    dump_with = fluid_dump.load_dump("./torus.out0.{0:05}.phdf".format(dumpval))
    rho2 = np.squeeze(dump_with['rho'])
    uu2 = np.squeeze(dump_with['UU'])
    game2 = 4.0/3.0
    KEL02 = np.squeeze(dump_with['Kel_Kawazura'])
    KEL12 = np.squeeze(dump_with['Kel_Werner'])
    KEL22 = np.squeeze(dump_with['Kel_Rowan'])
    KEL32 = np.squeeze(dump_with['Kel_Sharma'])
    uel02 = rho2**game2*KEL02/(game2-1.)
    uel12 = rho2**game2*KEL12/(game2-1.)
    uel22 = rho2**game2*KEL22/(game2-1.)
    uel32 = rho2**game2*KEL32/(game2-1.)
    theta02 = np.log10((game2-1.)*uel02*U_unit/(rho2*Ne_unit*Kbol))
    theta12 = np.log10((game2-1.)*uel12*U_unit/(rho2*Ne_unit*Kbol))
    theta22 = np.log10((game2-1.)*uel22*U_unit/(rho2*Ne_unit*Kbol))
    theta32 = np.log10((game2-1.)*uel32*U_unit/(rho2*Ne_unit*Kbol))
    t2 = dump_with['t']
    #t2 = "{:.3f}".format(t2)

    xp2k = xz_slice(grid['xk'], dump_with, patch_pole=True)
    zp2k = xz_slice(grid['zk'], dump_with)
    theta0p2 = xz_slice(theta02, dump_with)
    theta1p2 = xz_slice(theta12, dump_with)
    theta2p2 = xz_slice(theta22, dump_with)
    theta3p2 = xz_slice(theta32, dump_with)
    rhop2 = xz_slice(np.log10(rho2), dump_with)
    uup2 = xz_slice(np.log10(uu2), dump_with)

    fig = plt.figure(figsize=(16,9))
    heights = [1,6,6]
    gs = gridspec.GridSpec(nrows=3, ncols=6, height_ratios=heights, figure=fig)

    max_rho = np.amax(rho.reshape(16384) - rho2.reshape(16384))
    max_uu = np.amax(uu.reshape(16384) - uu2.reshape(16384))

    ax0 = fig.add_subplot(gs[0,:])
    ax0.annotate('Log(Temp in K) Iharm3d on top, Kharma on bottom, both with only heating at t ~ '+str(t)+' and max_rho='+str(max_rho)+' and max_uu='+str(max_uu),xy=(0.5,0.5),xycoords='axes fraction',va='center',ha='center',fontsize='x-large')
    ax0.axis("off")
#****************************************************************************************************************************************************************************************************************************************************************************************************************************
    #plotting rho:
    ax1 = fig.add_subplot(gs[1,0])
    rhopolplot = ax1.pcolormesh(xp, zp, rhop, cmap=cmap, vmin=-3, vmax=0, shading=shading)
    ax1.set_xlabel('$x (GM/c^2)$')
    ax1.set_ylabel('$z (GM/c^2)$')
    ax1.set_xlim(domain[:2])
    ax1.set_ylim(domain[2:])
    ax1.set_title('Log(density) (in code units)',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax1.add_artist(circle)
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(rhopolplot, cax=cax)

    #plotting rho2:
    ax2 = fig.add_subplot(gs[2,0])
    rho2polplot = ax2.pcolormesh(xp2k, zp2k, rhop2, cmap=cmap, vmin=-3, vmax=0, shading=shading)
    ax2.set_xlabel('$x (GM/c^2)$')
    ax2.set_ylabel('$z (GM/c^2)$')
    ax2.set_xlim(domain[:2])
    ax2.set_ylim(domain[2:])
    ax2.set_title('Log(density) (in code units)',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax2.add_artist(circle)
    ax2.set_aspect('equal')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(rho2polplot, cax=cax)
    
    #plotting uu:
    ax11 = fig.add_subplot(gs[1,1])
    uupolplot = ax11.pcolormesh(xp, zp, uup, cmap=cmap, vmin=-6, vmax=-2, shading=shading)
    ax11.set_xlabel('$x (GM/c^2)$')
    ax11.set_ylabel('$z (GM/c^2)$')
    ax11.set_xlim(domain[:2])
    ax11.set_ylim(domain[2:])
    ax11.set_title('Log(UU) (in code units)',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax11.add_artist(circle)
    ax11.set_aspect('equal')
    divider = make_axes_locatable(ax11)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(uupolplot, cax=cax)

    #plotting uu2:
    ax12 = fig.add_subplot(gs[2,1])
    uu2polplot = ax12.pcolormesh(xp2k, zp2k, uup2, cmap=cmap, vmin=-6, vmax=-2, shading=shading)
    ax12.set_xlabel('$x (GM/c^2)$')
    ax12.set_ylabel('$z (GM/c^2)$')
    ax12.set_xlim(domain[:2])
    ax12.set_ylim(domain[2:])
    ax12.set_title('Log(UU) (in code units)',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax12.add_artist(circle)
    ax12.set_aspect('equal')
    divider = make_axes_locatable(ax12)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(uu2polplot, cax=cax)
#****************************************************************************************************************************************************************************************************************************************************************************************************************************
    #plotting theta0p:
    ax3 = fig.add_subplot(gs[1,2])
    theta0polplot = ax3.pcolormesh(xp, zp, theta0p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax3.set_label('$x (GM/c^2)$')
    ax3.set_ylabel('$z (GM/c^2)$')
    ax3.set_xlim(domain[:2])
    ax3.set_ylim(domain[2:])
    ax3.set_title('Kawazura',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax3.add_artist(circle)
    ax3.set_aspect('equal')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta0polplot, cax=cax)

    #plotting theta1p:
    ax4 = fig.add_subplot(gs[1,3])
    theta1polplot = ax4.pcolormesh(xp, zp, theta1p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax4.set_xlabel('$x (GM/c^2)$')
    ax4.set_ylabel('$z (GM/c^2)$')
    ax4.set_xlim(domain[:2])
    ax4.set_ylim(domain[2:])
    ax4.set_title('Werner',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax4.add_artist(circle)
    ax4.set_aspect('equal')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta1polplot, cax=cax)

    #plotting theta2p:
    ax5 = fig.add_subplot(gs[1,4])
    theta2polplot = ax5.pcolormesh(xp, zp, theta2p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax5.set_xlabel('$x (GM/c^2)$')
    ax5.set_ylabel('$z (GM/c^2)$')
    ax5.set_xlim(domain[:2])
    ax5.set_ylim(domain[2:])
    ax5.set_title('Rowan',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax5.add_artist(circle)
    ax5.set_aspect('equal')
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta2polplot, cax=cax)
    
    #plotting theta3p:
    ax6 = fig.add_subplot(gs[1,5])
    theta3polplot = ax6.pcolormesh(xp, zp, theta3p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax6.set_xlabel('$x (GM/c^2)$')
    ax6.set_ylabel('$z (GM/c^2)$')
    ax6.set_xlim(domain[:2])
    ax6.set_ylim(domain[2:])
    ax6.set_title('Sharma',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax6.add_artist(circle)
    ax6.set_aspect('equal')
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta3polplot, cax=cax)

    #plotting theta0p2:
    ax7 = fig.add_subplot(gs[2,2])
    theta02polplot = ax7.pcolormesh(xp2k, zp2k, theta0p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax7.set_label('$x (GM/c^2)$')
    ax7.set_ylabel('$z (GM/c^2)$')
    ax7.set_xlim(domain[:2])
    ax7.set_ylim(domain[2:])
    ax7.set_title('Kawazura',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax7.add_artist(circle)
    ax7.set_aspect('equal')
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta02polplot, cax=cax)

    #plotting theta1p2:
    ax8 = fig.add_subplot(gs[2,3])
    theta12polplot = ax8.pcolormesh(xp2k, zp2k, theta1p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax8.set_label('$x (GM/c^2)$')
    ax8.set_ylabel('$z (GM/c^2)$')
    ax8.set_xlim(domain[:2])
    ax8.set_ylim(domain[2:])
    ax8.set_title('Werner',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax8.add_artist(circle)
    ax8.set_aspect('equal')
    divider = make_axes_locatable(ax8)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta12polplot, cax=cax)

    #plotting theta2p2:
    ax9 = fig.add_subplot(gs[2,4])
    theta22polplot = ax9.pcolormesh(xp2k, zp2k, theta2p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax9.set_label('$x (GM/c^2)$')
    ax9.set_ylabel('$z (GM/c^2)$')
    ax9.set_xlim(domain[:2])
    ax9.set_ylim(domain[2:])
    ax9.set_title('Rowan',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax9.add_artist(circle)
    ax9.set_aspect('equal')
    divider = make_axes_locatable(ax9)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta22polplot, cax=cax)

    #plotting theta3p2:
    ax10 = fig.add_subplot(gs[2,5])
    theta32polplot = ax10.pcolormesh(xp2k, zp2k, theta3p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax10.set_label('$x (GM/c^2)$')
    ax10.set_ylabel('$z (GM/c^2)$')
    ax10.set_xlim(domain[:2])
    ax10.set_ylim(domain[2:])
    ax10.set_title('Sharma',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax10.add_artist(circle)
    ax10.set_aspect('equal')
    divider = make_axes_locatable(ax10)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta32polplot, cax=cax)

    plt.tight_layout()
    plt.savefig("temp_plot_init_quick.png")
    plt.close()

# main(): Reads param file, writes grid dict and calls analysis function
if __name__=="__main__":

    # Setting grid dict
    dumpk = fluid_dump.load_dump("./torus.out0.{0:05}.phdf".format(0))
    grid['n1k'] = dumpk['nx1']; grid['n2k'] = dumpk['nx2']; grid['n3k'] = dumpk['nx3']
    grid['ak'] = dumpk['a']
    grid['rEHk'] = dumpk['r_eh']
    grid['phik'] = dumpk['phi']
    grid['xk'] = np.squeeze(dumpk['x']); grid['zk'] = np.squeeze(dumpk['z'])

    gfile = h5py.File(os.path.join('./dumps_iharm_3d_without','grid.h5'),'r')
    dfile = h5py.File(os.path.join('dumps_iharm_3d_without','dump_0000{0:04d}.h5'.format(0)),'r')
    grid['n1'] = dfile['/header/n1'][()]; grid['n2'] = dfile['/header/n2'][()]; grid['n3'] = dfile['/header/n3'][()]
    grid['dx1'] = dfile['/header/geom/dx1'][()]; grid['dx2'] = dfile['/header/geom/dx2'][()]; grid['dx3'] = dfile['/header/geom/dx3'][()]
    grid['startx1'] = dfile['header/geom/startx1'][()]; grid['startx2'] = dfile['header/geom/startx2'][()]; grid['startx3'] = dfile['header/geom/startx3'][()]
    grid['metric'] = dfile['header/metric'][()].decode('UTF-8')
    if grid['metric']=='MKS' or grid['metric']=='MMKS':
        try:
            grid['a'] = dfile['header/geom/mks/a'][()]
        except KeyError:
            grid['a'] = dfile['header/geom/mmks/a'][()]
        try:
            grid['rEH'] = dfile['header/geom/mks/Reh'][()]
        except KeyError:
            pass
        try:
            grid['rEH'] = dfile['header/geom/mks/r_eh'][()]
        except KeyError:
            pass
        try:
            grid['rEH'] = dfile['header/geom/mmks/Reh'][()]
        except KeyError:
            pass
        try:
            grid['rEH'] = dfile['header/geom/mmks/r_eh'][()]
        except KeyError:
            pass
        try:
            grid['hslope'] = dfile['header/geom/mks/hslope'][()]
        except KeyError:
            grid['hslope'] = dfile['header/geom/mmks/hslope'][()]
    if grid['metric']=='MMKS':
        grid['mks_smooth'] = dfile['header/geom/mmks/mks_smooth'][()]
        grid['poly_alpha'] = dfile['header/geom/mmks/poly_alpha'][()]
        grid['poly_xt'] = dfile['header/geom/mmks/poly_xt'][()]
        grid['D'] = (np.pi*grid['poly_xt']**grid['poly_alpha'])/(2*grid['poly_xt']**grid['poly_alpha']+(2/(1+grid['poly_alpha'])))
    grid['x1'] = gfile['X1'][()]; grid['x2'] = gfile['X2'][()]; grid['x3'] = gfile['X3'][()]
    grid['r'] = gfile['r'][()]; grid['th'] = gfile['th'][()]; grid['phi'] = gfile['phi'][()]
    grid['x'] = gfile['X'][()]; grid['y'] = gfile['Y'][()]; grid['z'] = gfile['Z'][()]
    grid['gcov'] = gfile['gcov'][()]; grid['gcon'] = gfile['gcon'][()]
    grid['gdet'] = gfile['gdet'][()]
    grid['lapse'] = gfile['lapse'][()]
    dfile.close()
    gfile.close()

    """    ncores = psutil.cpu_count(logical=True)
    pad = 0.25
    nthreads = int(ncores*pad); print("Number of threads: {0:03d}".format(nthreads))

    # Calling analysis function for torus2d
    run_parallel(analysis_torus2d,dlist,nthreads)"""
    analysis_torus2d(0)