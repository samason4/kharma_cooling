######################################################################
#
# Simple analysis and plotting script to process problem output
# Plots mhdmodes, bondi and torus problem (2D and 3D)
#
######################################################################

# python3 ./scripts/temp_analysis_average.py

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
from multiprocessing import Manager

# Use Manager to handle shared global variables
manager = Manager()
globalvars = manager.dict()

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

# Use Manager to handle shared global variables
manager = Manager()
globalvars = manager.dict()

# Take poloidal slice of 2D array
def xz_slice(var, dump, patch_pole=False):
  xz_var = np.zeros((dump['n1'], dump['n2']))
  for i in range(dump['n1']):
    xz_var[i,:] = var[dump['n1']-1-i,:]
  if patch_pole:
    xz_var[:,0] = xz_var[:,-1] = 0
  return xz_var

def analysis_torus2d(cmap='jet', vmin=5, vmax=12, domain = [-50,0,-50,50], bh=True, shading='gouraud'):
    plt.clf()
    
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

    globalvars['rho'] = np.zeros((grid['n1'], grid['n2']))
    globalvars['KEL0'] = np.zeros((grid['n1'], grid['n2']))
    globalvars['KEL1'] = np.zeros((grid['n1'], grid['n2']))
    globalvars['KEL2'] = np.zeros((grid['n1'], grid['n2']))
    globalvars['KEL3'] = np.zeros((grid['n1'], grid['n2']))
    globalvars['KEL4'] = np.zeros((grid['n1'], grid['n2']))
    globalvars['tracker'] = 0
    
    # Calculating total dump files
    dstart = 0#int(sorted(os.listdir('./dumps_with'))[1][11:16])
    dend = 599#int(sorted(os.listdir('./dumps_with'))[-1][11:16])
    dlist = range(dstart,dend+1)

    ncores = psutil.cpu_count(logical=True)
    pad = 0.25
    nthreads = int(ncores*pad); print("Number of threads: {0:03d}".format(nthreads))

    # Calling analysis function for torus2d
    run_parallel(helper, dlist, nthreads)
    print("tracker before:", globalvars['tracker'])
    globalvars['rho'] /= 600
    globalvars['KEL0'] /= 600
    globalvars['KEL1'] /= 600
    globalvars['KEL2'] /= 600
    globalvars['KEL3'] /= 600
    globalvars['KEL4'] /= 600
    globalvars['tracker'] /= 600
    print("tracker after:", globalvars['tracker'])

    rho = globalvars['rho']
    KEL0 = globalvars['KEL0']
    KEL1 = globalvars['KEL1']
    KEL2 = globalvars['KEL2']
    KEL3 = globalvars['KEL3']
    KEL4 = globalvars['KEL4']

    game = 4.0/3.0

    uel0 = rho**game*KEL0/(game-1.)
    uel1 = rho**game*KEL1/(game-1.)
    uel2 = rho**game*KEL2/(game-1.)
    uel3 = rho**game*KEL3/(game-1.)
    uel4 = rho**game*KEL4/(game-1.)
    tel0 = np.log10((game-1.)*uel0*U_unit/(rho*Ne_unit*Kbol))
    tel1 = np.log10((game-1.)*uel1*U_unit/(rho*Ne_unit*Kbol))
    tel2 = np.log10((game-1.)*uel2*U_unit/(rho*Ne_unit*Kbol))
    tel3 = np.log10((game-1.)*uel3*U_unit/(rho*Ne_unit*Kbol))
    tel4 = np.log10((game-1.)*uel4*U_unit/(rho*Ne_unit*Kbol))

    xp = xz_slice(grid['x'], dump, patch_pole=True)
    zp = xz_slice(grid['z'], dump)
    tel0p = xz_slice(tel0, dump)
    tel1p = xz_slice(tel1, dump)
    tel2p = xz_slice(tel2, dump)
    tel3p = xz_slice(tel3, dump)
    tel4p = xz_slice(tel4, dump)
    rhop = xz_slice(np.log10(rho), dump)

    fig = plt.figure(figsize=(16,6))
    heights = [1,6]
    gs = gridspec.GridSpec(nrows=2, ncols=6, height_ratios=heights, figure=fig)

    ax0 = fig.add_subplot(gs[0,:])
    ax0.annotate('avg Log(Temp in K) with heating and cooling from bh28',xy=(0.5,0.5),xycoords='axes fraction',va='center',ha='center',fontsize='x-large')
    ax0.axis("off")
#****************************************************************************************************************************************************************************************************************************************************************************************************************************
    #plotting theta0p:
    ax1 = fig.add_subplot(gs[1,0])
    theta0polplot = ax1.pcolormesh(xp, zp, tel0p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax1.set_label('$x (GM/c^2)$')
    ax1.set_ylabel('$z (GM/c^2)$')
    ax1.set_xlim(domain[:2])
    ax1.set_ylim(domain[2:])
    ax1.set_title('Howes',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax1.add_artist(circle)
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta0polplot, cax=cax)

    #plotting theta1p:
    ax2 = fig.add_subplot(gs[1,1])
    theta1polplot = ax2.pcolormesh(xp, zp, tel1p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax2.set_xlabel('$x (GM/c^2)$')
    ax2.set_ylabel('$z (GM/c^2)$')
    ax2.set_xlim(domain[:2])
    ax2.set_ylim(domain[2:])
    ax2.set_title('Kawazura',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax2.add_artist(circle)
    ax2.set_aspect('equal')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta1polplot, cax=cax)

    #plotting theta2p:
    ax3 = fig.add_subplot(gs[1,2])
    theta2polplot = ax3.pcolormesh(xp, zp, tel2p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax3.set_xlabel('$x (GM/c^2)$')
    ax3.set_ylabel('$z (GM/c^2)$')
    ax3.set_xlim(domain[:2])
    ax3.set_ylim(domain[2:])
    ax3.set_title('Werner',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax3.add_artist(circle)
    ax3.set_aspect('equal')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta2polplot, cax=cax)
    
    #plotting theta3p:
    ax4 = fig.add_subplot(gs[1,3])
    theta3polplot = ax4.pcolormesh(xp, zp, tel3p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax4.set_xlabel('$x (GM/c^2)$')
    ax4.set_ylabel('$z (GM/c^2)$')
    ax4.set_xlim(domain[:2])
    ax4.set_ylim(domain[2:])
    ax4.set_title('Rowan',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax4.add_artist(circle)
    ax4.set_aspect('equal')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta3polplot, cax=cax)

    #plotting theta4p:
    ax5 = fig.add_subplot(gs[1,4])
    theta3polplot = ax5.pcolormesh(xp, zp, tel4p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax5.set_xlabel('$x (GM/c^2)$')
    ax5.set_ylabel('$z (GM/c^2)$')
    ax5.set_xlim(domain[:2])
    ax5.set_ylim(domain[2:])
    ax5.set_title('Sharma',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax5.add_artist(circle)
    ax5.set_aspect('equal')
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta3polplot, cax=cax)
    
    #plotting rhop:
    ax6 = fig.add_subplot(gs[1,5])
    rhopolplot = ax6.pcolormesh(xp, zp, rhop, cmap=cmap, vmin=-3, vmax=0, shading=shading)
    ax6.set_xlabel('$x (GM/c^2)$')
    ax6.set_ylabel('$z (GM/c^2)$')
    ax6.set_xlim(domain[:2])
    ax6.set_ylim(domain[2:])
    ax6.set_title('log(rho in code units)',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax6.add_artist(circle)
    ax6.set_aspect('equal')
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(rhopolplot, cax=cax)

    plt.tight_layout()
    plt.savefig("temp_average_bh28")
    plt.close()

def helper(dumpno):
    dump_with = fluid_dump.load_dump("./dumps_with/torus.out0.{0:05}.phdf".format(dumpno))
    globalvars['rho'] = globalvars['rho'] + np.squeeze(dump_with['rho'])
    globalvars['KEL0'] = globalvars['KEL0'] + np.squeeze(dump_with['Kel_Howes'])
    globalvars['KEL1'] = globalvars['KEL1'] + np.squeeze(dump_with['Kel_Kawazura'])
    globalvars['KEL2'] = globalvars['KEL2'] + np.squeeze(dump_with['Kel_Werner'])
    globalvars['KEL3'] = globalvars['KEL3'] + np.squeeze(dump_with['Kel_Rowan'])
    globalvars['KEL4'] = globalvars['KEL4'] + np.squeeze(dump_with['Kel_Sharma'])
    globalvars['tracker'] = globalvars['tracker'] + 1
    print("analyzing:", dumpno)
    print("tracker:", globalvars['tracker'])

# main(): Reads param file, writes grid dict and calls analysis function
if __name__=="__main__":

    # Setting grid dict
    dump = fluid_dump.load_dump("./dumps_with/torus.out0.{0:05}.phdf".format(0))
    grid['n1'] = dump['nx1']; grid['n2'] = dump['nx2']; grid['n3'] = dump['nx3']
    grid['a'] = dump['a']
    grid['rEH'] = dump['r_eh']
    grid['phi'] = dump['phi']
    grid['x'] = np.squeeze(dump['x']); grid['z'] = np.squeeze(dump['z'])

    analysis_torus2d()
