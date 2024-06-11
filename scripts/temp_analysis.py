######################################################################
#
# Simple analysis and plotting script to process problem output
# Plots mhdmodes, bondi and torus problem (2D and 3D)
#
######################################################################

# python3 ./scripts/temp_analysis.py

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

def analysis_torus2d(dumpval, cmap='jet', vmin=5, vmax=12, domain = [-50,0,-50,50], bh=True, shading='gouraud'):
    plt.clf()
    print("Analyzing {0:04d} dump".format(dumpval))
    dump_without = fluid_dump.load_dump("./dumps_without/torus.out0.{0:05}.phdf".format(dumpval))
    rho = np.squeeze(dump_without['rho'])
    game = 4.0/3.0
    KEL0 = np.squeeze(dump_without['Kel_Howes'])
    KEL1 = np.squeeze(dump_without['Kel_Kawazura'])
    KEL2 = np.squeeze(dump_without['Kel_Werner'])
    KEL3 = np.squeeze(dump_without['Kel_Rowan'])
    KEL4 = np.squeeze(dump_without['Kel_Sharma'])
    uel0 = rho**game*KEL0/(game-1.)
    uel1 = rho**game*KEL1/(game-1.)
    uel2 = rho**game*KEL2/(game-1.)
    uel3 = rho**game*KEL3/(game-1.)
    uel4 = rho**game*KEL4/(game-1.)
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
    theta4 = np.log10((game-1.)*uel4*U_unit/(rho*Ne_unit*Kbol))
    t = dump_without['t']
    t = "{:.3f}".format(t)

    #print('x:', grid['x'])
    xp = xz_slice(grid['x'], dump_without, patch_pole=True)
    zp = xz_slice(grid['z'], dump_without)
    #print('xp:', xp)
    theta0p = xz_slice(theta0, dump_without)
    theta1p = xz_slice(theta1, dump_without)
    theta2p = xz_slice(theta2, dump_without)
    theta3p = xz_slice(theta3, dump_without)
    theta4p = xz_slice(theta4, dump_without)

#**********************************************************************************************************************

    dump_with = fluid_dump.load_dump("./dumps_with/torus.out0.{0:05}.phdf".format(dumpval))
    rho2 = np.squeeze(dump_with['rho'])
    game2 = 4.0/3.0
    KEL02 = np.squeeze(dump_with['Kel_Howes'])
    KEL12 = np.squeeze(dump_with['Kel_Kawazura'])
    KEL22 = np.squeeze(dump_with['Kel_Werner'])
    KEL32 = np.squeeze(dump_with['Kel_Rowan'])
    KEL42 = np.squeeze(dump_with['Kel_Sharma'])
    uel02 = rho2**game2*KEL02/(game2-1.)
    uel12 = rho2**game2*KEL12/(game2-1.)
    uel22 = rho2**game2*KEL22/(game2-1.)
    uel32 = rho2**game2*KEL32/(game2-1.)
    uel42 = rho2**game2*KEL42/(game2-1.)
    theta02 = np.log10((game2-1.)*uel02*U_unit/(rho2*Ne_unit*Kbol))
    theta12 = np.log10((game2-1.)*uel12*U_unit/(rho2*Ne_unit*Kbol))
    theta22 = np.log10((game2-1.)*uel22*U_unit/(rho2*Ne_unit*Kbol))
    theta32 = np.log10((game2-1.)*uel32*U_unit/(rho2*Ne_unit*Kbol))
    theta42 = np.log10((game2-1.)*uel42*U_unit/(rho2*Ne_unit*Kbol))
    t2 = dump_with['t']
    #t2 = "{:.3f}".format(t2)

    xp2 = xz_slice(grid['x'], dump_with, patch_pole=True)
    zp2 = xz_slice(grid['z'], dump_with)
    theta0p2 = xz_slice(theta02, dump_with)
    theta1p2 = xz_slice(theta12, dump_with)
    theta2p2 = xz_slice(theta22, dump_with)
    theta3p2 = xz_slice(theta32, dump_with)
    theta4p2 = xz_slice(theta42, dump_with)

    #just to check that there is a difference:
    max0 = np.amax(theta0.reshape(147456) - theta02.reshape(147456))
    max1 = np.amax(theta1.reshape(147456) - theta12.reshape(147456))
    max2 = np.amax(theta2.reshape(147456) - theta22.reshape(147456))
    max3 = np.amax(theta3.reshape(147456) - theta32.reshape(147456))
    max4 = np.amax(theta3.reshape(147456) - theta32.reshape(147456))
    max0 = "{:.10f}".format(max0)
    max1 = "{:.10f}".format(max1)
    max2 = "{:.10f}".format(max2)
    max3 = "{:.10f}".format(max3)
    max4 = "{:.10f}".format(max4)

    fig = plt.figure(figsize=(16,9))
    heights = [1,6,6]
    gs = gridspec.GridSpec(nrows=3, ncols=5, height_ratios=heights, figure=fig)

    ax0 = fig.add_subplot(gs[0,:])
    ax0.annotate('t= '+str(t)+'; max diff in: Howes: '+str(max0)+', Kawazura: '+str(max1)+', Werner: '+str(max2)+', Rowan: '+str(max3)+', Sharma: '+str(max4),xy=(0.5,0.5),xycoords='axes fraction',va='center',ha='center',fontsize='x-large')
    ax0.axis("off")

    #plotting theta0p:
    ax1 = fig.add_subplot(gs[1,0])
    theta0polplot = ax1.pcolormesh(xp, zp, theta0p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax1.set_xlabel('$x (GM/c^2)$')
    ax1.set_ylabel('$z (GM/c^2)$')
    ax1.set_xlim(domain[:2])
    ax1.set_ylim(domain[2:])
    ax1.set_title('log(temp in K) Howes with Heating',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax1.add_artist(circle)
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta0polplot, cax=cax)

    #plotting theta1p:
    ax2 = fig.add_subplot(gs[1,1])
    theta1polplot = ax2.pcolormesh(xp, zp, theta1p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
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
    theta2polplot = ax3.pcolormesh(xp, zp, theta2p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
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
    theta3polplot = ax4.pcolormesh(xp, zp, theta3p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
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
    ax9 = fig.add_subplot(gs[1,4])
    theta4polplot = ax9.pcolormesh(xp, zp, theta4p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax9.set_xlabel('$x (GM/c^2)$')
    ax9.set_ylabel('$z (GM/c^2)$')
    ax9.set_xlim(domain[:2])
    ax9.set_ylim(domain[2:])
    ax9.set_title('Sharma',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax9.add_artist(circle)
    ax9.set_aspect('equal')
    divider = make_axes_locatable(ax9)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta4polplot, cax=cax)

    #plotting theta0p2:
    ax5 = fig.add_subplot(gs[2,0])
    theta02polplot = ax5.pcolormesh(xp, zp, theta0p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax5.set_label('$x (GM/c^2)$')
    ax5.set_ylabel('$z (GM/c^2)$')
    ax5.set_xlim(domain[:2])
    ax5.set_ylim(domain[2:])
    ax5.set_title('log(temp in K) Howes with Heating & Cooling',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax5.add_artist(circle)
    ax5.set_aspect('equal')
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta02polplot, cax=cax)

    #plotting theta1p2:
    ax6 = fig.add_subplot(gs[2,1])
    theta12polplot = ax6.pcolormesh(xp, zp, theta1p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax6.set_label('$x (GM/c^2)$')
    ax6.set_ylabel('$z (GM/c^2)$')
    ax6.set_xlim(domain[:2])
    ax6.set_ylim(domain[2:])
    ax6.set_title('Kawazura',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax6.add_artist(circle)
    ax6.set_aspect('equal')
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta12polplot, cax=cax)

    #plotting theta2p2:
    ax7 = fig.add_subplot(gs[2,2])
    theta22polplot = ax7.pcolormesh(xp, zp, theta2p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax7.set_label('$x (GM/c^2)$')
    ax7.set_ylabel('$z (GM/c^2)$')
    ax7.set_xlim(domain[:2])
    ax7.set_ylim(domain[2:])
    ax7.set_title('Werner',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax7.add_artist(circle)
    ax7.set_aspect('equal')
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta22polplot, cax=cax)

    #plotting theta3p2:
    ax8 = fig.add_subplot(gs[2,3])
    theta32polplot = ax8.pcolormesh(xp, zp, theta3p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax8.set_label('$x (GM/c^2)$')
    ax8.set_ylabel('$z (GM/c^2)$')
    ax8.set_xlim(domain[:2])
    ax8.set_ylim(domain[2:])
    ax8.set_title('Rowan',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax8.add_artist(circle)
    ax8.set_aspect('equal')
    divider = make_axes_locatable(ax8)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta32polplot, cax=cax)

    #plotting theta3p2:
    ax10 = fig.add_subplot(gs[2,4])
    theta42polplot = ax10.pcolormesh(xp, zp, theta4p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
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
    plt.colorbar(theta42polplot, cax=cax)

    plt.tight_layout()
    plt.savefig("theta_plot_v2_{0:03}".format(dumpval))
    plt.close()

# main(): Reads param file, writes grid dict and calls analysis function
if __name__=="__main__":

    # Calculating total dump files
    dstart = int(sorted(os.listdir('./dumps_without'))[1][11:16])
    dend = 600#int(sorted(os.listdir('./dumps_without'))[-1][11:16])
    dlist = range(dstart,dend+1)

    # Setting grid dict
    dump = fluid_dump.load_dump("./dumps_with/torus.out0.{0:05}.phdf".format(0))
    grid['n1'] = dump['nx1']; grid['n2'] = dump['nx2']; grid['n3'] = dump['nx3']
    grid['a'] = dump['a']
    grid['rEH'] = dump['r_eh']
    grid['phi'] = dump['phi']
    grid['x'] = np.squeeze(dump['x']); grid['z'] = np.squeeze(dump['z'])
    ncores = psutil.cpu_count(logical=True)
    pad = 0.25
    nthreads = int(ncores*pad); print("Number of threads: {0:03d}".format(nthreads))

    # Calling analysis function for torus2d
    run_parallel(analysis_torus2d,dlist,nthreads)