"""
###############################################################################
##                                                                           ##
##      Name:           M.K. Swierstra                                       ##
##      University:     Delft University of Technology                       ##
##      Faculty:        3mE - Mechanical Engineering                         ##
##      Department:     Precision and Microsystems Engineering               ##
##      Date (d/m/y):   25/06/2018                                           ##
##                                                                           ##
###############################################################################

This file is used to visualize the results obtained at the three stages using 
Matplotlib.
"""

#%%
"""Import"""
import numpy as np
import scipy as sp
import variables, stage2

# Import plotting tools
import matplotlib.pyplot as mpl

# Import coloring
from matplotlib.colors import LinearSegmentedColormap as LSC
blue_map = LSC.from_list('mycmap',['white',tuple([31/255,78/255,121/255])])

#%%
""" PLOTTING FOR .PDF FILES - Matplotlib & Plotly """

#%%
""" Plot stage 1 - topology optimization """ 
def stage1(dens):
    nelx = variables.nelx
    nely = variables.nely
    
    dens = dens.reshape(nely,nelx)
    fig1,ax1 = mpl.subplots()
    fig1.canvas.draw()
    ax1 = ax1.imshow(dens,cmap=blue_map,origin='lower')
    mpl.xticks([],[])
    mpl.yticks([],[])
    mpl.savefig('stage1.pdf',bbox_inhes='tight',pad_inches=0)

#%%
""" Plot stage 2 or 3 - LSF/density """
def stage2_3(RBF,res,stage):
    nelx = variables.nelx
    nely = variables.nely
    rhomin = variables.rhomin
    kappa = variables.kappa
    
    # Sampling points
    nx = res*nelx
    ny = res*nely
    X = np.zeros((nx+1,ny+1))
    Y = np.zeros((nx+1,ny+1))
    phi = np.zeros((nx+1,ny+1))
    rho = np.zeros((nx+1,ny+1))    
    for i in range(0,nx+1): #for all x-points
        for j in range(0,ny+1): #for all y-points
            X[i,j] = i*(nelx/nx)
            Y[i,j] = j*(nely/ny)
            phi[i,j] = stage2.LSFeval(variables,RBF,X[i,j],Y[i,j])
            rho[i,j] = rhomin+(1-rhomin)*(1/(1+np.exp(-kappa*phi[i,j])))
                
    # Plot filled contour    
    fig2,ax2 = mpl.subplots()
    rho[rho<1e-4]=np.nan #force pure white in image
    ax2.contourf(X,Y,rho,cmap=blue_map) #'gray_r'
    ax2.set_aspect('equal')
    ax2.axis((0,nelx,0,nely))
    ax2.set_xticks([], [])
    ax2.set_yticks([], [])
    mpl.savefig('stage'+str(stage)+'.pdf',bbox_inhes='tight',pad_inches=0) # save as .pdf

#%%
"""end"""
