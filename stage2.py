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

This file performs stage 2 - geometry extraction.
"""

#%%
""" Import """
import numpy as np
import scipy as sp
import scipy.optimize as spopt
import variables, FCM
from scipy.sparse.linalg import spsolve as solver
#from pypardiso import spsolve as solver

#%%
""" Geometry extraction """
def geomextr(dens):
    
    nelx = variables.nelx
    nely = variables.nely
    nele = variables.nele
    RBF = variables.RBF
    
    X,Y = np.meshgrid(np.linspace(-3,3,7),np.linspace(3,-3,7))
    indices = (X-Y*nelx).reshape(-1).astype(int)
    jA = np.empty(nele*49,)
    sA = np.empty(nele*49,)
    cc = 0
    for ely in range(0,nely):
        for elx in range(0,nelx):
            x = elx + 0.5
            y = ely + 0.5
            imin = max([elx - 3, 0])
            imax = min([elx + 4, nelx])
            jmin = max([ely - 3, 0])
            jmax = min([ely + 4, nely])
            Rexp = np.zeros((7,7))
            Rexp[max([0,3-elx]):min([7,nelx-elx+3]),max(0,3-ely):min([7,nely-ely+3])] = \
                 np.exp(-(RBF[imin:imax,jmin:jmax,0]-x)**2-(RBF[imin:imax,jmin:jmax,1]-y)**2)            
            jA[cc*49:(cc+1)*49] = indices #column, RBF
            sA[cc*49:(cc+1)*49] = Rexp.T.reshape(-1)
            cc = cc + 1
            indices = indices + 1
    iA = np.repeat(np.arange(nele),49).astype(int)
    jA = np.minimum(np.maximum(jA,0),nele-1).astype(int)
    A = sp.sparse.coo_matrix((sA,(iA,jA)),shape=(nele,nele)).tocsr()
    
    # Find initial weights LSF
    RBF[:,:,2]=solver(A,dens).reshape(nely,nelx).T    
    
    # Threshold the LSF on a contour value satisfying the volfrac constraint
    phi,wgts = LSF(variables,RBF)
    th = spopt.newton(area,0.5,args=(variables,phi,wgts))
    
    # Change LSF to have solid-void contour on phi=0
    si_th = solver(A,np.ones(nele,)*th).reshape(nely,nelx).T
    RBF[:,:,2] = RBF[:,:,2] - si_th
    
    return RBF

#%%
""" Evaluate level set function at a point (x,y) """
def LSFeval(variables,RBF,x,y):
    
    nelx = variables.nelx
    nely = variables.nely
    
    imin = max([int(x)-3,0])
    imax = min([int(x)+4,nelx])
    jmin = max([int(y)-3,0])
    jmax = min([int(y)+4,nely])
    phi = np.sum(np.exp(-(RBF[imin:imax,jmin:jmax,0]-x)**2-(RBF[imin:imax,\
                        jmin:jmax,1]-y)**2)*RBF[imin:imax,jmin:jmax,2])    
    
    return phi

#%%
""" Compute area of LSF contour (th) """
def LSF(variables,RBF):
    nele = variables.nele
    nod = variables.nod
    ele = variables.ele
    
    # Gauss quadrature
    [pnt,wgt] = FCM.Gauss_scheme(1,2) #P=1, dim=2D
    
    # Numerical integration to compute volume
    phi = np.zeros((nele*len(wgt),))
    for e in range(nele):
        for gp in range(len(wgt)): #for the number of int points in element
            N = FCM.N(pnt[gp,:],1) #P=1
            x,y = np.dot(N,nod[ele[e,:],:]) #loc int point in global coord
            phi[e*len(wgt)+gp] = LSFeval(variables,RBF,x,y) #LSF at int point
            
    wgts = np.tile(wgt,nele)
        
    return phi,wgts

#%%
""" Compute error in volume within LSF contour (th) """
def area(th,variables,phi,wgts):
    nele = variables.nele
    volfrac = variables.volfrac
    rhomin = variables.rhomin
    kappa = variables.kappa
    
    # Density using Heaviside
    rho = rhomin+(1-rhomin)*(1/(1+np.exp(-kappa*(phi-th))))
    
    #sum contributions of all integration points scaled with density
    V = 0.25*rho.dot(wgts)
    g = V/nele-volfrac
        
    return g

#%%
""" end """

