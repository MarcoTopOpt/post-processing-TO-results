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

This file performs stage 3 - shape optimization.
"""

#%%
""" Import """
import numpy as np
import scipy.sparse as sp
import variables, FCM, optimizers
from scipy.sparse.linalg import spsolve as solver
#from pypardiso import spsolve as solver

#%%
""" Shape optimization """
def shapeopt(RBF,Kinit,Vinit,iel,LSFip):
    # Variables
    nelx = variables.nelx
    nely = variables.nely
    f = variables.f
    si_max = variables.si_max
    kappa = variables.kappa
    rhomin = variables.rhomin
    edof = variables.edof
    
    # Discard/free DOFs
    ndof=np.max(edof)+1
    dofs=np.arange(ndof) 
    disc=np.setdiff1d(edof[iel==0,:],edof[iel!=0,:])
    free=np.setdiff1d(dofs,np.append(variables.fix,disc))
    dofs[np.append(variables.fix,disc)]=-1
        
    # Optimizer initialization (MMA)
    mma = optimizers.MMA(variables.nele,dmax=si_max,dmin=-si_max,
                                movelimit=0.2,asy=(0.5,1.2,0.65),scaling=True)
    x = RBF[:,:,2].T.reshape(-1).copy()
    iter = 1
    while iter<10.5:
        # Stiffness matrix
        phi = LSFip.dot(x[:])
        rho = rhomin+(1-rhomin)*(1/(1+np.exp(-kappa*phi)))
        K = FCM.K(Kinit,iel,rho)
                
        # Reverse Cuthill-McKee ordering
        ind = sp.csgraph.reverse_cuthill_mckee(K,symmetric_mode=True)
        free = dofs[ind][dofs[ind]!=-1]
        
        # Solve (reduced) system
        u = np.zeros((np.size(K,0),))
        u[free]=solver(K[free,:][:,free],f[free])
        
        # Compliance objective and its sensitivity
        [C,dC] = obj(Kinit,iel,LSFip,u,phi,rho,f,free)
        
        # Volume fraction constraint and its sensitivity
        [g,dg] = constr(Vinit,iel,LSFip,phi,rho)
        
        # Shape optimization
        dx = mma.solve(x,C,dC,g,dg,iter)
        x = x + dx
        
        # Update log
        print(['%.3f' % i for i in [iter,C,g]])
        iter += 1

    RBF[:,:,2] = x.reshape(nely,nelx).T
    return RBF

#%%
""" Compliance objective and its sensitivity """
def obj(Kinit,iel,LSFip,u,phi,rho,f,free):
    
    # Import variables
    nele = variables.nele
    P = variables.P
    qt = variables.qt
    rhomin = variables.rhomin
    kappa = variables.kappa
    pen = variables.pen_LSM
    edof = variables.edof
    
    # Compliance objective
    C = f[free].T.dot(u[free])
    
    # Density wrt LSF
    drho_dphi = (1-rhomin)*((kappa*np.exp(-kappa*phi))/((1+np.exp(-kappa*phi))**2))
    
    # Compliance wrt density
    dC_drho = np.zeros(len(rho),)
    num = 0
    for el in range(nele): #for the number of elements
        cc = 0
        if iel[el] != 0:  #if element is not discarded
            i0 = int('0'.ljust(1+(iel[el]==2)*qt,'1'),4)*(P+1)**2
            i1 = int('0'.ljust(1+(iel[el]==1)+(iel[el]==2)*(qt+1),'1'),4)*(P+1)**2
            for j in range(i0,i1): #for number of int points
                dC_drho[num+cc]=(-pen*(rho[num+cc]**(pen-1)))*\
                   (u[edof[el,:]].dot(Kinit[:,:,j]).dot(u[edof[el,:]]))
                cc = cc + 1
            num = num+(i1-i0)
            
    # Total derivative: compliance wrt variables
    dC = (LSFip.transpose().dot(drho_dphi.transpose()*dC_drho.transpose())).transpose()
    
    return C,dC

#%%
""" Volume fraction constraint and its sensitivity """
def constr(Vinit,iel,LSFip,phi,rho):
    
    # Import variables
    nele = variables.nele
    qt = variables.qt
    P = variables.P
    volfrac = variables.volfrac
    rhomin = variables.rhomin
    kappa = variables.kappa
    
    V = 0
    dV_drho = np.zeros((len(rho),))
    num = 0
    for el in range(nele): #for the number of elements
        cc = 0
        if iel[el] != 0:  #if element is not discarded
            i0 = int('0'.ljust(1+(iel[el]==2)*qt,'1'),4)*(P+1)**2
            i1 = int('0'.ljust(1+(iel[el]==1)+(iel[el]==2)*(qt+1),'1'),4)*(P+1)**2
            for j in range(i0,i1): #for number of int points
                V=V+rho[num+cc]*Vinit[j] #volume
                dV_drho[num+cc] = Vinit[j] #volume wrt density
                cc = cc + 1
            num = num+(i1-i0)
    
    # Volume constraint
    g = ((V/nele)/volfrac)-1
    
    # Constraint wrt volume
    dg_dV = (1/nele)/volfrac
    
    # Density wrt LSF
    drho_dphi = (1-rhomin)*((kappa*np.exp(-kappa*phi))/((1+np.exp(-kappa*phi))**2))

    # Total derivative: volume constraint wrt variables
    dg = (LSFip.transpose().dot(drho_dphi.transpose()*dV_drho.transpose()*dg_dV)).transpose()
    
    return g,dg

#%%
""" end """