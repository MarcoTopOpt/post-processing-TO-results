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

This file sets the main variables for the entire design optimization procedure. 
Material properties, domain size, optimization variables, analysis variables, 
boundary conditions and level set properties are all defined in this file.
"""

#%%
""" Import """
import numpy as np
import scipy.optimize as spopt
import FCM

#%%
""" Material properties """
E = 1           # Young's modulus
nu = 0          # Poisson's ratio
rhomin = 1e-8   # minimum density

#%%
""" Size domain """
nelx = 64          # number of elements in x-direction
nely = 32          # number of elements in y-direction
nele = nelx*nely   # total number of elements

#%%
""" Case study """
case = 'MBB' #type case: 'canti', 'MBB', 'michell'

#%%
""" Topology optimization (SIMP) - stage one """
volfrac = 0.4    # volume fraction (constraint)
pen_SIMP = 3     # penalization of intermediate densities
rmin = 1.5       # filter radius
ft = 1           # filtering type

#%%
""" Shape optimization - stage three """
# FCM analysis
P = 2  #maximum polynomial order - pFEM
qt = 1 #number of quadtrees - adaptive integration
nod,ele,RBF,edof,fix,f = FCM.mesh(nelx,nely,nele,P,case) # Create an FCM mesh

# Penalization of intermediate densities
pen_LSM = 3

# Bound design variables
si_max = 0.5 #max value design variable

# Steepness LSF
max_grad = 1.24637*1.77264**2*(si_max) #max gradient that can occur
dx = ((1/(2**qt)))/(P+1) #approximate distance between integration points
phi_x = (max_grad*dx)/2  #equivalent of phi at distance dx/2

# Compute kappa using a minimization algorithm
def comp_kappa(kappa,phi_x,ref_dhpi):
    dphi = ((kappa*np.exp(-kappa*phi_x))/((1+np.exp(-kappa*phi_x))**2))
    err = np.abs(ref_dphi-dphi)
    return err

ref_dphi = 0.4    #minimum derivative value in a boundary integration point
kappa = spopt.fmin(comp_kappa,10,args=(phi_x,ref_dphi),disp=0)

#%%
""" end """
