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

The code is intended for educational purposes and theoretical details are
discussed in the paper "Automated and accurate geometry extraction and 
shape optimization of 3D topology optimization results", 
M.K. Swierstra, Computer-Aided-Design 2018, Vol NUMBER, pp. PAGES

Disclaimer:
The author reserves all rights but does not guarantee that the code is free 
from errors. Furthermore, the author shall not be liable in any event caused 
by the use of the program. 

This is the main file for running the topology optimization (stage 1), geometry 
extraction (stage 2) and the shape optimization (stage 3).
"""

#%% 
""" Import """
import numpy as np
import scipy as sp
import variables, stage1, stage2, stage3, FCM, visualize

print("Design domain = "+str(variables.nelx)+"x"+str(variables.nely)+\
", problem = "+variables.case+", volfrac = "+str(variables.volfrac),flush=True)

#%%
""" Obtain topology optimized result using SIMP """
print("Stage 1 - Topology optimization")
# Stage 1- Topology Optimization (SIMP)
dens = stage1.topopt()

# Visualize stage 1
visualize.stage1(dens)

#%%
""" Extract a geometry using an LSF """
print("Stage 2 - Geometry extraction")

# Stage 2 - Obtain initial geometry
RBF = stage2.geomextr(dens)

# Visualize stage 2
visualize.stage2_3(RBF,5,2)

#%%
""" Optimize the shape of the geometry """
print("Stage 3 - Shape optimization")
# Create stiffness matrices (per integration point)
Kinit,Vinit,ic = FCM.Kinit()

# Stage 3 - Shape optimization
steps = 1
while steps<1.5:
    
    # Create quadtree integration band
    print("- (re)create quadtree integration band")
    iel,LSFip = FCM.quadtree_band(RBF,ic)
    
    # Shape optimization
    RBF = stage3.shapeopt(RBF,Kinit,Vinit,iel,LSFip)
    steps += 1
    
# Visualize stage 3
visualize.stage2_3(RBF,5,3)

#%%
""" End """
print("End")
