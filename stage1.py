# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
# EDITED BY MARCO SWIERSTRA, MAY 2018
import numpy as np
import scipy.sparse as sp
import variables, optimizers
from scipy.sparse.linalg import spsolve as solver
#from pypardiso import spsolve as solver
# SIMP Topology Optimization
def topopt():
    # Variables
    E=variables.E; nu=variables.nu; Emin=variables.rhomin #material
    nelx=variables.nelx; nely=variables.nely; nele=variables.nele #domain
    edof=variables.edof #DOFs
    f=variables.f; fix=variables.fix #BCs, loads
    volfrac=variables.volfrac; pen=variables.pen_SIMP #SIMP variables
    rmin=variables.rmin; ft=variables.ft #SIMP variables    
    # Allocate design variables (as array), initialize and allocate sens.
    x=volfrac*np.ones(nele,dtype=float)
    xPhys=x.copy()
    # FE: Build the index vectors for the for coo matrix format.
    KE=lk(E,nu)
    edof=edof[:,0:8] #only nodal DOF
    # Construct the index pointers for the coo format
    iK = np.kron(edof,np.ones((8,1))).flatten().astype(int)
    jK = np.kron(edof,np.ones((1,8))).flatten().astype(int)
    # Filter: Build (and assemble) the index+data svectors for the coo matrix format
    nfilter=int(nele*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc=0
    for i1 in range(nely):
        for j1 in range(nelx):
            row=j1 + i1*nelx #current element
            for i2 in range(int(np.maximum(i1-(np.ceil(rmin)-1),0)),int(np.minimum(i1+np.ceil(rmin),nely))):
                for j2 in range(int(np.maximum(j1-(np.ceil(rmin)-1),0)),int(np.minimum(j1+np.ceil(rmin),nelx))):
                    col=j2 + i2*nelx #neighbour element
                    iH[cc]=row
                    jH[cc]=col
                    sH[cc]=np.maximum(0,rmin-np.sqrt((i1-i2)**2+(j1-j2)**2))
                    cc=cc+1
    # Finalize assembly and convert to csc format
    H=sp.coo_matrix((sH,(iH,jH)),shape=(nele,nele)).tocsc()	
    Hs=H.sum(1)
    # BCs, load, displ
    ndof=2*(nelx+1)*(nely+1)
    dofs=np.arange(ndof)
    fix=fix[fix<=ndof]
    free=np.setdiff1d(dofs,fix)
    # Force vector
    f=f[0:ndof]
    u=np.zeros((ndof,))
    # Initialize optimizers gradient vectors
    mma = optimizers.MMA(nele,dmax=1,dmin=0,movelimit=0.2,asy=(0.5,1.2,0.65),scaling=True)
    #oc = optimizers.OC(dmax=1,dmin=0,movelimit=0.2)
    #gOC = 0
    dg = np.ones(nele)
    dC = np.ones(nele)
    ce = np.ones(nele)
    iter=1
    while iter<50.5:
        # Setup and solve FE problem
        sK=((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**pen*(E-Emin))).flatten(order='F')
        K = sp.coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsr()
        # Solve system
        u[free]=solver(K[free,:][:,free],f[free])
        # Objective and sensitivity
        ce = (np.dot(u[edof].reshape(nele,8),KE)*u[edof].reshape(nele,8)).sum(1)
        C = ((Emin+xPhys**pen*(E-Emin))*ce ).sum()
        dC = (-pen*xPhys**(pen-1)*(E-Emin))*ce
        g = ((np.sum(xPhys)/nele)/volfrac)-1
        dg = np.ones(nele)
        # Sensitivity filtering:
        if ft==0:
            dC=np.asarray((H*(x*dC))[np.newaxis].T/Hs)[:,0]/np.maximum(0.001,x)
        elif ft==1:
            dC=np.asarray(H*(dC[np.newaxis].T/Hs))[:,0]
            dg=np.asarray(H*(dg[np.newaxis].T/Hs))[:,0]
        dx=mma.solve(x,C,dC,g,dg/(volfrac*nele),iter)
        #dx,gOC=oc.solve(x,C,dC,g,dg,gOC)
        x = x+dx
        # Filter design variables
        if ft==0:   xPhys=x
        elif ft==1:	xPhys=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
        # Update log
        print(['%.3f' % i for i in [iter,C,g]])
        iter+=1
    return xPhys

#derivative shape function wrt local coordinates in the Gauss points
def dN(pnt,cds): #Gauss point, nodal coordinates
    dN_dxi=np.zeros((4,)); dN_deta=np.zeros((4,))
    for i in range(np.size(cds,0)): #for all 8 nodes do
        dN_dxi[i]=(1/4)*(cds[i,0])*(1+cds[i,1]*pnt[1])
        dN_deta[i]=(1/4)*(1+cds[i,0]*pnt[0])*(cds[i,1])
    return dN_dxi,dN_deta

#element stiffness matrix (hex8)
def lk(E,nu):
    # Material (constitutive relations)
    D = ((E)/(1-nu**2))*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    # Gauss quadrature
    gp = 1/np.sqrt(3) #Gauss point
    pnts = np.array([[-gp,gp,gp,-gp],[-gp,-gp,gp,gp]]).T
    # Element stiffness matrix
    cds = np.array([[-1,-1],[1,-1],[1,1],[-1,1]]) #local nodal coordinates
    B = np.zeros((8,3))
    KE = np.zeros((8,8))
    for pp in range(4): #for number of int points
        dN_dxi,dN_deta = dN(pnts[pp,:],cds) #eval dN at gauss point
        for nn in range(len(dN_dxi)): #for number of nodes
            #dN_dx=dN_dxi,dN_dy=dN_deta (perfect square elements)
            B[2*nn:2*(nn+1),:]=np.array([[dN_dxi[nn],0,dN_deta[nn]],\
                                         [0,dN_deta[nn],dN_dxi[nn]]])
        KE = KE+0.25*B.dot(D).dot(B.T) #detJ=0.25
    return KE

#%%
""" end """