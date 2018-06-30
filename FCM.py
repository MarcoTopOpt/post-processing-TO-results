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

This is the file used to create a mesh, assemble the stiffness matrix using 
pFEM and to create a quadtree integration band.
"""

#%%
""" Import """
import numpy as np
import scipy.sparse as sp
import scipy.special as spec
import variables, stage2

#%% 
""" FCM/LSF mesh """
def mesh(nelx,nely,nele,P,case):
    
    nod = np.zeros(((nelx+1)*(nely+1),2))
    [i,j]=np.meshgrid(np.arange(nelx+1),np.arange(nely+1))
    nod[:,0]=i.reshape(-1); nod[:,1]=j.reshape(-1)
    
    RBF = np.empty((nelx,nely,3))
    RBF[:,:,1],RBF[:,:,0] = np.meshgrid(np.arange(0.5,nely),np.arange(0.5,nelx))
    
    ele = np.empty((nele,4))
    ndof = int((4+4*(P-1)+(np.maximum(0,P-3)*(P-2))/2)*2) #dof per element
    dofnod = 2*(nelx+1)*(nely+1)
    dofedg = 2*((nelx+1)*nely+nelx*(nely+1))
    edof = np.empty((nele,ndof))
    for ely in range(nely):
        for elx in range(nelx):
            el = elx + ely*nelx
            n1 = elx+(ely*(nelx+1))
            n2 = n1+(nelx+1)
            ele[el,:] = np.array([n1,n1+1,n2+1,n2])
            
            d1 = 2*n1
            d2 = 2*n2
            edof[el,0:8] = np.array([d1,d1+1,d1+2,d1+3,d2+2,d2+3,d2,d2+1]) #nodal DOF
            cc = 8
            for p in range(2,P+1):
                tmp = dofnod+(p-2)*dofedg+np.maximum(0,(p-4))*(p-3)*nele
                d1 = tmp+2*(elx+ely*(2*nelx+1))
                d2 = tmp+2*(elx+(ely+1)*(2*nelx+1))
                d3 = tmp+2*(elx+nelx+ely*(2*nelx+1))
                edof[el,cc:cc+8] = np.array([d1,d1+1,d3+2,d3+3,d2,d2+1,d3,d3+1]) #edge DOF
                cc = cc + 8
                
                for k in range(3,p):
                    tmp = dofnod+(p-1)*dofedg+np.maximum(0,(p-4))*(p-3)*nele
                    d1 = tmp+el*(p-3)*2+(k-3)*2
                    edof[el,cc:cc+2] = np.array([d1,d1+1]) #internal DOF
                    cc = cc + 2
    
    if case == 'canti': # cantilever beam
        # BCs 
        fixnod = np.arange(0,(nely+1)*(nelx+1),(nelx+1))
        fix=np.sort(np.array([2*fixnod,2*fixnod+1]).reshape(-1))
        fixedg = np.arange(nelx,(nely*(2*nelx+1)),(2*nelx+1))
        for p in range(2,P+1):
            tmp = dofnod+(p-2)*dofedg+np.maximum(0,(p-4))*(p-3)*nele
            fix = np.append(fix,np.sort(np.array([tmp+2*fixedg,tmp+2*fixedg+1]).reshape(-1)))
        
        # Force
        nodF = int(np.ceil((nely+1)/2)*(nelx+1)-1) #location force
        F=-1 #magnitude
        ndof=int(np.max(edof))+1
        f = np.zeros((ndof,))
        f[2*nodF+1]=F
    
    if case == 'MBB': # MBB beam
        # BCs
        fixnod = np.append(np.array([nelx]),np.arange(0,(nely+1)*(nelx+1),(nelx+1)))
        fix = np.append(np.array([2*fixnod[0]+1]),np.sort(np.array([2*fixnod[1:]]).reshape(-1)))
        fixedg = np.arange(nelx,(nely*(2*nelx+1)),(2*nelx+1))
        for p in range(2,P+1):
            tmp = dofnod+(p-2)*dofedg+np.maximum(0,(p-4))*(p-3)*nele
            fix = np.append(fix,np.sort(np.array([tmp+2*fixedg]).reshape(-1)))
            
        # Force
        nodF = int(nely*(nelx+1)) #location force
        F=-1 #magnitude
        ndof=int(np.max(edof))+1
        f = np.zeros((ndof,))
        f[2*nodF+1]=F
         
    if case == 'michell': # Michell truss
        # BCs 
        val = np.floor((nely+1)/2)*(nelx+1)+np.floor((nelx+1)/8)
        fixnod = np.array([val,val+1]).astype(int)
        fix=np.sort(np.array([2*fixnod,2*fixnod+1]).reshape(-1))

        fixedg = np.floor((nely+1)/2)*(2*nelx+1)+np.floor((nelx+1)/8) 
        for p in range(2,P+1):
            tmp = dofnod+(p-2)*dofedg+np.maximum(0,(p-4))*(p-3)*nele
            fix = np.append(fix,np.sort(np.array([tmp+2*fixedg,tmp+2*fixedg+1]).reshape(-1)))
        
        # Force
        nodF = int(np.ceil((nely+1)/2)*(nelx+1)-1) #location force
        F=-1 #magnitude
        ndof=int(np.max(edof))+1
        f = np.zeros((ndof,))
        f[2*nodF+1]=F
            
    return nod,ele.astype(int),RBF,edof.astype(int),fix,f

#%%
""" 
##                              pFEM                                         ##
"""

#%%
""" Evaluate psi and its derivative """
def Psi(x,po): #loc, polynomial-order
    [LP,dLP] = spec.lpn(po,x) #Legendre polynomial
    Psi = (1/np.sqrt(4*po-2))*(LP[po]-LP[po-2])
    dPsi = (1/np.sqrt(4*po-2))*(dLP[po]-dLP[po-2])
    return [Psi,dPsi]

#%%
""" Shape functions """
def N(pnt,P): #Gauss point, polynomial order
    nc = np.array([[-1,-1],[1,-1],[1,1],[-1,1]]) #local nodal coords
    ec = np.array([[0,-1],[1,0],[0,1],[-1,0]]) #local edge coords
    
    ndof = int((4+4*(P-1)+(np.maximum(0,P-3)*(P-2))/2))
    N=np.zeros((ndof,))
    cc = 0
    
    # Nodal modes
    for i in range(4): #for all 4 nodes do
        N[cc]=(1/4)*(1+nc[i,0]*pnt[0])*(1+nc[i,1]*pnt[1])
        cc = cc + 1
    
    # Edge modes
    for j in range(2,int(P+1)): #for every P=>2 do
        [Psi_xi,_] = Psi(pnt[0],j)
        [Psi_eta,_] = Psi(pnt[1],j)
        
        for e in range(4): #for all 4 edges do
            N[cc]=(1/2)*(1+ec[e,0]*pnt[0]+ec[e,1]*pnt[1])*\
                       ((1-abs(ec[e,0]))*Psi_xi+(1-abs(ec[e,1]))*Psi_eta)
            cc = cc + 1
        
        # Internal/face modes
        for k in range(2,(j-1)):
            [Psi_xi,_] = Psi(pnt[0],k)
            [Psi_eta,_] = Psi(pnt[1],j-k)
            N[cc] = Psi_xi*Psi_eta
            cc = cc + 1

    return N
    
#%% 
""" Derivatives shape functions """
def dN(pnt,P): #Gauss point, polynomial order
    nc = np.array([[-1,-1],[1,-1],[1,1],[-1,1]]) #local nodal coords
    ec = np.array([[0,-1],[1,0],[0,1],[-1,0]]) #local edge coords
    
    ndof = int((4+4*(P-1)+(np.maximum(0,P-3)*(P-2))/2))
    dxi=np.zeros((ndof,)); deta=np.zeros((ndof,))
    cc = 0
    
    # Nodal modes
    for i in range(4): #for all 4 nodes do
        dxi[i]=(1/4)*(nc[i,0])*(1+nc[i,1]*pnt[1])
        deta[i]=(1/4)*(1+nc[i,0]*pnt[0])*(nc[i,1])
        cc = cc + 1
        
    # Edge modes
    for j in range(2,int(P+1)): #for every P=>2 do
        [Psi_xi,dPsi_xi] = Psi(pnt[0],j)
        [Psi_eta,dPsi_eta] = Psi(pnt[1],j)
        
        for e in range(4): #for all 4 edges do
            dxi[cc]=(1/2)*(ec[e,0]+abs(ec[e,1])+ec[e,1]*pnt[1])*\
                       ((1-abs(ec[e,0]))*dPsi_xi+(1-abs(ec[e,1]))*Psi_eta)
            deta[cc]=(1/2)*(abs(ec[e,0])+ec[e,0]*pnt[0]+ec[e,1])*\
                       ((1-abs(ec[e,0]))*Psi_xi+(1-abs(ec[e,1]))*dPsi_eta)
            cc = cc + 1
        
        # Internal/face modes
        for k in range(2,(j-1)):
            [Psi_xi,dPsi_xi] = Psi(pnt[0],k)
            [Psi_eta,dPsi_eta] = Psi(pnt[1],j-k)
            dxi[cc] = dPsi_xi*Psi_eta
            deta[cc] = Psi_xi*dPsi_eta
            cc = cc + 1
    
    return dxi,deta
    
#%%
""" Initialize stiffness contributions """
# This definition will store all contributions to the element stiffness matrix
# of every integration point. These contributions can be easily scaled with the
# local density in an integration point
def Kinit():
    # Variables
    E = variables.E
    nu = variables.nu
    qt = variables.qt
    P = variables.P
    # Material (constitutive relations)
    D = ((E)/(1-nu**2))*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    # Gauss scheme - get the integration points of a (sub)cell
    [pnts,wgts]=Gauss_scheme(P,2)
    # Initialize
    quadtree = np.array([[-1,-1],[1,-1],[1,1],[-1,1]]) #relative loc subcell
    ndof = int((4+4*(P-1)+(np.maximum(0,P-3)*(P-2))/2)*2)
    nsc = int('0'.ljust(2+qt,'1'),4)
    Kinit = np.zeros((ndof,ndof,nsc*len(wgts)))
    Vinit = np.zeros((nsc*len(wgts),))
    ic = np.zeros((nsc,2)) # centers of subcells (local coordinates)   
    qt = 1 #quadtree level
    qb0 = '00'
    cc = 0
    for i in range(nsc): # for integration cells        
        # distance of cell-center from element-center at (0,0)
        if i == 0: dif = 1
        else: 
            digits = []; tmp = i
            while tmp>=1:
                digits.append(int(tmp % 4))
                tmp /= 4
            qb1 = '0'+''.join(str(e) for e in digits[::-1]) #quaternary base
            qt = np.maximum(qt,len(qb1.replace('0',"",2)))
            dif = 1/(2**qt)
            npc = int(qb0[:-1],4) #number parent cell
            ic[i,:] = ic[npc,:]+dif*quadtree[int(qb0[-1]),:]
            qb0 = qb1
        for j in range(len(wgts)): # for integration points
            dx,dy = dN(ic[i,:]+dif*pnts[j,:],P) #derivative shape func
            B = np.zeros((ndof,3)) #strain matrix initialization
            for k in range(len(dx)): #for number modes
                B[2*k:2*(k+1),:]=np.array([[dx[k],0,dy[k]],[0,dy[k],dx[k]]])
            # Elemental stiffness matrix
            Vinit[cc] = 0.25*(dif**2)*wgts[j]
            Kinit[:,:,cc] = (B).dot(D).dot(B.T)*Vinit[cc]
            cc = cc + 1
    
    return [Kinit,Vinit,ic]

#%%
""" Assembling stiffness matrix """
def K(Kinit,iel,rho):
    
    # Import variables
    nele = variables.nele
    P = variables.P
    qt = variables.qt
    pen = variables.pen_LSM
    edof = variables.edof
    
    # Initialize and assemble global stiffness matrix
    Ndof = np.max(edof)+1
    ndof = int((4+4*(P-1)+(np.maximum(0,P-3)*(P-2))/2)*2)
    cc = 0
    ii = np.zeros(ndof*ndof*nele,)
    jj = np.zeros(ndof*ndof*nele,)
    Kel = np.zeros(ndof*ndof*nele,)
    
    for e in range(nele): #for the number of elements
        if iel[e] != 0:  #if element is not discarded
            i0=int('0'.ljust(1+(iel[e]==2)*qt,'1'),4)*(P+1)**2 
            i1=int('0'.ljust(1+(iel[e]==1)+(iel[e]==2)*(qt+1),'1'),4)*(P+1)**2
            # Penalized density in integration points of this element
            rho_p = (rho[cc:cc+(i1-i0)]**pen)
            # Sum contribution of integration points to element stiffness
            Kel[ndof*ndof*e:ndof*ndof*(e+1)] = np.matrix.flatten(\
                                            np.sum(rho_p*Kinit[:,:,i0:i1],2))
            # Count integration points evaluated so far
            cc = cc+(i1-i0)
            [iK,jK] = np.meshgrid(edof[e,:],edof[e,:])
            ii[ndof*ndof*e:ndof*ndof*(e+1)] = iK.reshape(-1)
            jj[ndof*ndof*e:ndof*ndof*(e+1)] = jK.reshape(-1)
            
    K = sp.csr_matrix((Kel,(ii,jj)),shape=(Ndof,Ndof))

    return K

#%%
""" 
##                           ADAPTIVE INTEGRATION                            ##
"""

#%%
""" Gauss quadrature points/weights """
def Gauss_scheme(P,dim):
    Pn = P+1 
    xi = np.zeros(Pn)
    wgt = np.zeros(Pn)
    m = (Pn+1)//2
    e1 = Pn*(Pn+1)
    for i in range(m):
        t = (4*i+3)*np.pi / (4*Pn+2)
        x0 = (1-(1-1/Pn)/(8*Pn*Pn))*np.cos(t)
        pkm1 = 1
        pk = x0
        for k in range(2,Pn+1):
            t1 = x0*pk
            pkp1 = t1-pkm1-(t1-pkm1)/k+t1
            pkm1 = pk
            pk = pkp1
        den = 1-x0*x0
        d1 = Pn*(pkm1-x0*pk)
        dpn = d1/den
        d2pn = (2*x0*dpn - e1*pk)/den
        d3pn = (4*x0*d2pn+(2-e1)*dpn)/den
        d4pn = (6*x0*d3pn+(6-e1)*d2pn)/den
        u = pk/dpn
        v = d2pn/dpn
        h = -u*(1+0.5*u*(v+u*(v*v-d3pn/(3*dpn))))
        p = pk+h*(dpn+0.5*h*(d2pn+h/3*(d3pn+0.25*h*d4pn)))
        dp = dpn + h*(d2pn+0.5*h*(d3pn+h*d4pn/3))
        h = h-p/dp
        xi[i] = -x0 -h
        fx = d1-h*e1*(pk+0.5*h*(dpn+h/3*(d2pn+0.25*h*(d3pn+0.2*h*d4pn))))
        wgt[i] = 2*(1-xi[i]*xi[i])/(fx*fx)
    # copy values to the rest of the arrays
    xi[-1:-m-1:-1] = -xi[:m]
    wgt[-1:-m-1:-1] = wgt[:m]
    if m+m > Pn:
        xi[m-1] = 0
    pnt = np.zeros((Pn**dim,dim))
    if dim==1:
        pnt = xi.copy()
        wgt = wgt.copy()
    if dim==2:
        x,y = np.meshgrid(xi,xi)
        pnt[:,0] = x.reshape(-1)
        pnt[:,1] = y.reshape(-1)
        tmp1,tmp2 = np.meshgrid(wgt,wgt)
        wgt = np.asarray(tmp1*tmp2).reshape(-1)
    
    return [pnt,wgt]

#%%
""" Quadtree integration band """
def quadtree_band(RBF,ic):
    # Variables
    nod = variables.nod
    ele = variables.ele
    nelx = variables.nelx
    nely = variables.nely
    nele = variables.nele
    cp = np.size(nod,0)
    P = variables.P
    qt = variables.qt
    
                        ## initial boundary guess ##
    phi = np.zeros(cp,)  #LSF evaluated at nodes
    iel = np.zeros(nele,).astype(int) #intersected elements
    for nn in range(cp): #for all points of an element do
        phi[nn]=stage2.LSFeval(variables,RBF,nod[nn,0],nod[nn,1])
        
    for ee in range(nele): #for all elements
        # if an element is intersected then return '1'
        if np.sum(phi[ele[ee,:]]<0)!=4 and np.sum(phi[ele[ee,:]]>0)!=4:
            iel[ee] = 1
        else:
            iel[ee] = 0
    
                    ## intersected and fully solid elements ##
    R = np.zeros(cp,)
    tmp = np.zeros((nele,2))
    tmp[:,0] = RBF[:,:,0].T.reshape(-1)
    tmp[:,1] = RBF[:,:,1].T.reshape(-1)
    for nn in range(cp): #for all corner points (cp) do  
        val = tmp[iel==1,:]-nod[nn,:]
        if len(val):
            R[nn] = np.min(np.sqrt(val[:,0]**2+val[:,1]**2)) #closest RBF for cp
    
    # if cp is within R<1 from an intersected element then return '2'
    iel = np.zeros(nele,).astype(int)
    iel[np.sum(R[ele]<1,1)>=1] = 2 # at least 1 cp within R<1
    iel[(iel!=2) & (phi[ele[:,0]]>0)] = 1 # solid element (but no quadtree)
    
                ## LSF evaluation at integration points ##
    [pnts,wgts] = Gauss_scheme(P,2)
    nip = (np.sum(iel==2)*4**qt + np.sum(iel==1))*(P+1)**2
    jLSF = np.empty(nip*49,)
    sLSF = np.empty(nip*49,)
    X,Y = np.meshgrid(np.linspace(-3,3,7),np.linspace(3,-3,7))
    indices = (X-Y*nelx).reshape(-1).astype(int)
    cc=0
    for e in range(nele):
        # Find relevant RBFs
        xi,yi = tmp[e].astype(int)           
        imin = max([xi-3,0])
        imax = min([xi+4,nelx])
        jmin = max([yi-3,0])
        jmax = min([yi+4,nely])
        # Sub-cells
        sc0 = int('0'.ljust(1+(iel[e]==2)*qt,'1'),4)
        sc1 = int('0'.ljust(1+(iel[e]==1)+(iel[e]==2)*(qt+1),'1'),4)
        for i in range(sc0,sc1): # for integration cells
            if i == 0: dif = 1
            else: dif = 1/(2**qt)
            for j in range(len(wgts)): # for integration points 
                Ni = N(ic[i]+dif*pnts[j,:],1) #shape func (P=1)
                x,y = Ni.dot(nod[ele[e,:],:])
                
                Rexp = np.zeros((7,7))
                Rexp[max(0,3-xi):min(7,nelx-xi+3),max(0,3-yi):min(7,nely-yi+3)]\
                                =np.exp(-(RBF[imin:imax,jmin:jmax,0]-x)**2-\
                                            (RBF[imin:imax,jmin:jmax,1]-y)**2)
                
                jLSF[cc*49:(cc+1)*49] = indices #column, RBF
                sLSF[cc*49:(cc+1)*49] = Rexp.T.reshape(-1)
                cc = cc + 1
        indices = indices + 1
    
    iLSF = np.repeat(np.arange(nip),49).astype(int)
    jLSF = np.minimum(np.maximum(jLSF,0),nele-1).astype(int)
    LSFip = sp.csr_matrix((sLSF,(iLSF,jLSF)),shape=(nip,nele))
        
    return iel,LSFip

#%%
""" end """
