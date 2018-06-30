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

This file contains two optimizers: 
    - Optimality Criterion (OC) implemented by N. Aage and V. Egede Johansen
    - Method of Moving Asymptotes (MMA, Svanberg 2001) implemented by ....
"""

#%%
""" Import """
import numpy as np

#%%
""" 
##                              OPTIMIZERS                                   ##
"""

#%%
""" Optimality criterion (OC) """
class OC:
    """OC optimizer class."""
    
    def __init__(self,dmax=float('Infinity'),dmin=float('-Infinity'),movelimit=0.2):
        """Initialize the OC optimizer."""
        
        # max, min values of design variables
        self.dmax = dmax
        self.dmin = dmin
    
        # move limit
        self.movelimit = movelimit
    
    def solve(self,x,f,df,g,dg,gOC):
        l1=0
        l2=1e9
        xoc=np.zeros(len(x),)
        while (l2-l1)/(l1+l2)>1e-3:
            lmid=0.5*(l2+l1)
            xoc=np.maximum(self.dmin,np.maximum(x-self.movelimit,\
                        np.minimum(self.dmax,np.minimum(x+self.movelimit,\
                                                x*np.sqrt(-df/dg/lmid)))))
            gt=gOC+np.sum((dg*(xoc-x)))
            if gt>0:
                l1=lmid
            else:
                l2=lmid
        Delta_d = xoc-x
        return Delta_d,gt

#%% 
""" Method of Moving Asymptotes (MMA) """
class MMA:
    """MMA optimizer class."""

    def __init__(self,ndv,dmax=float('Infinity'),dmin=float('-Infinity'),
                 movelimit=1.0,asy=(0.5, 1.20, 0.65),scaling=True):
        """Initialize the MMA optimizer."""
        # init iterations to zero
        self.iter = 0

        # max, min values of design variables
        self.dmax = dmax
        self.dmin = dmin

        # init bounds
        self.ub = dmax * np.ones((ndv, 1))
        self.lb = dmin * np.ones((ndv, 1))

        # move limit (± as fraction of current values)
        self.movelimit = movelimit

        # init storage
        self.xold2 = np.zeros((ndv, 1))
        self.xold1 = np.zeros((ndv, 1))

        # asymptotes steps
        self.asyinit, self.asyincr, self.asydecr = asy

        # ?? value
        self.feps = 0.000001
        self.albefa = 0.1

        # scaling of objective and sensitivities
        """ When not specified, the optimizer will scale the objective to
            equal 100 at the first iteration. This is to match with the MMA
            guidelines. However, it's also possible to introduce your own
            scaling methods. Then, provide scaling = False when initializing
            the MMA class, and implement your own scaling directly onto your
            objective (f) function and sensitivities (df).
        """
        self.scale = scaling
        self.scaling = 1

    def solve(self,x,f,dfdx,g,dgdx,iter,dfdx2=None,dgdx2=None):
        """Solve a single iteration with MMA."""
        # determine scaling factor at first iteration
        if self.scale and iter == 1:
            self.scaling = np.divide(100, f)
        
        dgdx = dgdx.reshape(-1,1)
        
        # apply scaling to objective and sensitivity values
        f *= self.scaling
        dfdx *= self.scaling

        # store input format for robust ouput
        self.outputFormat = x.shape

        # convert to required format
        if len(x.shape) == 1:
            # convert to vector
            x = x.reshape(x.shape[0], 1)

        if len(dfdx.shape) == 1:
            dfdx = dfdx.reshape(dfdx.shape[0], 1)

        # set second derivatives of objective function to zero when not specified
        if dfdx2 is None:
            dfdx2 = 0 * dfdx

        # constraints must come as row
        dgdx = np.transpose(dgdx)
        if dgdx2 is None:
            dgdx2 = 0 * dgdx

        # actual optimization
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, self.low, self.upp = \
            self.mmasub(x, dfdx, dfdx2, g, dgdx, dgdx2)

        # compute design update
        Delta_d = xmma - x

        # return results in input format
        Delta_d = Delta_d.reshape(self.outputFormat)

        # reset scaling of the objective and sensitivities
        f *= 1./self.scaling
        dfdx *= 1./self.scaling

        return Delta_d

    def mmasub(self, xmma, dfdx, dfdx2, fval, dgdx, dgdx2):
        """Solve the subproblem."""
        # keep track of iterations
        self.iter += 1

        # number of design variables
        n = xmma.shape[0]

        # number of constraints
        m = 1#fval.size

        # change names, c is herre not the constraints just a parameter
        self.c = 5000*np.ones((m, 1)) #5000
        self.d = np.zeros((m, 1))    # % used internally by MMA
        self.a0 = np.ones((1, 1))     # % used internally by MMA
        self.a = np.zeros((m, 1))    # % used internally by MMA

        # set design vars
        xval = xmma

        # set bounds
        xmin = np.maximum(xmma-self.movelimit, self.lb)
        xmax = np.minimum(xmma+self.movelimit, self.ub)

        # inits
        epsimin = np.sqrt(m+n) * 10**(-9)

        # Calculation of the asymptotes self.low and self.upp :
        if self.iter < 2.5:
            # new asymptotes
            self.low = xval - self.asyinit * (xmax-xmin)
            self.upp = xval + self.asyinit * (xmax-xmin)
        else:
            # for iterations 3 and higher
            zzz = (xval-self.xold1)*(self.xold1-self.xold2)

            # compute factor
            factor = np.ones((n, 1))
            factor[np.where(zzz > 0)] = self.asyincr
            factor[np.where(zzz < 0)] = self.asydecr

            # new asymptotes
            self.low = xval - factor * (self.xold1 - self.low)
            self.upp = xval + factor * (self.upp - self.xold1)

        # Calculation of the bounds alfa and beta
        zz1 = self.low + self.albefa * (xval-self.low)
        zz2 = self.upp - self.albefa * (self.upp-xval)

        alfa = np.maximum(zz1, xmin)
        beta = np.minimum(zz2, xmax)

        # Calculations of p0, q0, P, Q and b.
        ux1 = self.upp - xval
        ux2 = ux1 * ux1
        ux3 = ux2 * ux1

        xl1 = xval-self.low
        xl2 = xl1 * xl1
        xl3 = xl2 * xl1

        ul1 = self.upp-self.low
        ulinv1 = np.reciprocal(ul1)
        uxinv1 = np.reciprocal(ux1)
        xlinv1 = np.reciprocal(xl1)
        uxinv3 = np.reciprocal(ux3)
        xlinv3 = np.reciprocal(xl3)
        diap = (ux3 * xl1) / (2*ul1)
        diaq = (ux1 * xl3) / (2*ul1)

        p0 = np.zeros((n,1))
        p0[dfdx > 0] = dfdx[dfdx > 0]
        p0 = p0 + 0.001 * np.absolute(dfdx) + self.feps * ulinv1
        p0 = p0 * ux2

        q0 = np.zeros((n,1))
        q0[dfdx < 0] = - dfdx[dfdx < 0]
        q0 = q0 + 0.001* np.absolute(dfdx) + self.feps*ulinv1
        q0 = q0 * xl2

        dg0dx2 = 2*(p0 / ux3 + q0 / xl3)
        del0 = dfdx2 - dg0dx2

        delpos0 = np.zeros((n,1))
        delpos0[del0 > 0] = del0[del0 > 0]
        p0 = p0 + delpos0 * diap
        q0 = q0 + delpos0 * diaq

        P = np.zeros((m,n))
        P[dgdx > 0] = dgdx[dgdx > 0]
        P = P * ux2.flatten()

        Q = np.zeros((m,n))
        Q[dgdx < 0] = -dgdx[dgdx < 0]
        Q = Q * xl2.flatten()

        dgdx2 = 2*(P*np.diag(uxinv3) + Q*np.diag(xlinv3))
        dell = np.zeros((m,n))      # these lines here might cause troubles for large systems;
        dell = dfdx2 - dgdx2
        delpos = np.zeros((m,n))
        delpos[np.where(dell > 0)] = dell[np.where(dell > 0)]

        P = P + delpos * diap.flatten()
        Q = Q + delpos * diaq.flatten()
        b = np.dot(P,uxinv1) + np.dot(Q,xlinv1) - fval
        self.b=b[0] # temp, to get scalar

        # store previous results
        self.xold2[:] = self.xold1[:]
        self.xold1[:] = xmma[:] # xmma will be updated in the next statement

        # Solving the subproblem by a primal-dual Newton method
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s = self.subsolve(m,n,epsimin,alfa,beta,p0,q0,P,Q)

        return xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,self.low,self.upp

    # subsolve.m
    def subsolve(self,m,n,epsimin,alfa,beta,p0,q0,P,Q):

        # init variables
        een     = np.ones((n,1))
        eem     = np.ones((m,1))
        epsi    = 1
        epsvecn = epsi * np.ones((n,1))
        epsvecm = epsi * np.ones((m,1))
        x       = 0.5*(alfa+beta)
        y       = np.ones((m,1))
        z       = np.ones((1,1))
        lam     = np.ones((m,1))
        xsi     = np.reciprocal(x-alfa)
        xsi     = np.maximum(xsi,een)
        eta     = np.reciprocal(beta-x)
        eta     = np.maximum(eta,een)
        mu      = np.maximum(eem,0.5*self.c)
        zet     = np.ones((1,1))
        s       = np.ones((m,1))
        itera   = 0

        while epsi > epsimin:

            epsvecn.fill(epsi)
            epsvecm.fill(epsi)
            ux1     = self.upp-x
            xl1     = x-self.low
            ux2     = ux1 * ux1
            xl2     = xl1 * xl1
            uxinv1  = np.reciprocal(ux1)
            xlinv1  = np.reciprocal(xl1)

            plam    = p0 + np.transpose(P) * lam
            qlam    = q0 + np.transpose(Q) * lam
            gvec    = np.dot(P , uxinv1) + np.dot(Q , xlinv1)
            dpsidx  = plam / ux2 - qlam / xl2

            rex     = dpsidx - xsi + eta
            rey     = self.c + self.d * y - mu - lam
            rez     = self.a0 - zet - np.transpose(self.a)*lam;
            relam   = gvec - np.dot(self.a , z) - y + s - self.b;
            rexsi   = xsi * (x-alfa) - epsvecn;
            reeta   = eta * (beta-x) - epsvecn;
            remu    = mu *y - epsvecm;
            rezet   = zet*z - epsi;
            res     = lam *s - epsvecm;

            residu1 = np.concatenate((rex,rey,rez))
            residu2 = np.concatenate((relam,rexsi,reeta,remu,rezet,res))

            residu  = np.concatenate((residu1, residu2))

            #compute norm & norm
            residunorm = np.sqrt((residu*residu).sum())
            residumax   = np.max(np.absolute(residu))

            ittt = 0

            while residumax > 0.9*epsi and ittt < 200:
                ittt  = ittt + 1;
                itera = itera + 1;

                ux1 = self.upp-x;
                xl1 = x-self.low;
                ux2 = ux1*ux1;
                xl2 = xl1*xl1;
                ux3 = ux1*ux2;
                xl3 = xl1*xl2;
                uxinv1 = np.reciprocal(ux1);

                xlinv1 = np.reciprocal(xl1);
                uxinv2 = np.reciprocal(ux2);
                xlinv2 = np.reciprocal(xl2);

                plam    = p0 + np.dot(np.transpose(P) , lam)
                qlam    = q0 + np.dot(np.transpose(Q) , lam)
                gvec    = np.dot(P , uxinv1) + np.dot(Q , xlinv1)

                # Note: NumpPy broadcasting interprets these multiplications as if multiplying by a diagonal matrix!
                GG = P * uxinv2.flatten() - Q * xlinv2.flatten()

                dpsidx = plam / ux2 - qlam / xl2
                delx = dpsidx - epsvecn / (x-alfa) + epsvecn /(beta-x);
                dely = self.c + self.d * y - lam - epsvecm /y;
                delz = self.a0 - np.dot(np.transpose(self.a) , lam) - epsi/z;
                dellam = gvec - self.a*z - y - self.b + epsvecm/lam;

                diagx = plam / ux3 + qlam / xl3;
                diagx = 2*diagx + xsi /(x-alfa) + eta /(beta-x);
                diagxinv = np.reciprocal(diagx);
                diagy = self.d + mu /y;
                diagyinv = np.reciprocal(diagy);
                diaglam = s /lam;
                diaglamyi = diaglam+diagyinv;

                # less constraints then design variables
                if m < n:
                    blam = dellam + dely/diagy - np.dot(GG,np.divide(delx,diagx))
                    Alam = np.diagflat(diaglamyi)+np.dot(GG * diagxinv.flatten(), np.transpose(GG))

                    # create right hand side
                    bb = np.transpose(np.concatenate((np.transpose(blam), delz), axis = 1))

                    # create left hand side
                    AA = np.concatenate((np.concatenate((Alam, self.a), axis = 1), np.concatenate(( np.transpose(self.a),-zet/z),axis = 1)))

                    # solve system
                    solut = np.linalg.solve(AA, bb)

                    # store solution
                    dlam = solut[0:m]
                    dz   = np.array([solut[m]])
                    dx   = -delx / diagx - (np.dot(np.transpose(GG), dlam)) / diagx


                # more constraints then design variables
                else:

                    diaglamyiinv = np.reciprocal(diaglamyi);
                    dellamyi     = dellam + dely / diagy;

                    # prepare LHS
                    Axx = np.diagflat(diagx)+np.dot(np.transpose(GG), np.dot(np.diagflat(diaglamyiinv), GG))
                    azz = zet/z + np.transpose(self.a)*(a/diaglamyi)
                    axz = -np.transose(GG)*(a/diaglamyi)

                    # prepare RHS
                    bx = delx + np.transose(GG)*(dellamyi/diaglamyi)
                    bz  = delz - np.transose(a) * (dellamyi/diaglamyi)

                    # create LHS
                    AA = np.vstack(( np.hstack(( Axx, axz )),np.hstack(( np.transpose(axz),azz )) ))

                    # create RHS
                    bb = np.transpose(np.hstack( (-np.transpose(bx), -bz) ))

                    # solve system
                    solut = numpy.linalg.solve(AA, bb)

                    # store solution
                    dx   = solut[1:n]
                    dz   = np.array([solut[n+1]])
                    dlam = (GG*dx) / diaglamyi - dz*(a /diaglamyi) + dellamyi /diaglamyi


                dy = -dely/diagy + dlam/diagy;
                dxsi = -xsi + epsvecn/(x-alfa) - (xsi*dx)/(x-alfa)
                deta = -eta + epsvecn/(beta-x) + (eta*dx)/(beta-x)
                dmu  = -mu + epsvecm/y - (mu*dy)/y
                dzet = -zet + epsi/z - zet*dz/z
                ds   = -s + epsvecm/lam - (s*dlam)/lam


                # construct xx
                xx = np.concatenate((y,z,lam,xsi,eta,mu,zet,s))

                # construct dxx
                dxx = np.concatenate((dy,dz,dlam,dxsi,deta,dmu,dzet,ds))


                stepxx    = -1.01*dxx/xx;
                stmxx     = np.amax(stepxx);
                stepalfa  = -1.01*dx/(x-alfa);
                stmalfa   = np.amax(stepalfa);
                stepbeta  = 1.01*dx/(beta-x);
                stmbeta   = np.amax(stepbeta);
                stmalbe   = np.maximum(stmalfa,stmbeta);
                stmalbexx = np.maximum(stmalbe,stmxx);
                stminv    = np.maximum(stmalbexx,1);
                steg      = 1/stminv;

                xold   =   x;
                yold   =   y;
                zold   =   z;
                lamold =  lam;
                xsiold =  xsi;
                etaold =  eta;
                muold  =  mu;
                zetold =  zet;
                sold   =   s;

                # init loop
                itto = 0;
                resinew = 2*residunorm;

                while(resinew > residunorm and itto < 50):

                    # increment loop
                    itto += 1;

                    x   =   xold + steg*dx;
                    y   =   yold + steg*dy;
                    z   =   zold + steg*dz;
                    lam = lamold + steg*dlam;
                    xsi = xsiold + steg*dxsi;
                    eta = etaold + steg*deta;
                    mu  = muold  + steg*dmu;
                    zet = zetold + steg*dzet;
                    s   =   sold + steg*ds;

                    ux1     = self.upp-x;
                    xl1     = x-self.low;
                    ux2     = ux1*ux1;
                    xl2     = xl1*xl1;
                    uxinv1  = np.reciprocal(ux1);
                    xlinv1  = np.reciprocal(xl1);
                    plam    = p0 + np.dot( np.transpose(P) , lam)
                    qlam    = q0 + np.dot( np.transpose(Q) , lam)
                    gvec    = np.dot(P , uxinv1) + np.dot(Q , xlinv1)
                    dpsidx  = plam/ux2 - qlam/xl2

                    rex     = dpsidx - xsi + eta;
                    rey     = self.c + self.d*y - mu - lam;
                    rez     = self.a0 - zet - np.transpose(self.a)*lam;
                    relam   = gvec - self.a*z - y + s - self.b;
                    rexsi   = xsi*(x-alfa) - epsvecn;
                    reeta   = eta*(beta-x) - epsvecn;
                    remu    = mu*y - epsvecm;
                    rezet   = zet*z - epsi;
                    res     = lam*s - epsvecm;

                    residu1 = np.concatenate((rex,rey,rez))
                    residu2 = np.concatenate((relam,rexsi,reeta,remu,rezet,res))

                    # complete residu
                    residu = np.concatenate((residu1,residu2))

                    #compute norm
                    resinew = np.sqrt((residu*residu).sum())

                    steg = steg/2;

                # after while loop add new residu's
                residunorm = resinew;
                residumax  = np.max(np.absolute(residu));
                steg = 2*steg;

#==============================================================================
#             if ittt > 99:
#                 print("Max iterations in subsolve reached: ittt:", ittt, ", epsi:", epsi)
#==============================================================================

            epsi = 0.1 * epsi

        # print("MMA: {0} inner iterations.".format(itera))

        # After all loops return values
        return x,y,z,lam,xsi,eta,mu,zet,s
