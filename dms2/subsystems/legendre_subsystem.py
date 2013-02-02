'''
Created on January 27, 2013

@author: Andrew-AM

The Legendre polynomial type of coarse grained variable.

'''

'''
Created on January 27, 2013

@author: Andrew-AM
'''
import subsystems
import numpy as np
from scipy.special import legendre
from scipy.linalg import qr
import MDAnalysis as md

class LegendreSubsystem(subsystems.SubSystem):
    """
    A set of CG variables.
    """
    def __init__(self, system, pindices, select):
        """
        Create a legendre subsystem.
        @param system: an object (typically the system this subsystem belongs to)
        which has the following attributes:
        box: an array describing the periodic boundary conditions.
        @param pindices: a N*3 array of Legendre polynomial indices. 
        @param select: a select string used for Universe.selectAtoms which selects 
        the atoms that make up this subsystem.
        """
        self.system = system
        self.select = select
        self.pindices = pindices
        
    def universe_changed(self, universe):
        self.atoms = universe.selectAtoms(self.select)
            
    def frame(self):
        # make a column vector
        ScaledPos = self.atoms.positions / self.system.box
        self._Basis = self.Construct_Basis() # Update this every CG step for now
        
        CG_Pos = self.ComputeCG(self.atoms.positions)
        CG_Vel = self.ComputeCG(self.atoms.velocities)
        CG_For = self.ComputeCG_Forces(self.atoms.forces)

        return (CG_Pos, CG_Vel, CG_For)
    
    def translate(self, values):
        self.atoms.positions += values
        
    def minimized(self):
        pass

    def equilibriated(self):
        pass

    def ComputeCGInv(self,CG):
        """
        Computes atomic positions from CG positions
        Using the simplest scheme for now
        """
        return np.dot(self._Basis,CG)

    def ComputeCG(self,var):
        """
        Computes CG momenta or positions
        CG = U^t * Mass * var
        var could be positions or velocities 
        """
        Utw = (self._Basis.T * self._Masses)
        
        return np.dot(Utw,var)
        
    def ComputeCG_Forces(self,atomic_forces):
        """
        Computes CG forces = U^t * <f>
        for an ensemble average atomic force <f>
        """
        return np.dot(self._Basis.T, atomic_forces)
        
    def Construct_Basis(self, Scaled_Pos):
        """
        Constructs a matrix of orthonormalized legendre basis functions
        of size 3*Natoms x NCG 
        """ 
        
        Masses = np.reshape(self._Masses, [len(self._Masses), 1])
        Basis = np.zeros([Scaled_Pos.shape[0], self.pindices.shape[0]],'f')
        
        for i in xrange(u.shape[1]):
            px = legendre(self.pindices[i,0])(x)
            py = legendre(self.pindeces[i,1])(y)
            pz = legendre(self.pindeces[i,2])(z)
            Basis[:,i] = px * py * pz
            
        WBasis = Basis * np.sqrt(Masses)
        WBasis,r = QR_Decomp(WBasis, 'unormalized')    
        WBasis /= np.sqrt(Masses)
        
        return WBasis

    def QR_Decomp(V,dtype):
        """ 
        QR_Decomp is an experimental function. Should be eventually deleted.
        """
        
        if dtype is 'normalized':
            V,R = qr(V, mode='economic')
        else:
            n,k = V.shape

            for j in xrange(k):
                U = V[:,j].copy()

                for i in xrange(j):
                    U -= (dot(V[:,i],V[:,j]) / norm(V[:,i])**2.0) * V[:,i]

                V[:,j] = U.copy()

            #normalize U; uncomment this line for orthonormalized GS
            for j in xrange(k):
                V[:,j] /= norm(V[:,j])

        return V

def poly_indexes(psum):
    """
    Create 2D array of Legendre polynomial indices with index sum <= psum. 

    For example, if psum is 1, the this returns
    [[0, 0, 0],
     [1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
    Note, the sum of each row is less than or equal to 1.
    """
    a = []
    for n in range(psum + 1):
        for i in range(n+1):
            for j in range(n+1-i):
                a.append([n-i-j, j, i])
    return np.array(a,'i')


def LegendreSubsystemFactory(system, selects, *args):
    """
    create a list of LegendreSubsystems.

    @param system
    @param selects
    @param args: a list of length 1 or 2. The first element is kmax.
    """
    kmax = 0
    pindices = None
    if len(args) == 1:
        kmax = int(args[0])
    elif len(args) == 2:
        kmax = int(args[0])
        toks = str(args[0]).split()
        if len(toks) == 2 and toks[0].lower() == "resid" and toks[1].lower() == "unique":
            groups = [system.universe.selectAtoms(s) for s in selects]
            resids = [resid for g in groups for resid in g.resids()]
            selects = ["resid " + str(resid) for resid in resids]
    else:
        raise ValueError("invalid args")
            
    # test to see if the generated selects work
    [system.universe.selectAtoms(select) for select in selects]

    # create the polynomial indices
    pindices = poly_indices(kmax)

    # the number of CG variables
    # actually, its sufficient to just say nrows * 3 as the 
    # number of columns had better be 3.
    ncg = pindices.shape[0] * pindices.shape[1]

    return (ncg, [LegendreSubsystem(system, pindices, select) for select in selects])
