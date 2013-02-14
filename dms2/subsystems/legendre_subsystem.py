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
        
        # select statement to get atom group
        self.select = select
        
        # polynomial indices, N_cg x 3 matrix.
        self.pindices = pindices
        
        # Atom-related variables
        self.coords = system.atoms.positions
        
    def universe_changed(self, universe):
        """ 
        universe changed, so select our atom group
        """
        self.atoms = universe.selectAtoms(self.select)
        
    def frame(self):

        self.CG = self.ComputeCG(self.atoms.positions)
        self.CG_Vel = self.ComputeCG(self.atoms.velocities)
        self.CG_For = self.ComputeCG_Forces(self.atoms.forces)

        return (self.CG, self.CG_Vel, self.CG_For)

    def center_of_mass(self, var):
        return np.dot(var,self.atoms.masses) / np.sum(self.atoms.masses)
        
    def translate(self, values):
        self.atoms.positions += values
        
    def minimized(self):
        pass

    def equilibriated(self):
        """
        this is called just after the structure is equilibriated, this is the starting struct
        for the MD runs, is this to calculate basis.
        """
        self.basis = self.Construct_Basis(self.coords)  # Update this every CG step for now

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
        var could be atomic positions or velocities 
        """
        Utw = (self.basis.T * self.atoms.masses)
        
        return np.dot(Utw,var - self.center_of_mass(var))
        
    def ComputeCG_Forces(self,atomic_forces):
        """
        Computes CG forces = U^t * <f>
        for an ensemble average atomic force <f>
        """
        return np.dot(self.basis.T, atomic_forces)
        
    def Construct_Basis(self):
        """
        Constructs a matrix of orthonormalized legendre basis functions
        of size 3*Natoms x NCG 
        """ 
        ScaledPos = (self.coords - self.atoms.centroid()) / self.system.box
        Masses = np.reshape(self.atoms.masses, [len(self.atoms.masses), 1])
        Basis = np.zeros([ScaledPos.shape[0], self.pindices.shape[0]],'f')
        
        for i in xrange(self.pindices.shape[0]):
            px = legendre(self.pindices[i,0])(ScaledPos[:,0])
            py = legendre(self.pindices[i,1])(ScaledPos[:,1])
            pz = legendre(self.pindices[i,2])(ScaledPos[:,2])
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
                U -= (np.dot(V[:,i],V[:,j]) / np.linalg.norm(V[:,i])**2.0) * V[:,i]

            V[:,j] = U.copy()

        #normalize U; comment the two lines below for orthogonal GS
        for j in xrange(k):
            V[:,j] /= np.linalg.norm(V[:,j])

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

    @param system: the system that the subsystem belongs to, this may be None
                   when the simulation file is created. 
    @param selects: A list of MDAnalysis selection strings, one for each
                    subsystem. 
    @param args: a list of length 1 or 2. The first element is kmax, and
                 the second element may be the string "resid unique", which can be
                 thought of as an additional selection string. What it does is 
                 generate a subsystem for each residue. So, for example, select
                 can be just "resname not SOL", to strip off the solvent, then
                 if args is [kmax, "resid unique"], an seperate subsystem is
                 created for each residue. 
    """
    kmax = 0
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
    pindices = poly_indexes(kmax)

    # the number of CG variables
    # actually, its sufficient to just say nrows * 3 as the 
    # number of columns had better be 3.
    ncg = pindices.shape[0] * pindices.shape[1]

    return (ncg, [LegendreSubsystem(system, pindices, select) for select in selects])
