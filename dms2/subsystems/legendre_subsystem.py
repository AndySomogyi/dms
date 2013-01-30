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
import MDAnalysis as md

class LegendreSubsystem(subsystems.SubSystem):
    """
    A set of CG variables.
    """
    def __init__(self, system, select):
        """
        Create a legendre subsystem.
        @param system: an object (typically the system this subsystem belongs to)
        which has the following attributes:
        box: an array describing the periodic boundary conditions.
        @param select: a select string used for Universe.selectAtoms which selects 
        the atoms that make up this subsystem.
        """
        self.system = system
        self.select = select
        
    def universe_changed(self, universe):
        self.atoms = universe.selectAtoms(self.select)
            
    def frame(self):
        # make a column vector
        ScaledPos = self.atoms.positions / self.system.box
        self._Basis = self.Construct_Basis() #Must NOT be updated every CG step
        
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

    def ComputeCG(self,r):
        """
        foo
        calculates \phi = U^t M r 
        for each column in r, i.e. x, y, z
        """
        phi = np.zeros([self._Basis.shape[1], 3], 'f')
        Utw = (self._Basis.T * self._Masses)
        
        phi[:,0] = np.dot(Utw, r[:,0])
        phi[:,1] = np.dot(Utw, r[:,1])
        phi[:,2] = np.dot(Utw, r[:,2])
        
        return phi 
    
    def ComputeCG_Forces(self,atomic_forces):
        """
        foo
        calculates \phi_force = U^t f
        for each column in r, i.e. x, y, z
        """
        ForcePhi = np.zeros([self._U.shape[1], 3], 'f')
        
        ForcePhi[:,0] = np.dot(self._Basis.T, atomic_forces[:,0])
        ForcePhi[:,1] = np.dot(self._Basis.T, atomic_forces[:,1])
        ForcePhi[:,2] = np.dot(self._Basis.T, atomic_forces[:,2])
        
        return ForcePhi
    
    def Construct_Basis(self, Scaled_Pos):
        """
        Constructs a matrix of orthonormalized legendre basis functions
        of size 3*Natoms x NCG 
        """ 
        Indices = self.ComputeIndices(kmax)
        Masses = np.reshape(self._Masses, [len(self._Masses), 1])
        Basis = np.zeros([Scaled_Pos.shape[0], Indices.shape[0]],'f')
        
        for i in xrange(u.shape[1]):
            px = legendre(indexes[i,0])(x)
            py = legendre(indexes[i,1])(y)
            pz = legendre(indexes[i,2])(z)
            Basis[:,i] = px * py * pz
            
        Basis = Basis * sqrt(w)
            
        ONBasis,r = linalg.qr(u, 'full')    
        ONBasis /= sqrt(Masses)
        
        return ONBasis

def LegendreSubsystemFactory(system, selects, *args): 
    
    if len(args) == 1:
        toks = str(args[0]).split()
        if len(toks) == 2 and toks[0].lower() == "resid" and toks[1].lower() == "unique":
            groups = [system.universe.selectAtoms(s) for s in selects]
            resids = [resid for g in groups for resid in g.resids()]
            selects = ["resid " + str(resid) for resid in resids]
            
    # test to see if the generated selects work
    [system.universe.selectAtoms(select) for select in selects]

    return (3, [LegendreSubsystem(system, select) for select in selects])
