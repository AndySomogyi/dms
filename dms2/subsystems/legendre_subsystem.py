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
import logging

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

        note, the system MAY be None, such as when the config is created, so don't
        access it yet.

        @param pindices: a N*3 array of Legendre polynomial indices.
        @param select: a select string used for Universe.selectAtoms which selects
        the atoms that make up this subsystem.
        """
        self.system = system

        # select statement to get atom group
        self.select = select

        # polynomial indices, N_cg x 3 matrix.
        self.pindices = pindices

        # we need to be called with a valid universe
        self.universe_changed(system.universe)

        logging.info("created LegendreSubsystem, pindices: {}, select: {}".
                     format(pindices, select))

    def universe_changed(self, universe):
        """
        universe changed, so select our atom group
        """
        self.atoms = universe.selectAtoms(self.select)

        # check to see if atoms is valid
        if len(self.atoms) <= 0:
            raise ValueError( \
                "The select statement '{}' is syntactically valid but returned a zero sized AtomGroup".format( \
                self.select))

        # call some stuff on the atomGroup to see if its valid
        self.atoms.bbox()

    def ComputeResiduals(self,CG):
        return self.EqAtomPos - self.ComputeCGInv(CG)

    def frame(self):
        """
        Returns a 3 tuple of CG variables, each one is a row vector of size n_cg
        """

        CG = self.ComputeCG(self.atoms.positions)
        CG_Vel = self.ComputeCG(self.atoms.velocities())
        CG_For = self.ComputeCG_Forces(self.atoms.forces)

        CG = np.reshape(CG.T,(CG.shape[0]*CG.shape[1]))
        CG_Vel = np.reshape(CG_Vel.T,(CG_Vel.shape[0]*CG_Vel.shape[1]))
        CG_For = np.reshape(CG_For.T,(CG_For.shape[0]*CG_For.shape[1]))

        return (CG,CG_Vel,CG_For)

    def translate(self, dCG):
        """
        translates the atomic positions from a given vectory of dCG positions,
        where dCG is a finite change in the CG velocities times dt,
        and then adds the residuals for better accuracy.

        @param CG: a length N_cg 1D array.
        """
        self.residuals = self.ComputeResiduals(self.CG)
        self.atoms.positions = self.ComputeCGInv(self.CG + dCG) + self.atoms.centerOfMass() + self.residuals
        # or self.atoms.positions += self.ComputeCGInv(dCG)

    def minimized(self):
        pass

    def equilibriated(self):
        """
        this is called just after the structure is equilibriated, this is the starting struct
        for the MD runs, this is to calculate basis.
        """
        boxboundary = self.atoms.bbox()
        self.box = (boxboundary[1,:] - boxboundary[0,:]) * 0.5
        self.basis = self.Construct_Basis(self.atoms.positions - self.atoms.centerOfMass())  # Update this every CG step for now

        CG = self.ComputeCG(self.atoms.positions)
        self.CG = np.reshape(CG.T,(CG.shape[0]*CG.shape[1]))
        self.EqAtomPos = self.atoms.positions

    def ComputeCGInv(self,CG):
        """
        Computes atomic positions from CG positions
        Using the simplest scheme for now
        @param CG: 3*n_cg x 1 array
        @return: a n_atom x 3array
        """
        NCG = CG.shape[0]/3
        x = self.box[0] / 2.0 * np.dot(self.basis,CG[:NCG])
        y = self.box[1] / 2.0 * np.dot(self.basis,CG[NCG:2*NCG])
        z = self.box[2] / 2.0 * np.dot(self.basis,CG[2*NCG:3*NCG])

        return np.array([x,y,z]).T

    def ComputeCG(self,var):
        """
        Computes CG momenta or positions
        CG = U^t * Mass * var
        var could be atomic positions or velocities
        """
        Utw = self.basis.T * self.atoms.masses()
        return 2.0 / self.box * np.dot(Utw,var - - np.dot(self.atoms.masses(),var) / np.sum(self.atoms.masses()))

    def ComputeCG_Forces(self, atomic_forces):
        """
        Computes CG forces = U^t * <f>
        for an ensemble average atomic force <f>
        """
        return 2.0 / self.box *  np.dot(self.basis.T, atomic_forces - np.dot(self.atoms.masses(),atomic_forces) / np.sum(self.atoms.masses()))

    def Construct_Basis(self,coords):
        """
        Constructs a matrix of orthonormalized legendre basis functions
        of size Natoms x NCG.
        """
        ScaledPos = 2.0 * coords / self.box
        # grab the masses, and make it a column vector
        Masses = self.atoms.masses()[:,np.newaxis]
        Basis = np.zeros([ScaledPos.shape[0], self.pindices.shape[0]],'f')

        for i in xrange(self.pindices.shape[0]):
            k1, k2, k3 = self.pindices[i,:]
            px = legendre(k1)(ScaledPos[:,0])
            py = legendre(k2)(ScaledPos[:,1])
            pz = legendre(k3)(ScaledPos[:,2])
            Basis[:,i] = px * py * pz

        WBasis = Basis * np.sqrt(Masses)
        QBasis = QR_Decomp(WBasis, 'unormalized')
        QBasis /= np.sqrt(Masses)

        return QBasis

def QR_Decomp(V,dtype):
    """
    QR_Decomp is an experimental function. Should be eventually deleted.
    """

    if dtype is 'normalized':
        V,R = - qr(V, mode='economic')
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

def poly_indexes(kmax):
    """
    Create 2D array of Legendre polynomial indices with index sum <= psum.

    For example, if psum is 1, the this returns
    [[0, 0, 0],
     [1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
    Note, the sum of each row is less than or equal to 1.
    """
    indices = []

    for n in range(kmax + 1):
        for i in range(n+1):
            for j in range(n+1-i):
                indices.append([n-i-j, j, i])

    return np.array(indices,'i')

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
