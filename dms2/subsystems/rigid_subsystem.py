'''
Created on Oct 3, 2012

@author: andy
'''
import subsystems

from numpy import sum, newaxis, pi, sin, cos, arctan2

class RigidSubsystem(subsystems.SubSystem):
    """
    A set of CG variables that store center of mass postion and 
    orientation. 
    """
    def __init__(self, system, select):
        """
        Create a rigid subsystem.
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
        masses = self.atoms.masses()[:,newaxis]
        # scaled positions, positions is natom * 3, box should be 3 vector.
        # arctan2 has a range of (-pi, pi], so have to shift positions to 
        # zero centered instead of box / 2 centered.
        spos = self.atoms.positions / self.system.box * 2.0 * pi - pi
        
        # get the x and y components, mass scale them, and add in cartesian space
        # shift back to box / 2 centered
        cm = (arctan2(sum(masses * sin(spos), axis=0), sum(masses * cos(spos), axis=0)) + pi) * self.system.box / 2.0 / pi
        
        # the center of mass velocity
        vel = sum(self.atoms.velocities() * masses,axis=0)/self.atoms.totalMass()
        # total force
        force = sum(self.atoms.forces,axis=0)
        return (cm,vel,force)
    
    def translate(self, values):
        self.atoms.positions += values
    
    def minimized(self):
        pass
    
    def equilibriated(self):
        pass
    
    
    
def RigidSubsystemFactory(system, selects, *args):
    return (3, [RigidSubsystem(system, select, *args) for select in selects])
