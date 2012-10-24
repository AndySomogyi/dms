'''
Created on Oct 3, 2012

@author: andy
'''
import subsystems
import MDAnalysis as mda

from numpy import sum, newaxis

class RigidSubsystem(subsystems.SubSystem):
    """
    A set of CG variables that store center of mass postion and 
    orientation. 
    """
    def __init__(self, system, select):
        self.system = system
        self.select = select
        
    def universe_changed(self, universe):
        self.atoms = universe.selectAtoms(self.select)
        
    def frame(self):
        # the center of mass position 
        pos = sum(self.atoms.positions*self.atoms.masses()[:,newaxis],axis=0)/self.atoms.totalMass()
        # the center of mass velocity
        vel = sum(self.atoms.velocities()*self.atoms.masses()[:,newaxis],axis=0)/self.atoms.totalMass()
        # total force
        force = sum(self.atoms.forces,axis=0)
        return (pos,vel,force)
    
    def translate(self, values):
        self.atoms.positions += values
    
    def minimized(self):
        pass
    
    def equilibriated(self):
        pass
    
    
def RigidSubsystemFactory(system, *args):
    return (3, [RigidSubsystem(system, arg) for arg in args])
