'''
Created on Oct 3, 2012

@author: andy
'''

import MDAnalysis as mda
from numpy import sum, newaxis

class RigidSubsystem(object):
    def frame(self):
        # the center of mass position 
        pos = sum(self.atoms.positions*self.atoms.masses()[:,newaxis],axis=0)/self.atoms.totalMass()
        # the center of mass velocity
        vel = sum(self.atoms.velocities()*self.atoms.masses()[:,newaxis],axis=0)/self.atoms.totalMass()
        # total force
        force = sum(self.atoms.forces,axis=0)
        return (pos,vel,force)
    
    def translate(self):
        pass
    
    def minimized(self):
        pass
    
    def thermalized(self):
        pass
    
    
def RigidSubsystemFactory(system):
    return None
