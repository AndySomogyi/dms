'''
Created on Oct 3, 2012

@author: andy
'''

class SubSystem(object):
    """
    Defines the interface that a subsystem object must implement. 
    As this is Python, it is not a requirement for subsystems to actually 
    derive from this clas, provided that they define all the methods defined
    here. This class just provides empty implementations of all the 
    required interface methods.
    """
    
    def universe_changed(self, universe):
        """ 
        the universe changed, this occurs in the event of solvation (different number of atoms), 
        or if there is no solvation, this occurs once on startup.
        """
        pass
    
    def frame(self):
        """
        notifices the subsystem that a new frame is ready. 
        @return a tuple of (position,velocity,force) CG variables for the current frame.
        """
        pass
    
    def translate(self, values):
        pass
    
    def minimized(self):
        pass
    
    def equilibriated(self):
        pass
    
