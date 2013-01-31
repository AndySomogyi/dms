'''
Created on Jan 30, 2013

@author: andy
'''

import integrator
import dynamics

class FactorizationIntegrator(integrator.Integrator):
    def cg_step(self):
        """
        perform a forward euler step (the most idiotic and unstable of all possible integrators)
        to integrate the state variables. 
        
        reads self.cg_momenta to advance the cg state variables. 
        
        X[n+1] = X[n] + dt*dX/dt[n], and dX/dt is cg_moment/cg_mass.
        """
        
        # forward euler 
        cg_translate = self.system.dt * self.system.cg_moment
        
        self.system.translate(cg_translate)
