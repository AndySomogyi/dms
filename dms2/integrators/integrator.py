'''
Created on Jan 30, 2013

@author: andy
'''

import logging

class Integrator(object):
    def __init__(self, system, *args):
        self.system = system
        
    def run(self):
        """
        Run the simulation for as many timesteps as specified by the configuration.
        
        This will automatically start at the last completed timestep and proceed untill
        all the specified timesteps are completed.
        """
        last = self.system.last_timestep
        start = last.timestep + 1 if last else 0
        del last
        end = self.system.cg_steps
        
        logging.info("running timesteps {} to {}".format(start, end))
        
        for _ in range(start, end):
            self.system.begin_timestep()
            self.system.atomistic_step()
            self.cg_step()
            self.system.end_timestep()
            
        logging.info("completed all {} timesteps".format(end-start))
        
    def cg_step(self):
        raise NotImplementedError
