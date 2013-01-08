'''
Created on Dec 30, 2012

@author: andy
'''

import system
import util
import sys
import tempfile
import os


if len(sys.argv) == 3:
    if sys.argv[1] == "dpc":
        system.dpctest(sys.argv[2])
    else:
        system.ctest(sys.argv[1], sys.argv[2])
elif len(sys.argv) == 2:
    tempfile.tempdir = os.path.curdir
    s = system.System('test.hdf')
    
    if sys.argv[1] == "mn":
        os.environ["DMS_DEBUG"] = "TRUE"
        s.begin_timestep()
        s.minimize()
        s.end_timestep()
    elif sys.argv[1] == "eq":
        os.environ["DMS_DEBUG"] = "TRUE"
        s.begin_timestep()
        s.equilibriate()
        s.end_timestep()
    elif sys.argv[1] == "md":
        os.environ["DMS_DEBUG"] = "TRUE"
        s.begin_timestep()
        s.md()
        s.end_timestep()
    elif sys.argv[1] == "step":
        os.environ["DMS_DEBUG"] = "TRUE"
        s.step()
    elif sys.argv[1] == "test":
        os.environ["DMS_DEBUG"] = "TRUE"
        s._load_ts(s.current_timestep)
        s.evolve()
        
    else:
    
        s.run()
    

