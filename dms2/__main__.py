
'''

Created on Dec 30, 2012

@author: andy


'''

import system
import util
import sys
import tempfile
import os

dpc100 = { 
    'box' : [65.0, 65.0, 65.0],      
    'temperature' : 300.0, 
    'struct': 'dpc100.sol.pdb',
    'top' :  'dpc100.top',
    "subsystem_selects": ["resname DPC"],
    "subsystem_args":["resid unique"],
    "cg_steps":5,
    "dt":100,
    "mn_steps":5000,
    "eq_steps":500,
    "md_steps":2000,
    "multi":10,
    "solvate":False,
    }

dpc60 = { 
    'box' : [75.0, 75.0, 75.0],      
    'temperature' : 285.0, 
    'struct': 'dpc60.sol.pdb',
    'top' :  'dpc60.top',
    "subsystem_selects": ["resname DPC"],
    "subsystem_args":["resid unique"],
    "cg_steps":100,
    "dt":100,
    "mn_steps":5000,
    "eq_steps":500,
    "md_steps":2000,
    "multi":10,
    "solvate":False,
    }


test_structs = {"dpc100":dpc100, "dpc60":dpc60}


if len(sys.argv) == 4 and sys.argv[1] == "config":
    print("making config...")
    conf = None
    try:
        conf = test_structs[sys.argv[2]]
    except KeyError:
        print("config {} not found in existing config".format(sys.argv[2]))
        sys.exit()
    system.create_config(fid=sys.argv[3], **conf)
    

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
    elif sys.argv[1] == "run":
        s.run()
    

