
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
    "dt":50,
    "mn_steps":5000,
    "eq_steps":500,
    "md_steps":2000,
    "multi":10,
    "should_solvate":True,
    }

dpc5 = { 
    'box' : [25.0, 25.0, 25.0],      
    'temperature' : 285.0, 
    'struct': 'dpc5.pdb',
    'top' :  'dpc5.top',
    "subsystem_selects": ["resname DPC"],
    "subsystem_args":["resid unique"],
    "cg_steps":100,
    "dt":10,
    "mn_steps":5000,
    "eq_steps":500,
    "md_steps":2000,
    "multi":2,
    "should_solvate":True,
    }

test_structs = {"dpc100":dpc100, "dpc60":dpc60, "dpc5":dpc5}


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
    s = system.System('test.hdf', "a")
    
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
    elif sys.argv[1] == "atomistic_step":
        os.environ["DMS_DEBUG"] = "TRUE"
        s.atomistic_step()
    elif sys.argv[1] == "step":
        os.environ["DMS_DEBUG"] = "TRUE"
        integrator = s.integrator()
        s.atomistic_step()
        integrator.cg_step()
    elif sys.argv[1] == "cg_step":
        os.environ["DMS_DEBUG"] = "TRUE"
        s._load_ts(s.current_timestep)
        integrator = s.integrator()
        integrator.cg_step()
    elif sys.argv[1] == "run":
        integrator = s.integrator()
        integrator.run()
        
elif len(sys.argv) == 3 and sys.argv[1] == "sol":
    tempfile.tempdir = os.path.curdir
    s = system.System('test.hdf', "a")
    
    if sys.argv[2] == "sol":
        os.environ["DMS_DEBUG"] = "TRUE"
        s.begin_timestep()
        s.solvate()
        s.end_timestep()
        
    if sys.argv[2] == "mn":
        os.environ["DMS_DEBUG"] = "TRUE"
        s.begin_timestep()
        sol = s.solvate()
        print("sol: {}".format(sol))
        mn = s.minimize(**sol)
        print("mn: {}".format(mn))
        s.end_timestep()
        
    if sys.argv[2] == "eq":
        os.environ["DMS_DEBUG"] = "TRUE"
        s.begin_timestep()
        sol = s.solvate()
        print("sol: {}".format(sol))
        eq = s.equilibriate(**sol)
        print("eq: {}".format(eq))
        s.end_timestep()
        
    if sys.argv[2] == "md":
        os.environ["DMS_DEBUG"] = "TRUE"
        s.begin_timestep()
        sol = s.solvate()
        print("sol: {}".format(sol))
        s.md(**sol)
        s.end_timestep()
        
    if sys.argv[2] == "mneq":
        os.environ["DMS_DEBUG"] = "TRUE"
        s.begin_timestep()
        sol = s.solvate()
        print("sol: {}".format(sol))
        mn = s.minimize(**sol)
        print("mn: {}".format(mn))
        eq = s.equilibriate(**mn)
        print("eq: {}".format(eq))
        s.end_timestep()
        
    if sys.argv[2] == "mneqmd":
        os.environ["DMS_DEBUG"] = "TRUE"
        s.begin_timestep()
        sol = s.solvate()
        print("sol: {}".format(sol))
        mn = s.minimize(**sol)
        print("mn: {}".format(mn))
        eq = s.equilibriate(**mn)
        print("eq: {}".format(eq))
        s.md(**eq)
        s.end_timestep()
    

