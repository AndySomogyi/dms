
'''

Created on Dec 30, 2012

@author: andy


'''

import system
import util
import sys
import tempfile
import os
import argparse

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
    'struct': 'dpc60.pdb',
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
    "cg_steps":30,
    "dt":10,
    "mn_steps":5000,
    "eq_steps":500,
    "md_steps":2000,
    "multi":2,
    "should_solvate":True,
    }


Au2 = { 
    'box' : [50.0, 50.0, 50.0],      
    'temperature' : 300.0, 
    'struct': '/home/andy/tmp/1OMB/1OMB.pdb',
    "subsystem_selects": "not resname SOL",
    "cg_steps":5,
    "dt":10.0,
    "top_args": {},
    "mn_steps":5,
    "eq_steps":50,
    "md_steps":50,
    "multi":4,
    "solvate":False,
    } 


test_structs = {"dpc100":dpc100, "dpc60":dpc60, "dpc5":dpc5}

def config_parser():
    """
    def create_config(fid,
                  struct,
                  box,
                 
    """
    

    
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-fid", dest="fid", required=True, type=str,
                    help="name of the simulation file to be created.")
    
    ap.add_argument("-struct", dest="struct", required=True, type=str,
                    help="the starting structure name")
    
    ap.add_argument("-box", dest="box", required=True, nargs=3, type=float,
                    help="x,y,z values of the system box in Angstroms")
    
    ap.add_argument("-top", dest="top", required=False, default=None,
                    help="name of the topology file. If this is not give, and topology file \
                    will be automatically generated, well at least, will attemtp to be automatically \
                    generated.")
    
    ap.add_argument("-posres", dest="posres", required=False,
                    help="name of a position restraints file.")
    
    ap.add_argument("-temperature", dest="temperature", required=False, type=int, default= 300,
                    help="the temperature at which to run the simulation.")
    
    ap.add_argument("-subsystem_factory", dest="subsystem_factory", required=False, 
                    default="dms2.subsystems.RigidSubsystemFactory",
                    help="fully qualified function name which can create a set of subsystems")
                  
    ap.add_argument("-subsystem_selects", dest="subsystem_selects", required=False, 
                    nargs="+", default=["not resname SOL"],
                    help="a list of MDAnalysis select statements, one for each subsystem.")
    
    ap.add_argument("-subsystem_args", dest="subsystem_args", required=False, 
                    nargs="+", default=[],
                    help="additional arguments passed to the subsystem factory")
    
    
    ap.add_argument("-integrator", default="dms2.integrators.LangevinIntegrator",
                    help="fully qualified name of the integrator function")
    
    ap.add_argument("-integrator_args", default = [],
                    help="additional arguments passed to the integrator function")
    
    ap.add_argument("-cg_steps", default = 10,
                    help="number of coarse grained time steps")
    
    ap.add_argument("-dt", default=0.1, 
                    help="size of coarse grained time step in picoseconds")
    
    ap.add_argument("-mn_steps", default = 500,
                    help="number of MD steps to take performing a minimization")
    
    ap.add_argument("-md_steps", default=100,
                    help="number of MD steps to take whilst performing the MD runs")
    
    ap.add_argument("-multi", default=1,
                    help="number of parallel MD runs")
    
    ap.add_argument("-eq_steps", default=10)
    ap.add_argument("-mn_args", default=system.DEFAULT_MN_ARGS),
    ap.add_argument("-eq_args", default=system.DEFAULT_EQ_ARGS),
    ap.add_argument("-md_args", default=system.DEFAULT_MD_ARGS),
    ap.add_argument("-ndx", default=None)


    
    ap.add_argument("-solvate", dest="solvate", action="store_true",
                    help="should the system be auto-solvated, if this is set, struct must NOT contain solvent. \
                    defaults to False. This is a boolean flag, to enable, just add \'-solvate\' with no args.")
    
    return ap

run_cmds = ["mn",  "eq",  "md",  "atomistic_step",  "step",  "cg_step",  "run",  "sol",  "mn", "mneq", "mneqmd"]
        
def cmd_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=run_cmds)

    return parser


def test(fid,
         struct,
         box,
         top = None,
         posres = None,
         temperature = 300,
         subsystem_factory = "dms2.subsystems.RigidSubsystemFactory",
         subsystem_selects = ["not resname SOL"],
         subsystem_args = [],
         integrator = "dms2.integrators.LangevinIntegrator",
         integrator_args = [],
         cg_steps = 10,
         dt  = 0.1,
         mn_steps = 500,
         md_steps = 100,
         multi = 1,
         eq_steps = 10,
         mn_args = None,
         eq_args = None,
         md_args = None,
         should_solvate = False,
         ndx=None,
         **kwargs):
    print("hello")
    

commands = {"config":    "full documentation in in config parser above",
            "run":       "start or continue a full simulation. This will automatically continue\n" + 
                         "\ta simulation",
            "sol":       "perform a solvation",
            "mn":        "perform a energy minimization", 
            "mneq":      "performs energy minimization followed by equilibriation",
            "mneqmd":    "performs energy minimzatin, equilibriation and molecular dynamics",
            "eq":        "performs an equilibriation",
            "md":        "perform an molecular dynamics step",
            "atomistic_step":  "perform an full atomistic step",
            "step":      "perform an full single coarse grained time step",
            "cg_step":   "forform just the coarse grained portion of the time step"}

def print_command_usage():
    if sys.argv[1] in commands.keys():
        print("usage: python -m dms2 {}\n  {}".format(sys.argv[1], commands[sys.argv[1]]))
        print("required arguments: \n  -sys\tthe name of a valid simulation file, created by a dms2 config command") 
    else:
        print("invalid command, valid dms2 commands are {}".format(commands.keys()))
        
if len(sys.argv) < 2:
    print("error, usage python -m dms2 COMMAND [args]")
    print("valid DMS command are {}".format(commands.keys()))
    print("for help on any particular DMS command, use python -m dms2 COMMAND -h")
    sys.exit(-1)
    
if sys.argv[1] == "config":
    parser = config_parser()
    args=parser.parse_args(sys.argv[2:])
    print(args)
    test(**args.__dict__)
    system.create_config(fid=sys.argv[3], **conf)
else:
    # do one of the "simulation" commands, i.e. run a sim, or a single step of a sim.
    # because of help for each 'command', simpler to manually parse the args
    if "-h" in sys.argv or "--help" in sys.argv:
        print_command_usage()
        sys.exit()
    if len(sys.argv) <= 3 or sys.argv[2] != "-sys":
        print("error invalid arguments")
        print_command_usage()
        sys.exit(-1)
        
    # make a new system object in APPEND mode.
    # TODO: some error checking to make sure the file is not already open.    
    s = system.System(sys.argv[3], "a")
    
    # not help, do one of the commands
        
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
        integrator = s.integrator()
        s.begin_timestep()
        integrator.atomistic_step()
        s.end_timestep()
    elif sys.argv[1] == "step":
        os.environ["DMS_DEBUG"] = "TRUE"
        integrator = s.integrator()
        integrator.step()
    elif sys.argv[1] == "cg_step":
        os.environ["DMS_DEBUG"] = "TRUE"
        s._load_ts(s.current_timestep)
        integrator = s.integrator()
        s.begin_timestep()
        integrator.cg_step()
        s.end_timestep()
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
"""

