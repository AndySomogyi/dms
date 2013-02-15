
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

def make_parser():
    """
    Create the argument parser that processes all the DMS command line arguments.
    """
    
    parser = argparse.ArgumentParser(prog="python -m dms2", description="DMS - coarse grained Brownian dynamics")
    subparsers = parser.add_subparsers()

    # create the config argument parser, lots and lots of junk can be handled
    # by the create config function
    ap = subparsers.add_parser("config", help="create a simulation database, from which a simulation can be run")
    
    ap.add_argument("-fid", dest="fid", required=True, type=str,
                    help="name of the simulation file to be created.")
    
    ap.add_argument("-struct", dest="struct", required=True, type=str,
                    help="the starting structure name")
    
    ap.add_argument("-box", dest="box", required=False, nargs=3, type=float, default=None,
                    help="x,y,z values of the system box in Angstroms")
    
    ap.add_argument("-top", dest="top", required=False, default=None,
                    help="name of the topology file. If this is not give, and topology file \
                    will be automatically generated, well at least, will attemtp to be automatically \
                    generated.")
    
    ap.add_argument("-posres", dest="posres", required=False,
                    help="name of a position restraints file, optional.")
    
    ap.add_argument("-temperature", dest="temperature", required=False, type=float, default= 300,
                    help="the temperature at which to run the simulation, defaults to 300K.")
    
    ap.add_argument("-subsystem_factory", dest="subsystem_factory", required=False, 
                    default="dms2.subsystems.RigidSubsystemFactory",
                    help="fully qualified function name which can create a set of subsystems, "
                         "can be set to \'dms2.subsystems.LegendreSubsystemFactory\'")
                  
    ap.add_argument("-subsystem_selects", dest="subsystem_selects", required=False, 
                    nargs="+", default=["not resname SOL"],
                    help="a list of MDAnalysis select statements, one for each subsystem.")
    
    ap.add_argument("-subsystem_args", dest="subsystem_args", required=False, 
                    nargs="+", default=[],
                    help="a list of additional arguments passed to the subsystem factory, "
                         "the first item of the list may be the string \'resid unique\', which "
                         "creates a separate subsystem for each residue." )
    
    ap.add_argument("-integrator", default="dms2.integrators.LangevinIntegrator",
                    help="fully qualified name of the integrator function, "
                         "defaults to \'dms2.integrators.LangevinIntegrator\', "
                         "but the other integrator we provide is \'dms2.integrators.FactorizationIntegrator\'")
    
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
    
    ap.add_argument("-eq_steps", default=10, 
                    help="number of timesteps to run equilibriation")

    ap.add_argument("-mn_args", default=system.DEFAULT_MN_ARGS, 
                    help="additional parameters to be changed in the minimization options")

    ap.add_argument("-eq_args", default=system.DEFAULT_EQ_ARGS)
    ap.add_argument("-md_args", default=system.DEFAULT_MD_ARGS)

    ap.add_argument("-ndx", default=None, 
                    help="name of index file")

    ap.add_argument("-solvate", dest="should_solvate", action="store_true",
                    help="should the system be auto-solvated, if this is set, struct must NOT contain solvent. \
                    defaults to False. This is a boolean flag, to enable, just add \'-solvate\' with no args.")

    ap.set_defaults(__func__=system.create_config)

    
    # Done with config, the MOST complicated command, now make parsers for the more
    # simple commands
    ap = subparsers.add_parser("run", help="start or continue a full simulation. "
                               " This will automatically continue a simulation",)
    ap.add_argument("sys", help="name of simulation file")
    def run(sys) :
        s=system.System(sys, "a")
        integrator = s.integrator()
        integrator.run()
    ap.set_defaults(__func__=run)

    ap = subparsers.add_parser("sol", help="perform only a solvation")
    ap.add_argument("sys", help="name of simulation file")
    def sol(sys) :
        s=system.System(sys, "a")
        s.begin_timestep()
        s.solvate()
        s.end_timestep()  
        print(sys)
    ap.set_defaults(__func__=sol)

    ap = subparsers.add_parser( "mn", help="perform a energy minimization")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-sol", action="store_true", help="auto solvate before minimization")
    def mn(sys, sol) :
        s=system.System(sys, "a")
        s.begin_timestep()
        if sol:
            sol = s.solvate()
            print("sol: {}".format(sol))
            mn = s.minimize(**sol)
            print("mn: {}".format(mn))
            s.end_timestep()
        else:
            s.minimize()
        s.end_timestep()
    ap.set_defaults(__func__=mn)

    ap = subparsers.add_parser("mneq", help="performs energy minimization followed by equilibriation")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-sol", action="store_true", help="auto solvate between steps")
    def mneq(sys,sol) :
        s=system.System(sys, "a")
        s.begin_timestep()
        if sol:
            sol = s.solvate()
            print("sol: {}".format(sol))
            mn = s.minimize(**sol)
            print("mn: {}".format(mn))
            eq = s.equilibriate(**mn)
            print("eq: {}".format(eq))
        else:
            s.minimize()
            s.equilibriate()
        s.end_timestep()    
    ap.set_defaults(__func__=mneq)

    ap = subparsers.add_parser("mneqmd", help="performs energy minimzatin, equilibriation and molecular dynamics")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-sol", action="store_true", help="auto solvate between steps")
    def mneqmd(sys,sol) :
        s=system.System(sys, "a")
        s.begin_timestep()
        if sol:
            sol = s.solvate()
            print("sol: {}".format(sol))
            mn = s.minimize(**sol)
            print("mn: {}".format(mn))
            eq = s.equilibriate(**mn)
            print("eq: {}".format(eq))
            s.md(**eq)
        else:
            s.minimize()
            s.equilibriate()
            s.md()
        s.en
        s.end_timestep()
    ap.set_defaults(__func__=mneqmd)

    ap = subparsers.add_parser("eq", help="performs an equilibriation")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-sol", action="store_true", help="auto solvate between steps")
    def eq(sys) :
        s=system.System(sys, "a")
        s.begin_timestep()
        if sol:
            sol = s.solvate()
            print("sol: {}".format(sol))
            eq = s.equilibriate(**sol)
            print("eq: {}".format(eq))
        else:
            s.equilibriate()
        s.end_timestep()
    ap.set_defaults(__func__=eq)

    ap = subparsers.add_parser("atomistic_step", help="perform an full atomistic step")
    ap.add_argument("sys", help="name of simulation file")
    def atomistic_step(sys) :
        s=system.System(sys, "a")
        integrator = s.integrator()
        s.begin_timestep()
        integrator.atomistic_step()
        s.end_timestep()
    ap.set_defaults(__func__=atomistic_step)

    ap = subparsers.add_parser("step", help="a single complete Langevin step")
    ap.add_argument("sys", help="name of simulation file")
    def step(sys) :
        s=system.System(sys, "a")
        integrator = s.integrator()
        integrator.step()
    ap.set_defaults(__func__=step)

    ap = subparsers.add_parser("md", help="perform ONLY an molecular dynamics step with the starting structure")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-sol", action="store_true", help="auto solvate before md")
    def md(sys,sol) :
        s=system.System(sys, "a")
        s.begin_timestep()
        if sol:
            sol = s.solvate()
            print("sol: {}".format(sol))
            s.md(**sol)
        else:
            s.md()
        s.end_timestep()
    ap.set_defaults(__func__=md)

    ap = subparsers.add_parser("cg_step", help="perform just the coarse grained portion of the time step")
    ap.add_argument("sys", help="name of simulation file")
    def cg_step(sys) :
        s=system.System(sys, "a")
        s._load_ts(s.current_timestep)
        integrator = s.integrator()
        s.begin_timestep()
        integrator.cg_step()
        s.end_timestep()
    ap.set_defaults(__func__=cg_step)
    
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
    


parser = make_parser()
args = parser.parse_args()
print(args)
func = args.__func__
del args.__dict__["__func__"]
func(**args.__dict__)




"""    
if sys.argv[1] == "config":
    parser = config_parser()
    args=parser.parse_args(sys.argv[2:])
    print(args)
    test(**args.__dict__)
    system.create_config(fid=sys.argv[3], **conf)
        os.environ["DMS_DEBUG"] = "TRUE"
        
"""
