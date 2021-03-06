'''

Created on Dec 30, 2012

@author: andy


'''
import system
import config
import util
import sys
import tempfile
import os
import os.path
import argparse
import logging
import analysis

def make_parser():
    """
    Create the argument parser that processes all the DMS command line arguments.
    """
    
    parser = argparse.ArgumentParser(prog="python -m dms2", description="DMS - coarse grained Brownian dynamics")
    subparsers = parser.add_subparsers()

    # create the config argument parser, lots and lots of junk can be handled
    # by the create config function
    ap = subparsers.add_parser("config", help="create a simulation database, from which a simulation can be run")
    
    ap.add_argument("-o", dest="fid", required=True, type=str,
                    help="name of the simulation file to be created.")
    
    ap.add_argument("-struct", dest="struct", required=True, type=str,
                    help="the starting structure name")
    
    ap.add_argument("-box", dest="box", required=False, nargs=3, type=float, default=None,
                    help="x,y,z values of the system box in Angstroms. "
                         "If box is not given, the system size is read from the CRYST line "
                         "in the structure pdb.")
    
    ap.add_argument("-top", dest="top", required=False, default=None,
                    help="name of the topology file. If this is not give, and topology file \
                    will be automatically generated, well at least, will attemtp to be automatically \
                    generated.")
    
    ap.add_argument("-posres", dest="posres", required=False,
                    help="name of a position restraints file, optional.")

    ap.add_argument('-I', action='append', dest='include_dirs',
                    default=[], type=str, required=False,
                    help="Include directories to search for topology file includes. There can be many " 
                         "additional include directories, just like gcc, but UNLIKE GCC, there must be a space "
                         "between the -I and the dir, for example -I /home/foo -I /home/foo/bar.")
    
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
                         "creates a separate subsystem for each residue. " 
                         ""
                         "The most comonly used subsystem is dms2.subsystems.LegendreSubsystemFactory, "
                         "The args for this subsystem are [kmax, OPTIONAL(\"resid unique\")], "
                         "kmax is the highest Legendre polynomial index to use, "
                         "and the last arg is the optional string \"resid unique\" to make a unique "
                         "subsystem for each residue." )
    
    ap.add_argument("-integrator", default="dms2.integrators.LangevinIntegrator",
                    help="fully qualified name of the integrator function, "
                         "defaults to \'dms2.integrators.LangevinIntegrator\', "
                         "but the other integrator we provide is \'dms2.integrators.FactorizationIntegrator\'")
    
    ap.add_argument("-integrator_args", default = ["hist_steps 1"], nargs="+", 
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

    ap.add_argument("-mn_args", default=config.DEFAULT_MN_ARGS, 
                    help="additional parameters to be changed in the minimization options", type=eval)

    ap.add_argument("-eq_args", default=config.DEFAULT_EQ_ARGS, type=eval)
    ap.add_argument("-md_args", default=config.DEFAULT_MD_ARGS, type=eval)

    ap.add_argument("-ndx", default=None, 
                    help="name of index file")

    ap.add_argument("-solvate", dest="should_solvate", action="store_true",
                    help="should the system be auto-solvated, if this is set, struct must NOT contain solvent. "
                    "defaults to False. This is a boolean flag, to enable, just add \'-solvate\' with no args.")
    
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    
    ap.add_argument("-mainselection", default="Protein", 
                    help="The name of make_ndx group which is used for the main selection. This should"
                    " be a group that consists of the non solvent molecules. \"Protein\" usually works when "
                    "simulating protein, however another selection command is required when simulationg lipids. "
                    "To find this out, either run dms2 config, and an error will pop up informing you of the available "
                    "groups, or run make_ndx to get a list.")

    ap.set_defaults(__func__=config.create_sim)
    
    
    ap = subparsers.add_parser("top", help="Given a atomic structure file, auto-generate a topology, "
                               "and save it in a directory. "
                               "This is usefull for testing topo auto generation. ")
    ap.add_argument("-o", help="output directory where topology will be generated")
    ap.add_argument("-struct", dest="struct", required=True, type=str,
                    help="the starting structure name")
    ap.add_argument("-posres", dest="posres", required=False,
                    help="name of a position restraints file, optional.")
    ap.set_defaults(__func__=config.create_top)


    #create_sol(o, struct, posres, top, box=None):
    ap = subparsers.add_parser("solvate", help="Given a atomic structure file, attempt auto-solvation, "
                               "and save it in a directory. "
                               "This is usefull for testing auto-solvation. "
                               "All the inputs to solvate are returned from top")
    ap.add_argument("-o", help="output directory where solvated structure will be generated")
    ap.add_argument("-struct", dest="struct", required=True, type=str,
                    help="the starting structure name")
    ap.add_argument("-top", dest="top", required=True,
                    help="name of the topology file.")
    ap.add_argument("-box", dest="box", required=False, nargs=3, type=float, default=None,
                    help="x,y,z values of the system box in Angstroms. "
                         "If box is not given, the system size is read from the CRYST line "
                         "in the structure pdb.")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    ap.set_defaults(__func__=config.create_sol)
    
    # Done with config, the MOST complicated command, now make parsers for the more
    # simple commands
    def analyze(struct, traj, var, kmax, delta_frames, ofname, plot):
        analysis.Plot_Time_Integral(struct, traj, kmax, var, delta_frames, ofname, plot)
        
    ap = subparsers.add_parser("analyze", help="time correlation and integral convergence analysis")
    ap.add_argument("-traj", dest="traj", required=True, type=str, default=None, help="trajectory filename")
    ap.add_argument("-struct", dest="struct", required=True, type=str,
                    help="the starting structure name")
    ap.add_argument("-var", dest="var", required=False, default='velocities()', help="which variables to use: velocities() "
                    "(non-inertial) or forces (inertial), or even positions (for dcd trajectories)")
    ap.add_argument("-kmax", dest="kmax", required=False, default=1, help="maximum polynomial order", type=int)
    ap.add_argument("-delta_frames", dest="delta_frames", required=True, help="minimum number of frames used to calculate"
                    " \delta (length of an mdrun)", type=int)
    ap.add_argument("-ofname", dest="ofname", required=False, default='tmp', help="output filename", type=str)
    ap.add_argument("--plot", action="store_true", required=False, help="plots the integral as a function of \delta")
    
    ap.set_defaults(__func__=analyze)
    
    ap = subparsers.add_parser("run", help="start or continue a full simulation. "
                               " This will automatically continue a simulation",)
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    
    def run(sys, debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["DMS_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        integrator = s.integrator()
        integrator.run()
    ap.set_defaults(__func__=run)

    ap = subparsers.add_parser("runsol", help="Like run, but perfoms only the auto-solvation step. " 
                               "Usefull for debugging run time auto-solvation. "
                               "It unlikely this will be needed as the auto-solvation can be tested "
                               "directly from the struture / top files," )
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def sol(sys, debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["DMS_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        s.begin_timestep()
        s.solvate()
        s.end_timestep()  
        print(sys)
    ap.set_defaults(__func__=sol)
    


    ap = subparsers.add_parser( "mn", help="perform a energy minimization")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-sol", action="store_true", help="auto solvate before minimization")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def mn(sys, sol, debug):
        config.set_tempdir(sys)
        if debug:
            os.environ["DMS_DEBUG"] = "TRUE"
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
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def mneq(sys,sol,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["DMS_DEBUG"] = "TRUE"
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
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def mneqmd(sys,sol,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["DMS_DEBUG"] = "TRUE"
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
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def eq(sys,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["DMS_DEBUG"] = "TRUE"
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
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def atomistic_step(sys,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["DMS_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        integrator = s.integrator()
        s.begin_timestep()
        integrator.atomistic_step()
        s.end_timestep()
    ap.set_defaults(__func__=atomistic_step)

    ap = subparsers.add_parser("step", help="a single complete Langevin step")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def step(sys,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["DMS_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        integrator = s.integrator()
        integrator.step()
    ap.set_defaults(__func__=step)

    ap = subparsers.add_parser("md", help="perform ONLY an molecular dynamics step with the starting structure")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-sol", action="store_true", help="auto solvate before md")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def md(sys,sol,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["DMS_DEBUG"] = "TRUE"
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
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def cg_step(sys,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["DMS_DEBUG"] = "TRUE"
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
    
# make the arg parser, and call whatever func was stored with the arg. 
parser = make_parser()
args = parser.parse_args()
print(args)
func = args.__func__
del args.__dict__["__func__"]
func(**args.__dict__)
