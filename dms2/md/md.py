'''
Created on Aug 8, 2012

@author: andy
'''
import os.path
import os
import time
import shutil
import logging
import re
import subprocess
import MDAnalysis
import gromacs
import gromacs.run

class MDrunnerLocal(gromacs.run.MDrunner):
    """Manage running :program:`mdrun` as mpich2_ multiprocessor job with the SMPD mechanism.

    .. _mpich2: http://www.mcs.anl.gov/research/projects/mpich2/
    """
    mdrun = "mdrun"
    mpiexec = "mpiexec"

    def mpicommand(self, *args, **kwargs):
        """Return a list of the mpi command portion of the commandline.

        Only allows primitive mpi at the moment:
           *mpiexec* -n *ncores* *mdrun* *mdrun-args*

        (This is a primitive example for OpenMP. Override it for more
        complicated cases.)
        """

        return ["mpiexec", "-n", "4"]

class ResourceManager(object):
    def __init__(self, files):
        import types
        self._files = files if isinstance(files, types.ListType) else [files]
        
    def  __enter__(self):
        return self
    
    def __exit__(self, tp, value, traceback):
        for f in self._files:
            try:
                #os.remove(f)
                print("would have removed {}".format(f))
            except OSError:
                logging.warn('could not remove {}'.format(f))
    
    @property
    def files(self):
        return self._files
    
def test_md(config, atoms):
#    return ResourceManager("dpc_125_nvt.trr")

    files = ["nvt60.trr", "nvt61.trr", "nvt62.trr", "nvt63.trr", "nvt64.trr", 
             "nvt65.trr", "nvt66.trr", "nvt67.trr", "nvt68.trr", "nvt69.trr"]
    
    return ResourceManager(files)
    
    
def md(config, atoms):
    """
    performs a molecular dynamics run
    @return: a tuple of (coordinate_file_names, velocity_file_names, force_file_names)
    """
    w = MDAnalysis.Writer("md_in.pdb")
    w.write(atoms)
    w.close()
    
    
    g=gromacs.grompp(f=config["md_mdp"], c="md_in.pdb", p=config["top"], o="md.tpr")

    if os.environ.has_key("SLURM_JOBID"):
        logging.info("running in SLURM")
        m=gromacs.run.MDrunnerOpenMPI(deffnm="md")
    else:
        logging.info("not using SLURM")
        m=MDrunnerLocal(deffnm="md")
    
    m.run()
    
    os.remove("md_in.pdb")
    os.remove("md.tpr")
    
    return ResourceManager("md.trr")

    
def minimise(universe):
    
    logging.debug('starting minimise(...)')
    
    w = MDAnalysis.Writer("md_in.pdb")
    w.write(atoms)
    w.close()
    
def thermalise(universe):
    pass


    

def topology(struct, protein='protein', dirname='.', **pdb2gmx_args):
    """Build Gromacs topology files from a Universe object.

    :Keywords:
       *struct*
           input structure (**required**)
       *protein*
           name of the output files
       *top*
           name of the topology file
       *dirname*
           directory in which the new topology will be stored
       *pdb2gmxargs*
           arguments for ``pdb2gmx`` such as ``ff``, ``water``, ...

    .. note::
       At the moment this function simply runs ``pdb2gmx`` and uses
       the resulting topology file directly. If you want to create
       more complicated topologies and maybe also use additional itp
       files or make a protein itp file then you will have to do this
       manually.
    """    
    pass
    
def solvate(struct='top/protein.pdb', top='top/system.top',
            distance=0.9, boxtype='dodecahedron',
            concentration=0, cation='NA+', anion='CL-',
            water='spc', solvent_name='SOL', with_membrane=False,
            ndx = 'main.ndx', mainselection = '"Protein"',
            dirname='solvate',
            **kwargs):
    """Put protein into box, add water, add counter-ions.

    Currently this really only supports solutes in water. If you need
    to embedd a protein in a membrane then you will require more
    sophisticated approaches.

    However, you *can* supply a protein already inserted in a
    bilayer. In this case you will probably want to set *distance* =
    ``None`` and also enable *with_membrane* = ``True`` (using extra
    big vdw radii for typical lipids).

    .. Note:: The defaults are suitable for solvating a globular
       protein in a fairly tight (increase *distance*!) dodecahedral
       box.

    :Arguments:
      *struct* : filename
          pdb or gro input structure
      *top* : filename
          Gromacs topology
      *distance* : float
          When solvating with water, make the box big enough so that
          at least *distance* nm water are between the solute *struct*
          and the box boundary.
          Set *boxtype*  to ``None`` in order to use a box size in the input
          file (gro or pdb).
      *boxtype* : string
          Any of the box types supported by :class:`~gromacs.tools.Editconf`
          (triclinic, cubic, dodecahedron, octahedron). Set the box dimensions
          either with *distance* or the *box* and *angle* keywords.

          If set to ``None`` it will ignore *distance* and use the box
          inside the *struct* file.
      *box*
          List of three box lengths [A,B,C] that are used by :class:`~gromacs.tools.Editconf`
          in combination with *boxtype* (``bt`` in :program:`editconf`) and *angles*.
          Setting *box* overrides *distance*.
      *angles*
          List of three angles (only necessary for triclinic boxes).
      *concentration* : float
          Concentration of the free ions in mol/l. Note that counter
          ions are added in excess of this concentration.
      *cation* and *anion* : string
          Molecule names of the ions. This depends on the chosen force field.
      *water* : string
          Name of the water model; one of "spc", "spce", "tip3p",
          "tip4p". This should be appropriate for the chosen force
          field. If an alternative solvent is required, simply supply the path to a box
          with solvent molecules (used by :func:`~gromacs.genbox`'s  *cs* argument)
          and also supply the molecule name via *solvent_name*.
      *solvent_name*
          Name of the molecules that make up the solvent (as set in the itp/top).
          Typically needs to be changed when using non-standard/non-water solvents.
          ["SOL"]
      *with_membrane* : bool
           ``True``: use special ``vdwradii.dat`` with 0.1 nm-increased radii on
           lipids. Default is ``False``.
      *ndx* : filename
          How to name the index file that is produced by this function.
      *mainselection* : string
          A string that is fed to :class:`~gromacs.tools.Make_ndx` and
          which should select the solute.
      *dirname* : directory name
          Name of the directory in which all files for the solvation stage are stored.
      *includes*
          List of additional directories to add to the mdp include path
      *kwargs*
          Additional arguments are passed on to
          :class:`~gromacs.tools.Editconf` or are interpreted as parameters to be
          changed in the mdp file.

    """
    pass
    
   

def rm(fname):
    try:
        os.remove(fname)
    except OSError:
        logging.warn('could not remove {}'.format(fname))
        
        


    

def run_namd(conf_file):
    logging.debug('starting run_namd(conf_file={})'.format(conf_file))
    max_retry = 5
    which_namd=which('namd2')
    
    #determine MPI command
    cmd = ['mpirun']
    
    #MPI if running on load leveler looks like:
    #mpirun -np $LOADL_TOTAL_TASKS -machinefile $LOADL_HOSTFILE $NAMD2
    LOADL_TOTAL_TASKS=os.environ.get('LOADL_TOTAL_TASKS', None)
    LOADL_HOSTFILE=os.environ.get('LOADL_HOSTFILE',None)
    
    if(LOADL_TOTAL_TASKS and LOADL_HOSTFILE):
        logging.debug('running in load leveler')
        cmd.append('-np')
        cmd.append(LOADL_TOTAL_TASKS)
        cmd.append('-machinefile')
        cmd.append(LOADL_HOSTFILE)
        
    cmd.append(which_namd)
    cmd.append(conf_file)
    
    out_open = lambda mode: open('namd_out.tmp.log', mode)
    err_open = lambda mode: open('namd_err.tmp.log', mode)
    
    for retry in range(0,max_retry):
        logging.info('calling namd retry {} with cmd: {}'.format(retry, cmd))
    
        proc = subprocess.Popen(args=cmd, stdout=out_open('w'), stderr=err_open('w'))
        code=proc.wait()

        # concatenate tmp logs with main logs
        shutil.copyfileobj(out_open('r'), open('namd_out.log', 'w+'))
        shutil.copyfileobj(err_open('r'), open('namd_err.log', 'w+'))
    
        # search output for bits. Note, this is really ugly, and needs to be cleaned up
        fatal_re = re.compile("FATAL ERROR")
        fatal_error = len([True for line in out_open('r').readlines() if fatal_re.search(line)]) > 0
        if fatal_error:
            logging.error("'FATAL ERROR' found in namd output file, see log files namd_out.log and namd_err.log")
            continue
        
        end_re = re.compile("End of program")
        end_of_program = len([True for line in out_open('r').readlines() if end_re.search(line)]) > 0
        if not end_of_program:
            logging.error("'End of program' NOT found in namd output, see log files namd_out.log and namd_err.log")
            continue
        
        # reached here, no error, can exit
        logging.debug('finished run_namd(...) = {}'.format(code))
        return code
    
    #after retry loop, max retries exceeded
    logging.critical("Maximum namd retries exceeded, see namd output logs, terminating...")
    epic_fail()
    
def epic_fail():
    """
    determine if we are running in load leveler or slurm, and
    kill the job
    """
    
    # find out what job scheduler we are runing under
    if os.environ.get('LOADL_STEP_ID', None):
        cmd = "llcancel {}".format(os.environ.get('LOADL_STEP_ID'))
        logging.critical("running in load leveler, about to terminate with {}".format(cmd))
        os.system(cmd)
    elif os.environ.get('SLURM_JOB_ID', None):
        cmd = "scancel {}".format(os.environ.get('SLURM_JOB_ID'))
        logging.critical("running in slurm, about to terminate with {}".format(cmd))
        os.system(cmd)
    elif os.environ.get('PBS_JOBID', None):
        cmd = "qdel {}".format(os.environ.get('PBS_JOBID'))
        logging.critical("running in Torque / PBS, about to terminate with {}".format(cmd))
        os.system(cmd)
    else:
        logging.critical("running in unknown environment, now what the fuck do we do?")
    
    # now just wait for job manager to kill us
    while True:
        time.sleep(5)
        logging.critical("waiting for job to terminate...")
        
def which(program):
    """
    determine the absolute path for a program,
    python version of the unix 'which' command
    """
    for path in os.environ.get('PATH', '').split(':'):
        if os.path.exists(os.path.join(path, program)) and \
           not os.path.isdir(os.path.join(path, program)):
            return os.path.join(path, program)
    return None
