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
