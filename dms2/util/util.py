"""
Created on Dec 19, 2012

@author: andy

dms2 io module

functions for reading and writng files to / from disk and hdf blobs.
"""

import h5py #@UnresolvedImport
import gromacs.utilities as utilities
import os.path  
import shutil
import numpy
import MDAnalysis.core

def data_tofile(data, fid, sep="", fmt="%s", dirname="."):
    """
    @param fid : file or str    
        An open file object, or a string containing a filename.
    @param sep : str
        Separator between array items for text output. If "" (empty), a binary file is written, 
        equivalent to file.write(a.tostring()).
    @param fmt : str
        Format string for text file output. Each entry in the array is formatted to text by 
        first converting it to the closest Python type, and then using "format" % item.
    @param dirname
    @return: absolute path of the created file
        """ 
    if data is not None:
        
            
        if isinstance(data, str) and os.path.isfile(data):
            src = os.path.abspath(data)
            with utilities.in_dir(dirname):
                shutil.copyfile(src, fid)
                return os.path.abspath(fid)
        else:
            with utilities.in_dir(dirname):
                if type(data) is h5py.Dataset:
                    data = data[()]
                if type(data) is numpy.ndarray:  
                    data.tofile(fid,sep,fmt)
                elif isinstance(data, MDAnalysis.core.AtomGroup.AtomGroup) or isinstance(data, MDAnalysis.core.AtomGroup.Universe):
                    print("pwd", os.path.curdir)
                    print("fid", fid)
                    w = MDAnalysis.Writer(fid,numatoms=len(data.atoms))
                    w.write(data)
                    del w
                else:
                    raise TypeError("expected either Dataset, ndarray or file path as src")
                return os.path.abspath(fid)
            
            
def hdf_linksrc(hdf, newname, src):
    """
    if src is a soft link, follow it's target until we get to a non-linked object, and
    create the new link to point to this head object.
    """
    
    try:
        while True:
            print(1,src)
            src = hdf.id.links.get_val(src)
            print(2,src)
    except (TypeError, KeyError):
        pass
    
    print("links.create_soft({}, {})".format(newname, src))
    hdf.id.links.create_soft(newname, src)
    
def get_class( klass ):
    """
    given a fully qualified class name, i.e. "datetime.datetime", 
    this loads the module and returns the class type. 
    
    the ctor on the class type can then be called to create an instance of the class.
    """
    parts = klass.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m

def is_env_set(env):
    """
    returns True if the enviornment variable is set to 'yes', 'true' or non-zero integer,
    False otherwise
    """
    try:
        var = os.environ[env].strip().upper()
        try:
            return int(var) != 0
        except:
            pass
        return var == "TRUE" or var == "YES"
    except:
        pass
    return False
        
        
        
