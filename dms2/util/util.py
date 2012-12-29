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
        if type(data) is h5py.Dataset:
            data = data[()]
            
        elif os.path.isfile(data):
            src = os.path.abspath(data)
            with utilities.in_dir(dirname):
                shutil.copyfile(src, fid)
                return os.path.abspath(fid)
        else:
            with utilities.in_dir(dirname):
                if type(data) is numpy.ndarray:  
                    data.tofile(fid,sep,fmt)
                if isinstance(data, MDAnalysis.core.AtomGroup.AtomGroup) or isinstance(data, MDAnalysis.core.AtomGroup.Universe):
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
            src = hdf.id.links.get_val(src)
    except TypeError:
        pass
    
    print("links.create_soft({}, {})".format(newname, src))
    hdf.id.links.create_soft(newname, src)
        
        
        
