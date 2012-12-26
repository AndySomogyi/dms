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
    if type(data) is h5py.Dataset:
        data = data[()]
    
    if type(data) is numpy.ndarray:  
        with utilities.in_dir(dirname):
            data.tofile(fid,sep,fmt)
            return os.path.abspath(fid)
    elif os.path.isfile(data):
        src = os.path.abspath(data)
        with utilities.in_dir(dirname):
            shutil.copyfile(src, fid)
            return os.path.abspath(fid)
    else:
        raise TypeError("expected either Dataset, ndarray or file path as src")
            
        
    
    



