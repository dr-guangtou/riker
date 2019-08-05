#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Useful tools."""

import os
import glob
import pickle

from astropy.io import fits

__all__ = [
    'read_from_pickle',
    'save_to_pickle',
    'save_to_fits',
    'linux_or_mac',
    'clean_after_ellipse',
]


def read_from_pickle(name):
    """Read the data from Pickle file."""
    return pickle.load(open(name, "rb"))


def save_to_pickle(obj, name):
    """Save an object to a cPickle/Pickle format binary file."""
    output = open(name, 'wb')
    pickle.dump(obj, output, protocol=2)
    output.close()

    return


def save_to_fits(data, fits_file, wcs=None, header=None, overwrite=True):
    """Save a NDarray to FITS file.

    Parameters
    ----------
    data : ndarray
        Data to be saved in FITS file.
    fits_file : str
        Name of the FITS file.
    wcs : astropy.wcs object, optional
        World coordinate system information. Default: None
    header : str, optional
        Header information. Default: None
    overwrite : bool, optional
        Overwrite existing file or not. Default: True

    """
    if wcs is not None:
        wcs_header = wcs.to_header()
        data_hdu = fits.PrimaryHDU(data, header=wcs_header)
    else:
        data_hdu = fits.PrimaryHDU(data)
    if header is not None:
        if 'SIMPLE' in header and 'BITPIX' in header:
            data_hdu.header = header
        else:
            data_hdu.header.extend(header)

    if os.path.islink(fits_file):
        os.unlink(fits_file)

    data_hdu.writeto(fits_file, overwrite=overwrite)

    return


def linux_or_mac():
    """Check the current platform.

    Parameters
    ----------

    Return
    ------
    platform : str
        "linux" or "macosx".

    """
    from sys import platform

    if platform == "linux" or platform == "linux2":
        return "linux"
    elif platform == "darwin":
        return "macosx"
    else:
        raise TypeError("# Sorry, only support Linux and MacOSX for now!")


def clean_after_ellipse(folder, prefix, remove_bin=False):
    """Clean all the unecessary files after ellipse run.

    Parameters
    ----------
    folder : str
        Directory that keeps all the output files.
    prefix : str
        Prefix of the file.
    remove_bin : bool, optional
        Remove the output binary table or not. Default: False

    """
    _ = [os.remove(par) for par in glob.glob("{}/{}*.par".format(folder, prefix))]
    _ = [os.remove(pkl) for pkl in glob.glob("{}/{}*.pkl".format(folder, prefix))]
    _ = [os.remove(img) for img in glob.glob("{}/{}*.fits".format(folder, prefix))]
    _ = [os.remove(tab) for tab in glob.glob("{}/{}*.tab".format(folder, prefix))]
    if remove_bin:
        _ = [os.remove(bin) for bin in glob.glob("{}/{}*.bin".format(folder, prefix))]
