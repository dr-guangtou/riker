#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Useful tools."""

import os

from astropy.io import fits

__all__ = [
    'save_to_fits',
]


def save_to_fits(img, fits_file, wcs=None, header=None, overwrite=True):
    """Save an image to FITS file."""
    if wcs is not None:
        wcs_header = wcs.to_header()
        img_hdu = fits.PrimaryHDU(img, header=wcs_header)
    else:
        img_hdu = fits.PrimaryHDU(img)
    if header is not None:
        if 'SIMPLE' in header and 'BITPIX' in header:
            img_hdu.header = header
        else:
            img_hdu.header.extend(header)

    if os.path.islink(fits_file):
        os.unlink(fits_file)

    img_hdu.writeto(fits_file, overwrite=overwrite)

    return
