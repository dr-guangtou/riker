#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Get 1-D profiles from the maps."""

import os
import copy

import numpy as np
import sep

import kungpao
from kungpao.galsbp import galSBP

from riker import utils

__all__ = [
    'detect_galaxy',
    'aperture_masses',
    'mass_weighted_prof',
    'TBL', 'ISO',
]

# Right now, galSBP still depends on IRAF, need to point it to the correct
# Binary file
IRAF_DIR = os.path.join(
    os.path.split(kungpao.__file__)[0], 'iraf/{}/'.format(utils.linux_or_mac()))
TBL = os.path.join(IRAF_DIR, 'x_ttools.e')
ISO = os.path.join(IRAF_DIR, 'x_isophote.e')


def detect_galaxy(info, mass_map, kernel=None, threshold=1e8,
                  img_cen_x=None, img_cen_y=None, verbose=False,
                  bkg_ratio=10, bkg_filter=5):
    """Detect the galaxy and get the average shape of the isophote.

    Parameters
    ----------
    info : dict
        Basic information of the galaxy.
    map : ndarray
        Stellar mass map or 2-D map.
    kernel : ndarray, optional
        2-D kernel used for detecting the galaxy. Default: None
    threshold : float, optional
        Mass threshold for detecting the galaxy. Default: 1E8
    img_cen_x : float, optional
        X coordinate of the galaxy center. Default: None
    img_cen_y : float, optional
        Y coordinate of the galaxy center. Default: None
    bkg_ratio : int, optional
        Ratio between the image size and sky box size. Default: 10
    bkg_filter : int, optional
        Filter size for sky background. Default: 5
    verbose : bool, optional
        Blah, Blah, Blah. Default: False

    Returns
    -------
    detect: dict
        A dictionary that contains the object detection result.

    """
    # If no additional information is provided, assume the galaxy centers at
    # the image center
    if img_cen_x is None:
        img_cen_x = info['img_cen_x']
    if img_cen_y is None:
        img_cen_y = info['img_cen_y']

    # There is no "background", we do this to make sure the shape is
    # measured for the central region of the galaxy
    bkg = sep.Background(
        mass_map, bw=(info['img_w'] / bkg_ratio), bh=(info['img_h'] / bkg_ratio),
        fw=bkg_filter, fh=bkg_filter)

    objs = sep.extract(
        mass_map - bkg.back(), threshold, filter_type='conv',
        filter_kernel=kernel, segmentation_map=False)
    n_objs = len(objs)
    if verbose and n_objs > 1:
        print("# Detect {} objects on the map for galaxy {}".format(n_objs, info['id']))

    # Find the objsect at the center
    index = np.argmin(np.sqrt((objs['x'] - img_cen_x) ** 2.0 +
                              (objs['y'] - img_cen_y) ** 2.0))

    # Get the naive ba, theta, xcen, ycen
    ba = objs[index]['b'] / objs[index]['a']
    theta = objs[index]['theta']
    xcen, ycen = objs[index]['x'], objs[index]['y']

    return {'x': xcen, 'y': ycen, 'ba': ba, 'theta': theta,
            'pa': np.rad2deg(theta), 'objs': objs, 'n_objs': n_objs}


def aperture_masses(info, mass_map, detect=None, rad=None, n_rad=15, linear=False,
                    r_min=0.1, r_max=None, subpix=5, **detect_kwargs):
    """Estimate stellar mass profiles based on aperture statistics.

    Parameters
    ----------
    info : dict
        Basic information of the galaxy.
    maps : dict
        All the stellar mass maps.
    detect : dict
        A dictionary that contains the average shape of the galaxy. Default: None.
    rad : ndarray, optional
        Array of boundaries for radius bins in unit of kpc. Default: None.
    n_rad : int, optional
        Number of radial bins. Default: 15.
    linear : bool, optional
        If True, radial bins will be uniformly spaced in linear space.
        If False, radial bins will be uniformly spaced in log10 space.
        Default: False
    r_min : float, optional
        Minimum radius of the radial bins. Default: 0.1
    r_max : float, optional
        Maximum radius of the radial bins. Default: None.
    subpix : int, optional
        Subpixel sampling factor. Default is 5.

    Returns
    -------

    """
    # If basic information is not available, detect the galaxy here.
    if detect is None:
        detect = detect_galaxy(info, mass_map, **detect_kwargs)

    # Get the radial bins in unit of kpc
    if rad is None:
        if r_max is None:
            r_max = info['img_w'] / 2.0
        if linear:
            rad = np.linspace(r_min, r_max * info['pix'], (n_rad + 1))
        else:
            rad = np.logspace(
                np.log10(r_min), np.log10(r_max * info['pix']), (n_rad + 1))

    # Mass within different apertures
    maper = sep.sum_ellipse(
        mass_map, detect['x'], detect['y'], rad / info['pix'], rad / info['pix'] * detect['ba'],
        detect['theta'], 1.0, bkgann=None, subpix=subpix)[0]

    return rad, maper


def mass_weighted_prof(data, mass_map, aper, r_inn, r_out, subpix=7, mask=None):
    """Get the stellar mass weighted properties in annulus.

    Parameters
    ----------
    data : ndarray
        Age or metallicity map, or other quantities that need to be weighted.
    mass_map : ndarray
        Stellar mass map.
    aper : dict
        Basic information of the galaxy
    r_inn : ndarray
        Array of inner radial bins to define the annulus.
    r_out : ndarray
        Array of outer radial bins to define the annulus.
    subpix : int, optional
        Subpixel sampling factor. Default: 5.
    mask : ndarray, optional
        Mask array.

    Returns
    -------
    prof : dict
        A dictionary that contains the mass-weighted and none-weighted profiles.

    """
    # Mass weighted data
    data_w = copy.deepcopy(data * mass_map)
    data_w[np.isnan(data_w)] = 0.0

    # Define a mask...just in case
    if mask is None:
        mask = (mass_map < 1.).astype(np.uint8)

    # Sum(data) term
    sum_data, _, flag = sep.sum_ellipann(
        data, aper['x'], aper['y'], 1.0, aper['ba'],
        aper['theta'], r_inn, r_out, mask=mask, subpix=subpix)

    # Total number of effective pixels
    n_pix_eff, _, flag = sep.sum_ellipann(
        (1.0 - mask), aper['x'], aper['y'], 1.0, aper['ba'],
        aper['theta'], r_inn, r_out, mask=mask, subpix=subpix)

    # Sum(mass_map * data) term
    sum_data_w, _, _ = sep.sum_ellipann(
        data_w, aper['x'], aper['y'], 1.0, aper['ba'],
        aper['theta'], r_inn, r_out, mask=mask, subpix=subpix)

    # Sum(mass_map) term
    sum_mass_map, _, _ = sep.sum_ellipann(
        mass_map, aper['x'], aper['y'], 1.0, aper['ba'],
        aper['theta'], r_inn, r_out, mask=mask, subpix=subpix)

    return {'prof_w': sum_data_w / sum_mass_map,
            'prof': sum_data / n_pix_eff, 'flag': flag}


def get_1d_prof(fits_name, aper, isophote=ISO, xttools=TBL, pix=1.0,
                ini_sma=15.0, max_sma=175., step=0.2, mode='mean'):
    """Get Step 2 and Step 3 1-D profile from the stellar mass map.

    Parameters
    ----------

    """
    # Just use the center from the aperture result
    xcen, ycen = aper['x'], aper['y']
    # Initial values of b/a and position angle are from the aperture result too.
    ba, pa = aper['ba'], aper['pa'] + 90.0
    if pa >= 90.0:
        pa = pa - 180.
    elif pa <= -90.0:
        pa = pa + 180.

    try:
        # Step 2 to get ellipticity and position angle profiles
        ell_2, bin_2 = galSBP.galSBP(
            fits_name, galX=xcen, galY=ycen, maxSma=max_sma, iniSma=ini_sma,
            verbose=False, savePng=False, saveOut=True, expTime=1.0,
            pix=pix, zpPhoto=0.0, galQ=ba, galPA=pa, stage=2,
            minSma=0.0, ellipStep=step, isophote=isophote, xttools=xttools,
            uppClip=2.5, lowClip=3.0, maxTry=5, nClip=2, intMode=mode,
            updateIntens=False, harmonics=True)
    except Exception:
        print("# Something went wrong during stage 2 for {}".format(fits_name))
        ell_2, bin_2 = None, None

    try:
        # Step 3 to get mass density profiles
        ell_3, bin_3 = galSBP.galSBP(
            fits_name, galX=xcen, galY=ycen, maxSma=max_sma, iniSma=ini_sma,
            verbose=False, savePng=False, saveOut=True, expTime=1.0,
            pix=pix, zpPhoto=0.0, galQ=ba, galPA=pa, stage=3,
            minSma=0.0, ellipStep=step, isophote=isophote, xttools=xttools,
            uppClip=2.5, lowClip=3.0, maxTry=3, nClip=2, intMode=mode,
            updateIntens=False, harmonics=True)
    except Exception:
        print("# Something went wrong during stage 3 for {}".format(fits_name))
        ell_3, bin_3 = None, None

    return ell_2, ell_3, bin_2, bin_3
