#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Get 1-D profiles from the maps."""

import os
import copy

import numpy as np
import sep

from astropy.table import Table, Column

import kungpao
from kungpao.sbp import galSBP

from riker import utils

__all__ = [
    'detect_galaxy',
    'aperture_masses',
    'mass_weighted_prof',
    'ell_prof',
    'fourier_profile',
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


def mass_weighted_prof(data, mass_map, aper, r_inn, r_out, subpix=7,
                       mask=None, return_mass=False):
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
    return_mass : bool, optional
        Return the stellar mass in each radial bins. Default: False

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

    if return_mass:
        return {'prof_w': sum_data_w / sum_mass_map,
                'prof': sum_data / n_pix_eff,
                'mass': sum_mass_map,
                'flag': flag}

    return {'prof_w': sum_data_w / sum_mass_map,
            'prof': sum_data / n_pix_eff, 'flag': flag}


def ell_prof(fits_name, aper, isophote=ISO, xttools=TBL, pix=1.0,
             ini_sma=15.0, max_sma=175., step=0.2, mode='mean',
             aper_force=None, in_ellip=None):
    """Get Step 2 and Step 3 1-D profile from the stellar mass map.

    Parameters
    ----------
    fits_name : str
        Name of the FITS image to run Ellipse on.
    aper : dict
        Dictionary that contains basic information of the galaxy.
    isophote : str, optional
        Location of the binary executable file: `x_isophote.e`. Default: ISO
    xttools : str, optional
        Location of the binary executable file: `x_ttools.e`. Default: TBL
    pix : float, optional
        Pixel scale. Default: 1.0.
    ini_sma : float, optional
        Initial radii to start the fitting. Default: 15.0.
    max_sma : float, optional
        Maximum fitting radius. Default: 175.0.
    step : float, optional
        Fitting step size foro Ellipse. Default: 0.2.
    mode : str, optional
        Integration mode for Ellipse fitting. Options: ['mean'|'median'|'bi-linear'].
        Default: 'mean'.
    aper_force : dict, optional
        Dictionary that contains external shape information of the galaxy. Default: None
    in_ellip : str, optional
        Input binary table from previous Ellipse run. Default: None

    Returns
    -------
    ell_shape : astropy.table
        Ellipse fitting results for isophotal shape.
    ell_mprof : astropy.table
        Ellipse fitting results for 1-D mass density profile along the major axis.
    bin_shape : str
        Location of the binary table for `ell_shape`.
    bin_mprof : str
        Location of the binary table for `ell_mprof`.

    """
    # Just use the center from the aperture result
    xcen, ycen = aper['x'], aper['y']

    # Initial values of b/a and position angle are from the aperture result too.
    if aper_force is None:
        ba, pa = aper['ba'], aper['pa'] + 90.0
    else:
        ba, pa = aper_force['ba'], aper_force['pa'] + 90.0
    if pa >= 90.0:
        pa = pa - 180.
    elif pa <= -90.0:
        pa = pa + 180.

    try:
        # Step 2 to get ellipticity and position angle profiles
        ell_shape, bin_shape = galSBP.galSBP(
            fits_name, galX=xcen, galY=ycen, maxSma=max_sma, iniSma=ini_sma,
            verbose=False, savePng=False, saveOut=True, expTime=1.0,
            pix=pix, zpPhoto=0.0, galQ=ba, galPA=pa, stage=2,
            minSma=0.0, ellipStep=step, isophote=isophote, xttools=xttools,
            uppClip=2.5, lowClip=3.0, maxTry=5, nClip=2, intMode=mode,
            updateIntens=False, harmonics=True)
        # Add an index array
        ell_shape.add_column(Column(data=np.arange(len(ell_shape)), name='index'))
    except Exception:
        print("# Something went wrong during stage 2 for {}".format(fits_name))
        ell_shape, bin_shape = None, None

    # Update the centroid
    if ell_shape is not None:
        xnew = np.nanmean(ell_shape['x0'][ell_shape['sma'] < (ini_sma * 2)])
        ynew = np.nanmean(ell_shape['y0'][ell_shape['sma'] < (ini_sma * 2)])
    else:
        xnew, ynew = xcen, ycen

    if in_ellip is None:
        try:
            # Step 3 to get mass density profiles
            ell_mprof, bin_mprof = galSBP.galSBP(
                fits_name, galX=xnew, galY=ynew, maxSma=max_sma, iniSma=ini_sma,
                verbose=False, savePng=False, saveOut=True, expTime=1.0,
                pix=pix, zpPhoto=0.0, galQ=ba, galPA=pa, stage=3,
                minSma=0.0, ellipStep=step, isophote=isophote, xttools=xttools,
                uppClip=2.5, lowClip=3.0, maxTry=3, nClip=2, intMode=mode,
                updateIntens=False, harmonics=True)
            # Add an index array
            ell_mprof.add_column(Column(data=np.arange(len(ell_mprof)), name='index'))
        except Exception:
            print("# Something went wrong during stage 3 for {}".format(fits_name))
            ell_mprof, bin_mprof = None, None
    else:
        ell_mprof, bin_mprof = ell_force(
            fits_name, in_ellip, aper, isophote=isophote, xttools=xttools, pix=pix,
            mode=mode)

    return ell_shape, ell_mprof, bin_shape, bin_mprof


def fourier_profile(ellip, r_min=0., r_max=None, pix=1.0):
    """Get the arrays of Fourier amplitudes.

    Parameters
    ----------
    ellip : astropy.table
        Table that contains the Ellipse output.
    r_min : float, optional
        Minimum radii to calculate Fourier amplitudes for. Default: 0.
    r_max : float, optional
        Maximum radii to calculate Fourier amplitudes for. Default: None.
    pix : float, optional
        Pixel scale in unit of kpc / pixel. Default: 1.0.

    Returns
    -------
    fourier_tab : astropy.table
        Astropy table that summaries the Fourier amplitudes.

    """
    # A1
    a1_arr = -100. * ellip['A1'] / ellip['grad'] / ellip['sma']
    a1_err = -100. * ellip['A1_err'] / ellip['grad'] / ellip['sma']
    # B1
    b1_arr = -100. * ellip['B1'] / ellip['grad'] / ellip['sma']
    b1_err = -100. * ellip['B1_err'] / ellip['grad'] / ellip['sma']

    # A2
    a2_arr = -100. * ellip['A2'] / ellip['grad'] / ellip['sma']
    a2_err = -100. * ellip['A2_err'] / ellip['grad'] / ellip['sma']
    # B2
    b2_arr = -100. * ellip['B2'] / ellip['grad'] / ellip['sma']
    b2_err = -100. * ellip['B2_err'] / ellip['grad'] / ellip['sma']

    # A3
    a3_arr = -100. * ellip['A3'] / ellip['grad'] / ellip['sma']
    a3_err = -100. * ellip['A3_err'] / ellip['grad'] / ellip['sma']
    # B3
    b3_arr = -100. * ellip['B3'] / ellip['grad'] / ellip['sma']
    b3_err = -100. * ellip['B3_err'] / ellip['grad'] / ellip['sma']

    # A4
    a4_arr = -100. * ellip['A4'] / ellip['grad'] / ellip['sma']
    a4_err = -100. * ellip['A4_err'] / ellip['grad'] / ellip['sma']
    # B4
    b4_arr = -100. * ellip['B4'] / ellip['grad'] / ellip['sma']
    b4_err = -100. * ellip['B4_err'] / ellip['grad'] / ellip['sma']

    # Limit the radial range if necessary
    sma = ellip['sma']
    if r_max is None:
        r_max = np.max(sma)

    sma_mask = (sma > r_min) & (sma < r_max)

    # Design a table for the Fourier amplitudes
    fourier_tab = Table()
    fourier_tab.add_column(Column(data=sma[sma_mask], name='r_pix'))
    fourier_tab.add_column(Column(data=sma[sma_mask] * pix, name='r_kpc'))
    fourier_tab.add_column(Column(data=a1_arr[sma_mask], name='a1'))
    fourier_tab.add_column(Column(data=a1_err[sma_mask], name='a1_err'))
    fourier_tab.add_column(Column(data=b1_arr[sma_mask], name='b1'))
    fourier_tab.add_column(Column(data=b1_err[sma_mask], name='b1_err'))
    fourier_tab.add_column(Column(data=a2_arr[sma_mask], name='a2'))
    fourier_tab.add_column(Column(data=a2_err[sma_mask], name='a2_err'))
    fourier_tab.add_column(Column(data=b2_arr[sma_mask], name='b2'))
    fourier_tab.add_column(Column(data=b2_err[sma_mask], name='b2_err'))
    fourier_tab.add_column(Column(data=a3_arr[sma_mask], name='a3'))
    fourier_tab.add_column(Column(data=a3_err[sma_mask], name='a3_err'))
    fourier_tab.add_column(Column(data=b3_arr[sma_mask], name='b3'))
    fourier_tab.add_column(Column(data=b3_err[sma_mask], name='b3_err'))
    fourier_tab.add_column(Column(data=a4_arr[sma_mask], name='a4'))
    fourier_tab.add_column(Column(data=a4_err[sma_mask], name='a4_err'))
    fourier_tab.add_column(Column(data=b4_arr[sma_mask], name='b4'))
    fourier_tab.add_column(Column(data=b4_err[sma_mask], name='b4_err'))
    try:
        fourier_tab.add_column(Column(data=ellip['index'][sma_mask], name='index'))
    except KeyError:
        pass

    return np.asarray(fourier_tab)


def ell_force(fits_name, in_ellip, aper, isophote=ISO, xttools=TBL, pix=1.0,
              mode='mean'):
    """Get Step 4 force photometry 1-D profile.

    Parameters
    ----------
    fits_name : str
        Name of the FITS image to run Ellipse on.
    in_ellip : str
        Input binary table from previous Ellipse run.
    aper : dict
        Dictionary that contains basic information of the galaxy.
    isophote : str, optional
        Location of the binary executable file: `x_isophote.e`. Default: ISO
    xttools : str, optional
        Location of the binary executable file: `x_ttools.e`. Default: TBL
    pix : float, optional
        Pixel scale. Default: 1.0.
    mode : str, optional
        Integration mode for Ellipse fitting. Options: ['mean'|'median'|'bi-linear'].
        Default: 'mean'.

    Returns
    -------
    ell_force : astropy.table
        Ellipse fitting results for force photometry mode.
    bin_force : str
        Location of the binary table for `ell_force`.

    """
    # Just use the center from the aperture result
    xcen, ycen = aper['x'], aper['y']

    # Check if the output binary file is available
    if not os.path.isfile(in_ellip):
        raise FileNotFoundError(
            "# Cannot find the input ellipse binary output file: {}".format(in_ellip))

    try:
        # Step 4 to get the "forced"-photometry mode of 1-D profile
        ell_force, bin_force = galSBP.galSBP(
            fits_name, inEllip=in_ellip, galX=xcen, galY=ycen,
            verbose=False, savePng=False, saveOut=True, expTime=1.0,
            pix=pix, zpPhoto=0.0, stage=4, isophote=isophote, xttools=xttools,
            uppClip=3.0, lowClip=3.0, maxTry=2, nClip=2, intMode=mode,
            updateIntens=False, harmonics=True)
        # Add an index array
        ell_force.add_column(Column(data=np.arange(len(ell_force)), name='index'))
    except Exception:
        print("# Something went wrong during stage 4 for {}".format(fits_name))
        ell_force, bin_force = None, None

    return ell_force, bin_force
