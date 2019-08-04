#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deal with galaxy from Illustris or TNG simulation."""

import os

import numpy as np

from . import profile
from . import utils


__all__ = [
    'GalaxyMap',
    'KERNEL'
    ]

# This is the detection kernel used in sep
KERNEL = np.asarray([[0.092163, 0.221178, 0.296069, 0.221178, 0.092163],
                     [0.221178, 0.530797, 0.710525, 0.530797, 0.221178],
                     [0.296069, 0.710525, 0.951108, 0.710525, 0.296069],
                     [0.221178, 0.530797, 0.710525, 0.530797, 0.221178],
                     [0.092163, 0.221178, 0.296069, 0.221178, 0.092163]])


class GalaxyMap(object):
    """Dealing with 2-D projected map of a galaxy.

    Parameters
    ----------
    hdf5 : BeneMassAgeZMaps object
        HDF5 data for all the 2-D maps.
    idx : int
        Index of the galaxy to be analyzed.
    proj : str, optional
        Projection of the 2-D map. Default: 'xy'

    """

    def __init__(self, hdf5, idx, proj='xy'):
        """Gather basic information and all the maps.
        """
        # Parent directory to keep the files
        self.dir = hdf5.dir

        # Prefix for output files
        self.prefix = hdf5.label

        # Pixel scale in unit of kpc per pixel
        self.pix = hdf5.pix

        # Redshift of the snapshot
        self.redshift = hdf5.redshift

        # Index of the galaxy
        self.idx = idx

        # Projection of the 2-D map
        self.proj = proj

        # Gather all the maps and the basic information
        self.info, self.maps = hdf5.get_maps(self.idx, self.proj)

        # Design the radial bins
        # TODO: control this using config dict
        self.rad_bins = None
        self.rad_inn = None
        self.rad_out = None
        self.rad_mid = None
        self.radial_bins(rad=None, n_rad=15, r_min=0.01, r_max=None, linear=False)

        # Placeholder for results
        # Basic information
        self.detect_gal = None
        self.detect_ins = None
        self.detect_exs = None

        # Aperture stellar mass profiles
        self.maper_gal = None
        self.maper_ins = None
        self.maper_exs = None

        # Age profiles
        self.age_prof_gal = None
        self.age_prof_ins = None
        self.age_prof_exs = None

        # Metallicity profiles
        self.met_prof_gal = None
        self.met_prof_ins = None
        self.met_prof_exs = None


    def radial_bins(self, rad=None, n_rad=15, r_min=0.1, r_max=None, linear=False,
                    output=False):
        """Design radial bins to get aperture profiles.

        Parameters
        ----------
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

        """
        # Get the radial bins in unit of kpc
        if rad is None:
            if r_max is None:
                r_max = self.info['img_w'] / 2.0
            if linear:
                rad = np.linspace(r_min, r_max * self.info['pix'], (n_rad + 1))
            else:
                rad = np.logspace(
                    np.log10(r_min), np.log10(r_max * self.info['pix']), (n_rad + 1))

        # Arrays of inner and outer radius
        r_inn, r_out = rad[:-1], rad[1:]

        # Use the mid point of the radial bins
        r_mid = (r_inn + r_out) / 2.0

        setattr(self, 'rad_bins', rad)
        setattr(self, 'rad_inn', r_inn)
        setattr(self, 'rad_out', r_out)
        setattr(self, 'rad_mid', r_mid)

        if output:
            return rad

    def detect(self, map_type, verbose=False, kernel=KERNEL, threshold=1e8,
               bkg_ratio=10, bkg_filter=5, output=False, **detect_kwargs):
        """Detect the galaxy and get basic properties.

        Parameters
        ----------
        map_type : str
            Type of the stellar mass map. Options ['gal'|'ins'|'exs']
        kernel : ndarray, optional
            2-D kernel used for detecting the galaxy. Default: None
        threshold : float, optional
            Mass threshold for detecting the galaxy. Default: 1E8
        bkg_ratio : int, optional
            Ratio between the image size and sky box size. Default: 10
        bkg_filter : int, optional
            Filter size for sky background. Default: 5
        verbose : bool, optional
            Blah, Blah, Blah. Default: False
        output : bool, optional
            Return the `detect` dictionary when True. Default: False

        Return
        ------
        detect : dict
            Dictionary that contains the basic information of this component.

        """
        # Basic geometry information of the galaxy
        # Here we have three configuration parameters:
        # threshold, bkg_ratio, bkg_filter
        detect = profile.detect_galaxy(
            self.info, self.maps['mass_{}'.format(map_type)], kernel=kernel,
            threshold=threshold, bkg_ratio=bkg_ratio, bkg_filter=bkg_filter,
            **detect_kwargs)

        if verbose:
            print("# Detection for {}".format(map_type))
            print(detect)

        setattr(self, 'detect_{}'.format(map_type), detect)

        if output:
            return detect

    def maper(self, map_type, verbose=False, using_gal=True, subpix=5,
              output=False, **detect_kwargs):
        """Get aperture stellar mass curves from the stellar mass map.

        Parameters
        ----------
        map_type : str
            Type of the stellar mass map. Options ['gal'|'ins'|'exs']
        using_gal : bool, optional
            Using the basic information of the whole galaxy. Default: True.
        subpix : int, optional
            Subpixel sampling factor. Default is 5.
        verbose : bool, optional
            Blah, Blah, Blah. Default: False
        output : bool, optional
            Return the `maper` array when True. Default: False

        """
        # Get the basic information
        detect = getattr(self, 'detect_{}'.format(map_type))
        if map_type is not 'gal' and using_gal:
            detect = getattr(self, 'detect_gal')

        # If not basic information is available, run the detection again.
        if detect is None:
            detect = self.detect(
                self.maps['mass_{}'.format(map_type)], output=True, **detect_kwargs)

        # Here we have 5 configuration parameters:
        # n_rad, linear, r_min, r_max, subpix
        _, maper = profile.aperture_masses(
            self.info, self.maps['mass_{}'.format(map_type)],
            detect=detect, rad=self.rad_out, subpix=subpix)

        if verbose:
            print("# Aperture masses for {}".format(map_type))
            print(maper)

        setattr(self, 'maper_{}'.format(map_type), maper)

        if output:
            return maper

    def aprof(self, data_type, map_type, using_gal=True, subpix=5, mask=None,
              output=False, verbose=False, **detect_kwargs):
        """Get the average profiles of a property using pre-defined apertures.

        Parameters
        ----------
        data_type : str
            Galaxy property to be used. `age` for stellar age, `met` for stellar metallicity.
        map_type : str
            Type of the stellar mass map. Options ['gal'|'ins'|'exs']
        using_gal : bool, optional
            Using the basic information of the whole galaxy. Default: True.
        subpix : int, optional
            Subpixel sampling factor. Default: 5.
        mask : ndarray, optional
            Mask array.
        verbose : bool, optional
            Blah, Blah, Blah. Default: False
        output : bool, optional
            Return the `maper` array when True. Default: False

        """
        # Get the basic information
        detect = getattr(self, 'detect_{}'.format(map_type))
        if map_type is not 'gal' and using_gal:
            detect = getattr(self, 'detect_gal')

        # If not basic information is available, run the detection again.
        if detect is None:
            detect = self.detect(
                self.maps['mass_{}'.format(map_type)], output=True, **detect_kwargs)

        prof = profile.mass_weighted_prof(
            self.maps['{}_{}'.format(data_type, map_type)],
            self.maps['mass_{}'.format(map_type)], detect, self.rad_inn, self.rad_out,
            subpix=subpix, mask=mask)

        if verbose:
            print("# {} profile for {}".format(data_type, map_type))
            print(prof)

        setattr(self, "{}_prof_{}".format(data_type, map_type), prof)

        if output:
            return prof

    def aper_summary(self):
        """Get all the stellar mass, age, and metallicity profiles.
        """
        pass

    def maps_to_fits(self, data_type, map_type, folder=None):
        """Save a 2-D map into a FITS file.

        Parameters
        ----------
        data_type : str
            Galaxy property to be used. `age` for stellar age, `met` for stellar metallicity.
        map_type : str
            Type of the stellar mass map. Options ['gal'|'ins'|'exs']
        folder : str, optional
            Output directory name. Default: None

        """
        # Output folder for the FITS file
        if folder is None:
            folder = self.dir

        # Name of the output file
        fits_name = os.path.join(
            folder, "{}_{}_{}_{}_{}.fits".format(
                self.prefix, self.idx, self.proj, data_type, map_type))

        utils.save_to_fits(self.maps['{}_{}'.format(data_type, map_type)], fits_name)
        if not os.path.isfile(fits_name):
            raise FileNotFoundError("# Did not save the FITS file successfully!")

        return fits_name
