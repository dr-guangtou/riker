#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deal with galaxy from Illustris or TNG simulation."""

import os

import numpy as np

from astropy.table import Table, Column, join

from riker import profile
from riker import utils


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

# Useful columns from the Ellipse output
ELL_COL_USE = [
    'index', 'sma', 'intens', 'int_err', 'ell', 'ell_err', 'pa', 'pa_err',
    'x0', 'x0_err', 'y0', 'y0_err', 'stop', 'tflux_e', 'tflux_c',
    'pa_norm', 'growth_ori'
]


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
            self.info, self.maps["mass_{}".format(map_type)], kernel=kernel,
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
        detect = getattr(self, "detect_{}".format(map_type))
        if map_type is not 'gal' and using_gal:
            detect = getattr(self, 'detect_gal')

        # If not basic information is available, run the detection again.
        if detect is None:
            detect = self.detect(map_type, output=True, **detect_kwargs)

        # Here we have 1 configuration parameter:
        # subpix
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

    def aper_summary(self, gal_only=False, subpix=5, output=False):
        """Get all the stellar mass, age, and metallicity profiles.

        Parameters
        ----------
        gal_only: bool, optional
            Only provide summary of the whole galaxy. Default: False.
        subpix : int, optional
            Subpixel sampling factor. Default is 5.
        output : bool, optional
            Return the `maper` array when True. Default: False

        """
        # Aperture mass profiles
        self.maper('gal', subpix=subpix, using_gal=True)
        if not gal_only:
            self.maper('ins', subpix=subpix, using_gal=True)
            self.maper('exs', subpix=subpix, using_gal=True)

        # Aperture age profiles
        self.aprof('age', 'gal', subpix=subpix, using_gal=True)
        if not gal_only:
            self.aprof('age', 'ins', subpix=subpix, using_gal=True)
            self.aprof('age', 'exs', subpix=subpix, using_gal=True)

        # Aperture metallicity profiles
        self.aprof('met', 'gal', subpix=subpix, using_gal=True)
        if not gal_only:
            self.aprof('met', 'ins', subpix=subpix, using_gal=True)
            self.aprof('met', 'exs', subpix=subpix, using_gal=True)

        # Gather these results into an Astropy Table
        aper_sum = Table()
        aper_sum.add_column(Column(data=self.rad_inn, name='rad_inn'))
        aper_sum.add_column(Column(data=self.rad_out, name='rad_out'))
        aper_sum.add_column(Column(data=self.rad_mid, name='rad_mid'))

        aper_sum.add_column(Column(data=self.maper_gal, name='maper_gal'))
        if not gal_only:
            aper_sum.add_column(Column(data=self.maper_ins, name='maper_ins'))
            aper_sum.add_column(Column(data=self.maper_exs, name='maper_exs'))

        aper_sum.add_column(Column(data=self.age_prof_gal['prof_w'], name='age_gal_w'))
        aper_sum.add_column(Column(data=self.age_prof_gal['prof'], name='age_gal'))
        aper_sum.add_column(Column(data=self.age_prof_gal['flag'], name='age_gal_flag'))
        if not gal_only:
            aper_sum.add_column(Column(data=self.age_prof_ins['prof_w'], name='age_ins_w'))
            aper_sum.add_column(Column(data=self.age_prof_ins['prof'], name='age_ins'))
            aper_sum.add_column(Column(data=self.age_prof_ins['flag'], name='age_ins_flag'))
            aper_sum.add_column(Column(data=self.age_prof_exs['prof_w'], name='age_exs_w'))
            aper_sum.add_column(Column(data=self.age_prof_exs['prof'], name='age_exs'))
            aper_sum.add_column(Column(data=self.age_prof_exs['flag'], name='age_exs_flag'))
        aper_sum.add_column(Column(data=self.met_prof_gal['prof_w'], name='met_gal_w'))
        aper_sum.add_column(Column(data=self.met_prof_gal['prof'], name='met_gal'))
        aper_sum.add_column(Column(data=self.met_prof_gal['flag'], name='met_gal_flag'))

        if not gal_only:
            aper_sum.add_column(Column(data=self.met_prof_ins['prof_w'], name='met_ins_w'))
            aper_sum.add_column(Column(data=self.met_prof_ins['prof'], name='met_ins'))
            aper_sum.add_column(Column(data=self.met_prof_ins['flag'], name='met_ins_flag'))
            aper_sum.add_column(Column(data=self.met_prof_exs['prof_w'], name='met_exs_w'))
            aper_sum.add_column(Column(data=self.met_prof_exs['prof'], name='met_exs'))
            aper_sum.add_column(Column(data=self.met_prof_exs['flag'], name='met_exs_flag'))

        setattr(self, 'aper_sum', aper_sum.as_array())

        if output:
            return aper_sum

    def map_to_fits(self, data_type, map_type, folder=None):
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
            folder, "{}_{}_{}_{}_{}_{}.fits".format(
                self.prefix, self.idx, self.info['catsh_id'],
                self.proj, data_type, map_type))

        utils.save_to_fits(self.maps['{}_{}'.format(data_type, map_type)], fits_name)
        if not os.path.isfile(fits_name):
            raise FileNotFoundError("# Did not save the FITS file successfully!")

        return fits_name

    def ell_summary(self, gal_only=False, output=False):
        """Gather all necessary Ellipse profiles.

        Parameters
        ----------
        gal_only: bool, optional
            Only provide summary of the whole galaxy. Default: False.
        output : bool, optional
            Return the `maper` array when True. Default: False

        """
        # Ellipse run for the whole galaxy
        ell_shape_gal, ell_mprof_gal, bin_shape_gal, bin_mprof_gal = self.ell_prof(
            'gal')
        # TODO
        pass

    def ell_prof(self, map_type, isophote=profile.ISO, xttools=profile.TBL,
                 ini_sma=15.0, max_sma=175.0, step=0.2, mode='mean',
                 remove_bin=False, aper_force=None, in_ellip=None):
        """Run Ellipse on the stellar mass map.

        Parameters
        ----------
        map_type : str
            Type of the stellar mass map. Options ['gal'|'ins'|'exs']
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
        remove_bin : bool, optional
            Remove the output binary file or not. Default: False.
        aper_force : dict, optional
            Dictionary that contains external shape information of the galaxy. Default: None.
        in_ellip : str, optional
            Input binary table from previous Ellipse run. Default: None

        """
        # Save the file to a FITS image
        fits_name = self.map_to_fits('mass', map_type, folder=None)

        # Get the isophotal shape and mass density profiles.
        ell_shape, ell_mprof, bin_shape, bin_mprof = profile.ell_prof(
            fits_name, self.detect_gal, isophote=isophote, xttools=xttools,
            pix=self.info['pix'], ini_sma=ini_sma, max_sma=max_sma, step=step,
            mode=mode, aper_force=aper_force, in_ellip=in_ellip)

        # Clean up a little
        folder, file_name = os.path.split(fits_name)
        utils.clean_after_ellipse(
            folder, file_name.replace('.fits', ''), remove_bin=remove_bin)

        # Calculate the Fourier amplitude information
        fourier_shape = profile.fourier_profile(ell_shape, pix=self.info['pix'])
        fourier_mprof = profile.fourier_profile(ell_mprof, pix=self.info['pix'])

        # Join the useful columns from the Ellipse output with the Fourier amplitudes.
        ell_shape_new = join(
            ell_shape[ELL_COL_USE], fourier_shape, keys='index', join_type='inner')
        ell_mprof_new = join(
            ell_mprof[ELL_COL_USE], fourier_mprof, keys='index', join_type='inner')

        return ell_shape_new, ell_mprof_new, bin_shape, bin_mprof
