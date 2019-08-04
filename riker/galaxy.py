#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deal with galaxy from Illustris or TNG simulation."""

import numpy as np

from riker import profile

__all__ = ['KERNEL', 'GalaxyMap']

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

        # Placeholder for results
        self.detect_gal = None
        self.detect_ins = None
        self.detect_exs = None

        self.maper_cen = None
        self.maper_ins = None
        self.maper_exs = None

    def detect(self, map_type, verbose=False, kernel=KERNEL, threshold=1e8,
               bkg_ratio=10, bkg_filter=5, **detect_kwargs):
        """Detect the galaxy and get basic properties.

        Parameters
        ----------
        map_type: str
            Type of the stellar mass map. Options ['gal'|'ins'|'exs']
        kernel: ndarray, optional
            2-D kernel used for detecting the galaxy. Default: None
        threshold: float, optional
            Mass threshold for detecting the galaxy. Default: 1E8
        bkg_ratio: int, optional
            Ratio between the image size and sky box size. Default: 10
        bkg_filter: int, optional
            Filter size for sky background. Default: 5

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
