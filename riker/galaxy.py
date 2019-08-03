#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deal with galaxy from Illustris or TNG simulation."""

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
