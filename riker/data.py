#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deal with the Illustris or TNG maps."""

import os

import h5py
import numpy as np

from astropy.table import Table, Column


__all__ = [
    'BeneMassAgeZMaps',
    'Z_SUN',
    ]

# Solar metallicity
Z_SUN = 0.0134


class BeneMassAgeZMaps(object):
    """Stellar mass, age, metallicity maps provided by Benedikt Diemer.

    Parameters
    ----------
    hdf5_file : str
        Location of the HDF5 file of the map.
    label : str
        Label of this map file. e.g. "tng100_z0.4_high_resolution".
        Default: None

    """

    def __init__(self, hdf5_file, label=None):
        """Read in the HDF5 map file.
        """
        self.hdf5_file = hdf5_file
        directory, file_name = os.path.split(hdf5_file)
        self.dir = directory
        self.hdf5_name = file_name

        # Open and read in the file
        self.data = h5py.File(self.hdf5_file, 'r')
        # TODO
        #self.all = self.get_data(self.hdf5_file)

        # Keys for available data
        self.keys = list(self.data.keys())

        # Get the attributes in the config group
        self.config_maps = self.data['config']
        self.config_keys = [
            key for key in self.config_maps.attrs.keys()]

        # Redshift of the snapshot
        self.redshift = self.get_config_key('snap_z')

        # Pixel scale in unit of kpc/pixel
        self.pix = self.get_pixel_scale()

        # Label the dataset
        if label is not None:
            self.label = label
        else:
            self.label = "{}_z{}".format(self.get_config_key('sim'), self.redshift)

        # Number of galaxies
        self.n_gal = len(self.data['catsh_id'])

    def get_pixel_scale(self):
        """Get the pixel scale of the map."""
        return 2. * self.get_config_key('map_range_min') / self.get_config_key('map_npixel')

    def get_config_key(self, key):
        """Get the key value from the map config."""
        if key in self.config_keys:
            return self.config_maps.attrs[key]
        else:
            print("# Key: {} is not available!".format(key))
            return None
    
    def get_data(self, hdf5_file):
        """Extract all useful data from the HDF5 file.

        Parameters
        ----------
        hdf5_file : str
            Location of the HDF5 file of the map.

        """
        hdf5 = h5py.File(self.hdf5_file, 'r')
        data = {
            'catsh_id': list(hdf5['catsh_id']), 
            'catgrp_is_primary': list(hdf5['catgrp_is_primary']), 
            'scalar_star_mass': list(hdf5['scalar_star_mass']), 
            'catgrp_Group_M_Crit200': list(hdf5['catgrp_Group_M_Crit200']), 
            'scalar_star_age': list(hdf5['scalar_star_age']), 
            'scalar_star_metallicity': list(hdf5['scalar_star_metallicity']), 
            'map_star_rho_insitu_xy': list(hdf5['map_star_rho_insitu_xy']),
            'map_star_rho_insitu_xz': list(hdf5['map_star_rho_insitu_xz']),
            'map_star_rho_insitu_yz': list(hdf5['map_star_rho_insitu_yz']),
            'map_star_rho_exsitu_xy': list(hdf5['map_star_rho_exsitu_xy']),
            'map_star_rho_exsitu_xz': list(hdf5['map_star_rho_exsitu_xz']),
            'map_star_rho_exsitu_yz': list(hdf5['map_star_rho_exsitu_yz']),
            'map_star_age_insitu_xy': list(hdf5['map_star_age_insitu_xy']),
            'map_star_age_insitu_xz': list(hdf5['map_star_age_insitu_xz']),
            'map_star_age_insitu_yz': list(hdf5['map_star_age_insitu_yz']),
            'map_star_age_exsitu_xy': list(hdf5['map_star_age_exsitu_xy']),
            'map_star_age_exsitu_xz': list(hdf5['map_star_age_exsitu_xz']),
            'map_star_age_exsitu_yz': list(hdf5['map_star_age_exsitu_yz']),
            'map_star_metallicity_insitu_xy': list(hdf5['map_star_metallicity_insitu_xy']),
            'map_star_metallicity_insitu_xz': list(hdf5['map_star_metallicity_insitu_xz']),
            'map_star_metallicity_insitu_yz': list(hdf5['map_star_metallicity_insitu_yz']),
            'map_star_metallicity_exsitu_xy': list(hdf5['map_star_metallicity_exsitu_xy']),
            'map_star_metallicity_exsitu_xz': list(hdf5['map_star_metallicity_exsitu_xz']),
            'map_star_metallicity_exsitu_yz': list(hdf5['map_star_metallicity_exsitu_yz'])
        }

        # Close the HDF5 file
        hdf5.close()

        return data

    def sum_table(self, save=False):
        """Put the basic information of all galaxies in a table.

        Parameters
        ----------
        save : bool, optional
            Save the summary table in a npy output. Default: False.

        """
        summary = Table()
        summary.add_column(Column(data=np.arange(self.n_gal), name='index'))
        summary.add_column(Column(data=np.asarray(self.data['catsh_id']), name='catsh_id'))
        summary.add_column(Column(
            data=np.asarray(self.data['catgrp_is_primary']), name='cen_flag'))
        summary.add_column(Column(
            data=np.log10(np.asarray(self.data['scalar_star_mass'])), name='logms'))
        summary.add_column(Column(
            data=np.log10(np.asarray(self.data['catgrp_Group_M_Crit200'])), name='logm200c'))
        summary.add_column(Column(
            data=np.asarray(self.data['scalar_star_age']), name='age'))
        summary.add_column(Column(
            data=np.log10(np.asarray(self.data['scalar_star_metallicity']) / Z_SUN),
            name='metallicity'))

        if save:
            np.save(os.path.join(self.dir, "{}_galaxies.npy".format(self.label)), summary)

        return summary

    def get_basic_info(self, idx):
        """Gather basic information of the galaxy.

        Parameters
        ----------
        idx: int
            Index of the galaxy.

        Return
        ------
        info: dict
            A dictionary that contains basic information of the galaxy.
        """
        return {
            'catsh_id': self.data['catsh_id'][idx],
            'cen_flag': self.data['catgrp_is_primary'][idx],
            'logms': np.log10(self.data['scalar_star_mass'][idx]),
            'logm200c': np.log10(self.data['catgrp_Group_M_Crit200'][idx]),
            'age': self.data['scalar_star_age'][idx],
            'metallicity': self.data['scalar_star_metallicity'][idx],
            'pix':self.pix
        }

    def get_maps(self, idx, proj, verbose=False, maps_only=False):
        """Gather the stellar mass, age, metallicity map.

        Parameters
        ----------
        idx: int
            Index of the galaxy.
        proj: str
            Projection of the map. [xy|xz|yz]

        Return
        ------
        info: dict
            A dictionary that contains basic information of the galaxy.
        maps: dict
            A dictionary that contains all the necessary maps.

        """
        # Basic information about the content
        info = self.get_basic_info(idx)
        if verbose:
            print("\n# Subhalo ID: {}".format(info['catsh_id']))

        # Projection
        if proj not in ['xy', 'xz', 'yz']:
            raise Exception("# Wrong projection: [xy | xz | yz]")
        info['proj'] = proj

        # Get the stellar mass maps
        mass_ins = self.data['map_star_rho_insitu_{}'.format(proj)][idx] * (self.pix ** 2)
        mass_exs = self.data['map_star_rho_exsitu_{}'.format(proj)][idx] * (self.pix ** 2)
        mass_gal = mass_ins + mass_exs

        # Stellar mass on the maps
        info['logms_map_ins'] = np.log10(mass_ins.sum())
        info['logms_map_exs'] = np.log10(mass_exs.sum())
        info['logms_map_gal'] = np.log10(mass_gal.sum())

        # Image size
        img_h, img_w = mass_ins.shape
        info['img_h'] = img_h
        info['img_w'] = img_w
        info['img_cen_x'] = img_h / 2.
        info['img_cen_y'] = img_w / 2.

        if verbose:
            print("\n# log(M*_ins): {:6.2f}".format(info['logms_map_ins']))
            print("# log(M*_exs): {:6.2f}".format(info['logms_map_exs']))
            print("# log(M*_gal): {:6.2f}".format(info['logms_map_gal']))

        # Get the stellar age map
        age_ins = self.data['map_star_age_insitu_{}'.format(proj)][idx]
        age_exs = self.data['map_star_age_exsitu_{}'.format(proj)][idx]
        age_gal = (age_ins * mass_ins + age_exs * mass_exs) / (mass_ins + mass_exs)
        age_ins[age_ins == 0.] = np.nan
        age_exs[age_ins == 0.] = np.nan

        if verbose:
            print("\n# (Age_ins/Gyr): {:6.2f}".format(np.nanmedian(age_ins)))
            print("# (Age_exs/Gyr): {:6.2f}".format(np.nanmedian(age_exs)))

        # Get the stellar metallicity map
        met_ins = self.data['map_star_metallicity_insitu_{}'.format(proj)][idx]
        met_exs = self.data['map_star_metallicity_exsitu_{}'.format(proj)][idx]
        met_gal = (met_ins * mass_ins + met_exs * mass_exs) / (mass_ins + mass_exs)
        met_ins[met_ins == 0.] = np.nan
        met_exs[met_ins == 0.] = np.nan

        if verbose:
            print("# log(Z_ins/Z_sun): {:6.2f}".format(
                np.log10(np.nanmedian(met_ins / Z_SUN))))
            print("# log(Z_exs/Z_sun): {:6.2f}".format(
                np.log10(np.nanmedian(met_exs / Z_SUN))))

        maps = {'mass_ins': mass_ins, 'mass_exs': mass_exs, 'mass_gal': mass_gal,
                'age_ins': age_ins, 'age_exs': age_exs, 'age_gal': age_gal,
                'met_ins': met_ins, 'met_exs': met_exs, 'met_gal': met_gal}
        
        if maps_only:
            return maps

        return info, maps
