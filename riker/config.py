#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configurations of the analysis."""

import yaml

__all__ = [
    'parse_config'
]

DEFAULT_MAP_CONFIG = {
    # About radial bins
    'rad': {
        'n_rad': 15, 'r_min': 0.1, 'r_max': None, 'linear': False,
    },
    'detect': {
        'threshold': 1e8, 'bkg_ratio': 10, 'bkg_filter': 5,
    },
    'aper': {
        'using_gal': True, 'subpix': 5,
    },
    'ellip': {
        'ini_sma': 15.0, 'max_sma': 175.0, 'step': 0.2, 'mode': 'mean',
    }
}


class BeneMassAgeZConfig(object):
    """Configuration parameters to deal with the stellar mass, age, Z maps.

    Parmameters
    -----------
    config_file : str, optional
        A yaml configuration file.

    """

    def __init__(self, config_file=None):
        """Initiate the configuration parameters.
        """
        if config_file is not None:
            self.config = parse_config(config_file)
        else:
            self.config = DEFAULT_MAP_CONFIG

        # Definining radius bins
        self._n_rad = (self.config['rad']['n_rad'] if 'n_rad' in self.config['rad']
                       else DEFAULT_MAP_CONFIG['rad']['n_rad'])
        self._r_min = (self.config['rad']['r_min'] if 'r_min' in self.config['rad']
                       else DEFAULT_MAP_CONFIG['rad']['r_min'])
        self._r_max = (self.config['rad']['r_max'] if 'r_max' in self.config['rad']
                       else DEFAULT_MAP_CONFIG['rad']['r_max'])
        self._linear = (self.config['rad']['linear'] if 'linear' in self.config['rad']
                        else DEFAULT_MAP_CONFIG['rad']['linear'])

        # Parameters about detecting the galaxy
        self._threshold = (
            self.config['detect']['threshold'] if 'threshold' in self.config['detect']
            else DEFAULT_MAP_CONFIG['detect']['threshold'])
        self._bkg_ratio = (
            self.config['detect']['bkg_ratio'] if 'bkg_ratio' in self.config['detect']
            else DEFAULT_MAP_CONFIG['detect']['bkg_ratio'])
        self._bkg_filter = (
            self.config['detect']['bkg_filter'] if 'bkg_filter' in self.config['detect']
            else DEFAULT_MAP_CONFIG['detect']['bkg_filter'])

        # Parameters about the aperture profiles
        self._using_gal = (self.config['aper']['using_gal'] if 'using_gal' in self.config['aper']
                           else DEFAULT_MAP_CONFIG['aper']['using_gal'])
        self._subpix = (self.config['aper']['subpix'] if 'subpix' in self.config['aper']
                        else DEFAULT_MAP_CONFIG['aper']['subpix'])

        # Parameters about the Ellipse profiles
        self._ini_sma = (self.config['ellip']['ini_sma'] if 'ini_sma' in self.config['ellip']
                         else DEFAULT_MAP_CONFIG['ellip']['ini_sma'])
        self._max_sma = (self.config['ellip']['max_sma'] if 'max_sma' in self.config['ellip']
                         else DEFAULT_MAP_CONFIG['ellip']['max_sma'])
        self._step = (self.config['ellip']['step'] if 'step' in self.config['ellip']
                      else DEFAULT_MAP_CONFIG['ellip']['step'])
        self._mode = (self.config['ellip']['mode'] if 'mode' in self.config['ellip']
                      else DEFAULT_MAP_CONFIG['ellip']['mode'])

    @property
    def n_rad(self):
        """Number of radial bins."""
        return self._n_rad

    @n_rad.setter
    def n_rad(self, n_rad):
        self._n_rad = n_rad

    @property
    def r_min(self):
        """Number of radial bins."""
        return self._r_min

    @r_min.setter
    def r_min(self, r_min):
        self._r_min = r_min

    @property
    def r_max(self):
        """Number of radial bins."""
        return self._r_max

    @r_max.setter
    def r_max(self, r_max):
        self._r_max = r_max

    @property
    def linear(self):
        """Number of radial bins."""
        return self._linear

    @linear.setter
    def linear(self, linear):
        self._linear = linear

    @property
    def threshold(self):
        """Number of radial bins."""
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    @property
    def bkg_ratio(self):
        """Number of radial bins."""
        return self._bkg_ratio

    @bkg_ratio.setter
    def bkg_ratio(self, bkg_ratio):
        self._bkg_ratio = bkg_ratio

    @property
    def bkg_filter(self):
        """Number of radial bins."""
        return self._bkg_filter

    @bkg_filter.setter
    def bkg_filter(self, bkg_filter):
        self._bkg_filter = bkg_filter

    @property
    def using_gal(self):
        """Number of radial bins."""
        return self._using_gal

    @using_gal.setter
    def using_gal(self, using_gal):
        self._using_gal = using_gal

    @property
    def subpix(self):
        """Number of radial bins."""
        return self._subpix

    @subpix.setter
    def subpix(self, subpix):
        self._subpix = subpix

    @property
    def ini_sma(self):
        """Number of radial bins."""
        return self._ini_sma

    @ini_sma.setter
    def ini_sma(self, ini_sma):
        self._ini_sma = ini_sma

    @property
    def max_sma(self):
        """Number of radial bins."""
        return self._max_sma

    @max_sma.setter
    def max_sma(self, max_sma):
        self._max_sma = max_sma

    @property
    def step(self):
        """Number of radial bins."""
        return self._step

    @step.setter
    def step(self, step):
        self._step = step

    @property
    def mode(self):
        """Number of radial bins."""
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode


def parse_config(config_file):
    """Parse the `yaml` format configuration file.

    Parameters
    ----------
    config_file : string
        Location and name of the configuration file in `yaml` format.

    Return
    ------
        Configuration parameters in dictionary format.

    """
    return yaml.load(open(config_file))
